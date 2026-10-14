"""Behavioral regressions for preparation correctness, delivery, and cold reload."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from olp_ai_26.cli import main
from olp_ai_26.core.artifacts import save_json, write_single_file_zip
from olp_ai_26.core.config import TrainerConfig
from olp_ai_26.core.inference import weighted_ensemble
from olp_ai_26.core.inspect import load_table
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.submission import build_submission, validate_submission, write_submission
from olp_ai_26.core.trainer import Trainer, local_hf_trainer_class
from olp_ai_26.cv.classification import build_image_classifier
from olp_ai_26.cv.evaluation import practice_average_precision
from olp_ai_26.cv.segmentation import DiceBCELoss, SegmentationTableDataset, restore_binary_masks
from olp_ai_26.cv.weights import local_checkpoint
from olp_ai_26.ml.tabular import TabularBaseline, TabularConfig
from olp_ai_26.nlp.classification import TfidfTextClassifier
from olp_ai_26.nlp.seq2seq import generate_texts
from olp_ai_26.nlp.token_classification import predict_word_tags
from olp_ai_26.practice import create_practice_pack


def test_predictions_follow_shuffled_composite_ids():
    sample = pd.DataFrame({"id": ["A", "B", "A"], "rank": [2, 1, 1], "value": [0, 0, 0]})
    keys = pd.DataFrame({"id": ["A", "A", "B"], "rank": [1, 2, 1]})
    submission = build_submission(
        sample, {"value": [10, 20, 30]}, prediction_ids=keys, id_columns=["id", "rank"]
    )
    assert submission["value"].tolist() == [20, 30, 10]
    assert validate_submission(submission, sample, id_columns=["id", "rank"]).valid
    assert not validate_submission(submission, sample, id_columns=["id", "rank"]).warnings


@pytest.mark.parametrize("ids", [["A", "A"], ["A", "C"], ["A", None], ["A", ""], ["A"]])
def test_alignment_rejects_incomplete_or_ambiguous_keys(ids):
    sample = pd.DataFrame({"id": ["A", "B"], "label": [0, 0]})
    with pytest.raises(ValueError):
        build_submission(sample, {"label": [1, 0]}, prediction_ids=ids, id_columns="id")


def test_failed_csv_write_preserves_previous_submission(tmp_path, monkeypatch):
    destination = tmp_path / "submission.csv"
    destination.write_text("previous valid submission", encoding="utf-8")

    def interrupted_write(self, path, **kwargs):
        Path(path).write_text("partial", encoding="utf-8")
        raise OSError("simulated interruption")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted_write)
    with pytest.raises(OSError):
        write_submission(pd.DataFrame({"id": [1]}), destination)
    assert destination.read_text(encoding="utf-8") == "previous valid submission"
    assert list(tmp_path.iterdir()) == [destination]


def test_submission_rejects_infinite_scores_and_missing_label_columns():
    sample = pd.DataFrame({"id": [1], "score": [0.0]})
    candidate = pd.DataFrame({"id": [1], "score": [np.inf]})
    assert not validate_submission(candidate, sample).valid
    assert not validate_submission(sample, sample, allowed_labels={"absent": [0]}).valid


def test_duplicate_columns_fail_validation_without_crashing():
    duplicate = pd.DataFrame([[1, 2]], columns=["label", "label"])
    assert not validate_submission(duplicate, duplicate, allowed_labels={"label": [1]}).valid


def test_csv_ids_and_permitted_empty_masks_survive_cli_validation(tmp_path):
    sample = tmp_path / "sample.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,rle\nNA,\n001,\n", encoding="utf-8")
    submission.write_text("id,rle\nNA,\n001,1 1\n", encoding="utf-8")
    assert load_table(submission, id_columns=["id"])["id"].tolist() == ["NA", "001"]
    assert (
        main(
            [
                "validate-submission",
                str(submission),
                str(sample),
                "--id-column",
                "id",
                "--allow-empty-column",
                "rle",
            ]
        )
        == 0
    )
    assert main(["validate-submission", str(submission), str(sample), "--id-column", "id"]) == 1


def test_binary_mask_reductions_match_hand_calculation():
    truth = np.array([[1] * 100, [1] + [0] * 99])
    predictions = np.array([[1] * 100, [0] * 100])
    assert evaluate_metric("dice", truth, predictions) == pytest.approx(200 / 201)
    assert evaluate_metric("dice_mean_image", truth, predictions) == pytest.approx(0.5)
    assert evaluate_metric("iou_mean_image", [[0, 0]], [[0, 0]], empty_score=0) == 0
    assert evaluate_metric("dice", [1, 255], [1, 0], ignore_index=255) == 1
    with pytest.raises(ValueError):
        evaluate_metric("dice", [[1, 0]], [1, 0])


def test_probability_metrics_use_scores_and_direction():
    assert evaluate_metric("roc_auc", [0, 1], [0.1, 0.9]) == 1
    assert evaluate_metric("average_precision", [0, 1], [0.1, 0.9]) == 1
    assert evaluate_metric("log_loss", [0, 1], [[0.9, 0.1], [0.1, 0.9]]) == pytest.approx(
        -np.log(0.9)
    )


def test_accumulation_tail_clips_and_advances_scheduler(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    trainer = Trainer(
        model,
        nn.MSELoss(),
        TrainerConfig(gradient_accumulation_steps=4, max_grad_norm=0.01),
        output_dir=tmp_path,
        device="cpu",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = SimpleNamespace(calls=0)

    def step():
        scheduler.calls += 1

    scheduler.step = step
    loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)), batch_size=1)
    trainer._train_epoch(loader, optimizer, scheduler)
    assert model.weight.item() == pytest.approx(0.001, abs=1e-7)
    assert scheduler.calls == 1


def test_uneven_microbatches_match_a_single_mean_gradient(tmp_path):
    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.0], [0.0], [2.0]])
    results = []
    for batch_size, accumulation in [(3, 1), (2, 4)]:
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        trainer = Trainer(
            model,
            nn.MSELoss(),
            TrainerConfig(
                gradient_accumulation_steps=accumulation, max_grad_norm=None, scheduler="none"
            ),
            output_dir=tmp_path,
            device="cpu",
        )
        trainer._train_epoch(
            DataLoader(TensorDataset(inputs, targets), batch_size=batch_size),
            torch.optim.SGD(model.parameters(), lr=0.1),
            None,
        )
        results.append(model.weight.item())
    assert results[0] == pytest.approx(results[1])


def test_expired_budget_does_not_load_an_older_run(tmp_path):
    model = nn.Linear(1, 1)
    before = model.weight.detach().clone()
    stale = tmp_path / "best.pt"
    stale.write_bytes(b"old checkpoint, deliberately incompatible")
    trainer = Trainer(
        model,
        nn.MSELoss(),
        TrainerConfig(),
        output_dir=tmp_path,
        device="cpu",
        time_budget=SimpleNamespace(should_stop=True),
    )
    loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)))
    with pytest.raises(TimeoutError):
        trainer.fit(loader, loader)
    assert torch.equal(model.weight, before)
    assert stale.read_bytes() == b"old checkpoint, deliberately incompatible"


def test_unlabeled_segmentation_preprocessing_matches_validation(tmp_path):
    Image.new("RGB", (10, 12), (20, 60, 90)).save(tmp_path / "image.png")
    Image.new("L", (10, 12), 255).save(tmp_path / "mask.png")
    frame = pd.DataFrame({"image": ["image.png"], "mask": ["mask.png"]})
    labeled, _ = SegmentationTableDataset(
        frame, image_column="image", mask_column="mask", root=tmp_path, size=16
    )[0]
    unlabeled = SegmentationTableDataset(frame, image_column="image", root=tmp_path, size=16)[0]
    torch.testing.assert_close(labeled, unlabeled)
    mask = restore_binary_masks(
        torch.full((1, 1, 16, 16), 10.0), frame, image_column="image", root=tmp_path
    )[0]
    assert mask.shape == (12, 10)
    assert mask.all()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("model", ["linear", "hist_gradient_boosting"])
def test_tabular_reload_unseen_categories_and_missing_features(tmp_path, task, model):
    frame = pd.DataFrame({"numeric": np.arange(40, dtype=float), "category": ["a", "b"] * 20})
    frame.loc[3, "numeric"] = np.nan
    targets = np.array([0, 1] * 20) if task == "classification" else np.arange(40, dtype=float)
    pipeline = TabularBaseline(TabularConfig(task=task, model=model, max_iter=5)).fit(
        frame, targets
    )
    test = pd.DataFrame({"category": ["never seen", None], "numeric": [3.0, np.nan]})
    before = pipeline.predict(test)
    after = TabularBaseline.load(pipeline.save(tmp_path / "pipeline.pkl")).predict(test)
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(before).all()
    with pytest.raises(ValueError, match="Missing feature"):
        pipeline.predict(test.drop(columns="numeric"))


def test_tfidf_pipeline_reload_preserves_original_labels(tmp_path):
    model = TfidfTextClassifier(min_df=1).fit(
        ["good happy", "great happy", "bad sad", "awful sad"], ["yes", "yes", "no", "no"]
    )
    texts = ["happy", "sad", None]
    np.testing.assert_array_equal(
        model.predict(texts),
        TfidfTextClassifier.load(model.save(tmp_path / "text.pkl")).predict(texts),
    )


def test_invalid_ensemble_weights_are_rejected():
    with pytest.raises(ValueError):
        weighted_ensemble([np.ones(2), np.zeros(2)], [-1, 2])
    with pytest.raises(ValueError):
        weighted_ensemble([np.ones(2)], [np.nan])


def test_practice_ap_confidence_order_and_duplicate_matching():
    target = {"boxes": np.array([[0, 0, 2, 2]]), "labels": np.array([1])}
    prediction = {
        "boxes": np.array([[5, 5, 7, 7], [0, 0, 2, 2], [0, 0, 2, 2]]),
        "labels": np.array([1, 1, 1]),
        "scores": np.array([0.9, 0.8, 0.7]),
    }
    assert practice_average_precision([prediction], [target]) == pytest.approx(0.5)
    assert (
        practice_average_precision([{**prediction, "scores": np.array([0.7, 0.9, 0.8])}], [target])
        == 1
    )


def test_pretrained_cv_request_requires_local_file(tmp_path):
    with pytest.raises(ValueError, match="Stage"):
        local_checkpoint(True, None)
    with pytest.raises(FileNotFoundError):
        local_checkpoint(True, tmp_path / "missing.pt")


def test_synthetic_packs_permute_ids_and_refuse_overwrite(tmp_path):
    root = create_practice_pack(tmp_path / "pack", task="text")
    test = pd.read_csv(root / "data/test.csv")
    sample = pd.read_csv(root / "data/sample_submission.csv")
    assert set(sample["id"]) == set(test["id"])
    assert sample["id"].tolist() != test["id"].tolist()
    assert "label" not in test
    assert (root / "grading/answers.csv").exists()
    with pytest.raises(FileExistsError):
        create_practice_pack(root)


def test_json_artifacts_retain_numeric_labels(tmp_path):
    path = save_json({"labels": [np.int64(3), np.int64(7)]}, tmp_path / "run.json")
    assert json.loads(path.read_text(encoding="utf-8"))["labels"] == [3, 7]


def test_ignored_pixels_do_not_affect_mask_loss_or_gradients():
    logits = torch.tensor([[[[-1.0, 1000.0]]]], requires_grad=True)
    truth = torch.tensor([[[[1.0, 255.0]]]])
    loss = DiceBCELoss(ignore_index=255)(logits, truth)
    expected = DiceBCELoss()(logits[..., :1], truth[..., :1])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[..., 1].item() == 0


def test_staged_timm_backbone_adapts_head_without_network(tmp_path, monkeypatch):
    import timm

    original = timm.create_model("resnet18", pretrained=False)
    checkpoint = tmp_path / "permitted-original.pt"
    torch.save(original.state_dict(), checkpoint)

    def forbidden_download(*args, **kwargs):
        raise AssertionError("Weight loading attempted network access")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden_download)
    model = build_image_classifier(
        "resnet18", 3, pretrained_allowed=True, checkpoint_path=checkpoint
    )
    assert model.get_classifier().out_features == 3
    torch.testing.assert_close(model.conv1.weight, original.conv1.weight)


def test_ner_overflow_keeps_one_first_subword_tag_per_word():
    class Encoding(dict):
        def word_ids(self, batch_index):
            return [[None, 0, 1, 1, None], [None, 1, 2, None, None]][batch_index]

    def tokenizer(tokens, **kwargs):
        return Encoding(
            input_ids=torch.tensor([[0, 2, 1, 2, 0], [0, 2, 0, 0, 0]]),
            overflow_to_sample_mapping=torch.tensor([0, 0]),
            offset_mapping=torch.tensor(
                [
                    [[0, 0], [0, 3], [0, 3], [3, 6], [0, 0]],
                    [[0, 0], [3, 6], [0, 3], [0, 0], [0, 0]],
                ]
            ),
        )

    class Model(nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(
                logits=torch.nn.functional.one_hot(input_ids, num_classes=3).float()
            )

    predictions = predict_word_tags(
        Model(), tokenizer, [["abc", "defghi", "jkl"]], label_names=["O", "B-X", "I-X"]
    )
    assert predictions == [["I-X", "B-X", "O"]]


def test_unlabeled_generation_does_not_call_supervised_forward():
    class Tokenizer:
        def __call__(self, sources, **kwargs):
            assert all(text.startswith("translate: ") for text in sources)
            return {"input_ids": torch.ones(len(sources), 2, dtype=torch.long)}

        def batch_decode(self, token_ids, **kwargs):
            return ["generated" for row in token_ids]

    class Model(nn.Module):
        def forward(self, **kwargs):
            raise AssertionError("Unlabeled generation must not call supervised forward")

        def generate(self, input_ids, **kwargs):
            assert kwargs == {"max_length": 12, "max_new_tokens": None, "num_beams": 1}
            return input_ids

    assert (
        generate_texts(
            Model(),
            Tokenizer(),
            ["one", "two", "three"],
            device="cpu",
            source_prefix="translate: ",
            max_target_length=12,
            batch_size=2,
        )
        == ["generated"] * 3
    )


def test_single_file_submission_zip_preserves_exact_contract(tmp_path):
    source = tmp_path / "candidate.csv"
    source.write_text("id,label\n001,1\n", encoding="utf-8")
    destination = write_single_file_zip(
        source, tmp_path / "submission.zip", arcname="task_output.csv"
    )
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["task_output.csv"]
        assert archive.read("task_output.csv") == source.read_bytes()
    previous = destination.read_bytes()
    with pytest.raises(FileNotFoundError):
        write_single_file_zip(tmp_path / "missing.csv", destination)
    assert destination.read_bytes() == previous


@pytest.mark.parametrize("invalid_keys", [False, True])
def test_hf_best_reload_preserves_wrapped_model_and_rejects_missing_keys(invalid_keys):
    class Model(nn.Linear):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "own-evaluated-checkpoint"
            assert kwargs == {"local_files_only": True, "output_loading_info": True}
            restored = cls(1, 1)
            restored.weight.data.fill_(7)
            return restored, {"missing_keys": ["weight"] if invalid_keys else []}

    trainer_class = local_hf_trainer_class()
    trainer = object.__new__(trainer_class)
    original = Model(1, 1)
    original.weight.data.zero_()
    trainer.model = original
    trainer.state = SimpleNamespace(best_model_checkpoint="own-evaluated-checkpoint")
    if invalid_keys:
        with pytest.raises(RuntimeError, match="does not match"):
            trainer._load_best_model()
        assert original.weight.item() == 0
    else:
        trainer._load_best_model()
        assert trainer.model is original
        assert original.weight.item() == 7
