"""Execute the hidden answers and advanced lessons against isolated offline schema fixtures."""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import re
import socket
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import ArrayLike

from olp_ai_26.core.colab import sync_artifacts
from olp_ai_26.cv.segmentation import DiceBCELoss
from olp_ai_26.learning_data import LEARNING_FAMILIES, create_learning_pack
from olp_ai_26.notebooks import build_all_notebooks, percent_script_to_notebook
from olp_ai_26.validation.competition_metrics import (
    log_price_rmse,
    matching_f1,
    salt_score,
    wheat_score,
)

ROOT = Path(__file__).parents[1]
LESSONS = ROOT / "notebooks/learning"


def _notebook(name: str) -> dict:
    return json.loads((LESSONS / f"{name}_template.ipynb").read_text(encoding="utf-8"))


def _expected_micro_code(notebook: dict) -> list[tuple[str, str]]:
    """Recover each supplied/worked cell or baseline answer without executing reveals."""
    cells = notebook["cells"]
    result = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        if "exercise" in cell["metadata"].get("tags", []):
            solution = "".join(cells[index + 1]["source"])
            match = re.search(r"<pre><code>(.*?)</code></pre>", solution, flags=re.DOTALL)
            assert match is not None
            code = unescape(match.group(1))
        else:
            code = "".join(cell["source"])
        stage = next(tag for tag in cell["metadata"]["tags"] if tag.startswith("stage-"))
        result.append((stage, code))
    return result


def _expected_code(notebook: dict) -> list[str]:
    """Group small cells by competition stage for fixture setup and targeted helper checks."""
    stages: dict[str, list[str]] = {}
    for stage, code in _expected_micro_code(notebook):
        stages.setdefault(stage, []).append(code)
    return ["\n\n".join(cells) for cells in stages.values()]


def _configure(code: str, data: Path, output: Path, runtime: Path) -> str:
    values = {
        "DATA_DIR": f"Path({str(data)!r})",
        "OUTPUT": f"Path({str(output)!r})",
        "paths": (
            f"runtime_paths(runtime_dir=Path({str(runtime)!r}), persistent_dir=PERSISTENT_DIR)"
        ),
    }
    lines = code.splitlines(keepends=True)
    assignments = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in values
    ]
    assert len(assignments) == len(values)
    for node in reversed(assignments):
        name = node.targets[0].id
        lines[node.lineno - 1 : node.end_lineno] = [f"{name} = {values[name]}\n"]
    return "".join(lines)


def test_seven_pairs_have_guided_stages_checks_hints_and_collapsed_answers(
    tmp_path,
):
    sources = sorted(LESSONS.glob("*_template.py"))
    assert len(sources) == 14
    split_implementations = set()
    download_implementations = set()
    for source in sources:
        built = percent_script_to_notebook(source, tmp_path / source.with_suffix(".ipynb").name)
        notebook = json.loads(built.read_text(encoding="utf-8"))
        assert notebook == _notebook(source.name.removesuffix("_template.py"))
        cells = notebook["cells"]
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert len(_expected_code(notebook)) == 12  # Stages remain, cell count follows pedagogy.
        assert len(code_cells) > 40
        assert len("".join(code_cells[0]["source"]).splitlines()) <= 35
        assert sum("check" in cell["metadata"]["tags"] for cell in code_cells) >= 10
        configuration = [cell for cell in code_cells if "configuration" in cell["metadata"]["tags"]]
        assert configuration
        assert all(
            not cell["metadata"].get("jupyter", {}).get("source_hidden") for cell in configuration
        )
        download_settings = next(
            cell for cell in code_cells if "DOWNLOAD_DATA = False" in "".join(cell["source"])
        )
        assert "configuration" in download_settings["metadata"]["tags"]
        assert len({cell["id"] for cell in cells}) == len(cells)
        for _, code in _expected_micro_code(notebook):
            compile(code, source.name, "exec")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "make_split":
                    split_implementations.add(ast.dump(node, include_attributes=False))
                if isinstance(node, ast.FunctionDef) and node.name == "download_practice_data":
                    download_implementations.add(ast.dump(node, include_attributes=False))
                if isinstance(node, ast.ImportFrom):
                    assert node.level == 0 and not node.module.startswith("olp_ai_26")
                if isinstance(node, ast.Import):
                    assert all(not item.name.startswith("olp_ai_26") for item in node.names)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in {"exec", "eval", "__import__"}
            assert "PROJECT_ROOT" not in code
            assert "sys.path" not in code
            assert "_colab_bootstrap" not in code
        if "_baseline_" in source.name:
            text = "\n".join("".join(cell["source"]) for cell in cells)
            assert "Hint 1" in text and "Hint 2" in text
            assert "Stop and explain" in text and "Transfer and closed-book" in text
            assert (
                sum("raise NotImplementedError" in "".join(cell["source"]) for cell in code_cells)
                == 8
            )
            for index, cell in enumerate(cells):
                if cell["cell_type"] == "code":
                    solution = cells[index + 1]
                    assert solution["cell_type"] == "markdown"
                    assert solution["metadata"]["collapsed"]
                    assert solution["metadata"]["tags"] == ["solution"]
                    assert "<details>" in "".join(solution["source"])
                    assert "<details open" not in "".join(solution["source"])
                    compile("".join(cell["source"]), source.name, "exec")
                    if "exercise" in cell["metadata"]["tags"]:
                        assert len("".join(cell["source"]).splitlines()) <= 65
        else:
            assert not any(
                "raise NotImplementedError" in "".join(cell["source"]) for cell in code_cells
            )
        for cell in code_cells:
            if "provided" in cell["metadata"]["tags"]:
                assert cell["metadata"]["jupyter"]["source_hidden"]
    assert len(split_implementations) == 1, "Standalone split copies must retain identical guards"
    assert len(download_implementations) == 1, "Every downloader must retain identical guards"


def test_recursive_builder_includes_learning_subdirectory(tmp_path):
    nested = tmp_path / "learning"
    nested.mkdir()
    (nested / "example_template.py").write_text("# %%\nx = 1\n", encoding="utf-8")
    assert build_all_notebooks(tmp_path) == [nested / "example_template.ipynb"]


@pytest.mark.parametrize("fragment", ["# >>> exercise: missing end\nx = 1\n", "# <<< exercise\n"])
def test_malformed_exercise_boundaries_fail_export(tmp_path, fragment):
    source = tmp_path / "bad_template.py"
    source.write_text("# %% [exercise]\n" + fragment, encoding="utf-8")
    with pytest.raises(ValueError):
        percent_script_to_notebook(source, tmp_path / "bad.ipynb")


@pytest.mark.parametrize(
    "family,function_name,broken",
    [
        ("01_credit", "credit_imbalance", lambda *args: (None, 0)),
        ("02_prices", "log_training_target", lambda values: np.zeros_like(values)),
        ("03_toxic", "score_multilabel", lambda *args: ({}, 0)),
        ("04_leaves", "one_batch_step", lambda *args: 0.0),
        ("05_salt", "select_mask_threshold", lambda *args: ({}, 0.3, 1.0)),
        ("06_wheat", "select_confidence", lambda *args: ({}, 0.2, 0)),
        (
            "07_matching",
            "match_positions",
            lambda features, *args, **kwargs: [[index] for index in range(len(features))],
        ),
    ],
)
def test_embedded_feedback_rejects_plausible_broken_exercises(
    tmp_path, family, function_name, broken
):
    """Passing examples must not reward no-op updates, wrong metrics or self-only retrieval."""
    notebook = _notebook(f"{family}_baseline")
    check = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "check" in cell["metadata"].get("tags", [])
        and f"{function_name}(" in "".join(cell["source"])
    )
    namespace = {
        "np": np,
        "pd": pd,
        "torch": torch,
        "nn": torch.nn,
        "OUTPUT": tmp_path,
        function_name: broken,
    }
    with pytest.raises(AssertionError):
        exec(compile(check, "exercise-feedback", "exec"), namespace)


@pytest.mark.parametrize(
    "family,variant,asset_mode",
    [
        (family, variant, "cache")
        for variant in ("baseline", "advanced")
        for family in LEARNING_FAMILIES
    ]
    + [("07_matching", "advanced", "encoders")],
)
def test_learning_solutions_execute_offline_through_reload_and_delivery(
    tmp_path, monkeypatch, family, variant, asset_mode
):
    matplotlib = pytest.importorskip(
        "matplotlib", reason="Execution uses the optional notebook group"
    )
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)  # Figure files are still rendered and checked.
    if family == "01_credit" and variant == "advanced":
        pytest.importorskip(
            "lightgbm", reason="Advanced credit uses the optional kaggle-extra group"
        )

    def forbidden_network(*args, **kwargs):
        raise AssertionError("A learning notebook attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    pack = create_learning_pack(tmp_path / "pack", family=family)
    with pytest.raises(FileExistsError):
        create_learning_pack(pack, family=family)
    output = tmp_path / "candidate"
    notebook = _notebook(f"{family}_{variant}")
    code = _expected_code(notebook)
    code[0] = _configure(code[0], pack, output, tmp_path / "runtime")
    # A copy of the notebook is the only lesson artifact in the working directory.
    isolated = tmp_path / "standalone"
    isolated.mkdir()
    (isolated / "lesson.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    monkeypatch.chdir(isolated)
    original_import = builtins.__import__

    def without_repository(name, *args, **kwargs):
        if name == "olp_ai_26" or name.startswith("olp_ai_26."):
            raise AssertionError("Standalone notebook tried to import the project")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_repository)
    namespace = {"__name__": "__main__"}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        exec(compile(code[0], "lesson:setup", "exec"), namespace)
        namespace.update(IMAGE_SIZE=64, BATCH_SIZE=2, EPOCHS=1, IMAGE_SUFFIX=".png")
        if family == "07_matching" and variant == "advanced":
            # Untrained deterministic fixture vectors, independent of all task labels.
            train = pd.read_csv(pack / "train.csv")
            test = pd.read_csv(pack / "test.csv")
            rows = pd.concat([train, test], ignore_index=True)
            rng = np.random.default_rng(42)
            features = rng.normal(size=(len(rows), 8)).astype(np.float32)
            cache = tmp_path / "frozen-fixture-features.npz"
            np.savez_compressed(
                cache,
                ids=rows["posting_id"].to_numpy(dtype=str),
                image=features,
                text=rng.normal(size=(len(rows), 6)).astype(np.float32),
                fit_ids=np.asarray([], dtype=str),
            )
            namespace.update(
                PRETRAINED_ALLOWED=True,
                ASSET_PROVENANCE="Untrained synthetic vectors; no task-label fitting",
                EMBEDDING_ARCHIVE=cache,
            )
            namespace["competition"].pretrained_allowed = True
            if asset_mode == "encoders":
                import timm
                from tokenizers import Tokenizer
                from tokenizers.models import WordLevel
                from tokenizers.pre_tokenizers import Whitespace
                from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

                vocabulary = {"[PAD]": 0, "[UNK]": 1, "brand": 2, "product": 3}
                vocabulary.update({f"item{i}": i + 4 for i in range(20)})
                tokenizer_core = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
                tokenizer_core.pre_tokenizer = Whitespace()
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=tokenizer_core,
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                )
                text_dir = tmp_path / "permitted-untrained-text-fixture"
                tokenizer.save_pretrained(text_dir)
                BertModel(
                    BertConfig(
                        vocab_size=len(vocabulary),
                        hidden_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        intermediate_size=32,
                        max_position_embeddings=64,
                    )
                ).save_pretrained(text_dir)
                image_checkpoint = tmp_path / "permitted-untrained-image-fixture.pt"
                torch.save(
                    timm.create_model("resnet18", pretrained=False).state_dict(), image_checkpoint
                )
                namespace.update(
                    EMBEDDING_ARCHIVE=None,
                    TEXT_MODEL_DIR=text_dir,
                    IMAGE_MODEL_CHECKPOINT=image_checkpoint,
                    ASSET_PROVENANCE="Local untrained tiny BERT and ResNet fixture; no task labels",
                )
        for index, (stage, cell) in enumerate(_expected_micro_code(notebook), 1):
            if stage != "stage-1":  # The configured setup was already executed above.
                exec(compile(cell, f"{family}:{variant}:{stage}:cell-{index}", "exec"), namespace)
    finally:
        torch.set_num_threads(previous_threads)
    for filename in (
        "submission.csv",
        "run.json",
        "fold_ids.json",
        "technical_report.md",
        "lesson_config.json",
        "inspection.png",
        "imbalance.png",
        "validation.png",
    ):
        assert (output / filename).is_file(), filename
    assert namespace["checkpoint_path"].is_file()
    submission, sample = namespace["submission"], namespace["sample"]
    id_column = namespace["ID_COLUMN"]
    assert submission[id_column].tolist() == sample[id_column].tolist()
    assert submission.columns.tolist() == sample.columns.tolist()
    folds = json.loads((output / "fold_ids.json").read_text(encoding="utf-8"))
    assert not set(folds["train_ids"]) & set(folds["validation_ids"])
    assert np.isfinite(namespace["validation_score"])
    assert namespace["fit_seconds"] >= 0 and namespace["inference_seconds"] >= 0


@pytest.mark.parametrize("implementation", ["package", "baseline", "advanced"])
def test_competition_metric_fixtures_and_coordinate_conventions(implementation):
    metrics = {
        "log_price_rmse": log_price_rmse,
        "salt_score": salt_score,
        "wheat_score": wheat_score,
        "matching_f1": matching_f1,
    }
    if implementation != "package":
        namespace = {"np": np, "Sequence": Sequence, "ArrayLike": ArrayLike}
        for family, name in zip(
            ("02_prices", "05_salt", "06_wheat", "07_matching"), metrics, strict=True
        ):
            for cell in _expected_code(_notebook(f"{family}_{implementation}")):
                for node in ast.parse(cell).body:
                    if isinstance(node, ast.FunctionDef) and node.name == name:
                        function = ast.Module(body=[node], type_ignores=[])
                        exec(compile(function, "standalone-metric", "exec"), namespace)
            metrics[name] = namespace[name]
    check_log, check_salt, check_wheat, check_matching = metrics.values()
    assert check_log([1, np.exp(2)], [np.e, np.e]) == pytest.approx(1)
    with pytest.raises(ValueError):
        check_log([1], [0])
    truth = np.ones((1, 2, 2), dtype=bool)
    prediction = truth.copy()
    prediction[0, 0, 0] = False  # IoU=.75: exactly five strict threshold hits.
    assert check_salt(truth, prediction) == 0.5
    assert check_salt(np.zeros_like(truth), np.zeros_like(truth)) == 1
    assert check_salt(truth, np.zeros_like(truth)) == 0
    gt = [np.array([[0, 0, 2, 2]])]
    duplicates = [np.array([[0, 0, 2, 2], [0, 0, 2, 2]])]
    assert check_wheat(gt, duplicates, [[0.9, 0.8]]) == 0.5
    assert check_wheat(gt, gt, [[0.5]], score_threshold=0.5) == 0
    assert check_wheat([[]], [[]], [[]]) == 0
    boundary_gt, boundary_pred = [[[0, 0, 2, 1]]], [[[0, 0, 3, 1]]]
    assert check_wheat(boundary_gt, boundary_pred, [[0.9]]) == pytest.approx(5 / 6)
    assert check_wheat(boundary_gt, boundary_pred, [[0.9]], strict_iou=False) == 1
    assert check_wheat(
        gt, [np.array([[0, 0, 3, 3]])], [[0.9]], inclusive_pixels=True
    ) == pytest.approx(1 / 3)
    assert check_wheat(gt, [np.array([[0, 0, 3, 3]])], [[0.9]], inclusive_pixels=False) == 0
    assert check_matching(["a", "a", "b"], [[0], [1], [2]]) == pytest.approx(7 / 9)
    assert check_matching(["a", "a", "b"], [[0, 1], [0, 1], [2]]) == 1


@pytest.mark.parametrize("family", ["01_credit", "04_leaves"])
def test_copied_notebook_runs_in_fresh_isolated_python_process(tmp_path, family):
    pytest.importorskip("matplotlib")
    pack = create_learning_pack(tmp_path / "data", family=family)
    output = tmp_path / "candidate"
    cells = _expected_code(_notebook(f"{family}_baseline"))
    cells[0] = _configure(cells[0], pack, output, tmp_path / "runtime")
    work = tmp_path / "copied-notebook-only"
    work.mkdir()
    (work / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    runner = work / "run_lesson.py"
    runner.write_text(
        """import builtins, json, socket
from pathlib import Path
original_import = builtins.__import__
def without_project(name, *args, **kwargs):
    if name == 'olp_ai_26' or name.startswith('olp_ai_26.'):
        raise AssertionError('Project import attempted')
    return original_import(name, *args, **kwargs)
def without_network(*args, **kwargs):
    raise AssertionError('Network connection attempted')
builtins.__import__ = without_project
socket.socket.connect = without_network
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.show = lambda: None
import torch
torch.set_num_threads(2)
cells = json.loads(Path('cells.json').read_text(encoding='utf-8'))
for index, cell in enumerate(cells):
    exec(compile(cell, f'copied-notebook:stage-{index+1}', 'exec'), globals())
    if index == 0:
        IMAGE_SIZE, BATCH_SIZE, EPOCHS = 64, 2, 1
assert checkpoint_path.is_file()
assert submission[ID_COLUMN].tolist() == sample[ID_COLUMN].tolist()
print('STANDALONE_DELIVERY_OK')
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", str(runner)],
        cwd=work,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]
    assert "STANDALONE_DELIVERY_OK" in result.stdout
    assert (output / "submission.csv").is_file()


def test_standalone_storage_and_chronological_split_guards(tmp_path):
    """Exercise the inline copies, including newly independent storage and time boundaries."""
    pytest.importorskip("matplotlib")
    cells = _expected_code(_notebook("01_credit_baseline"))
    namespace = {"__name__": "__main__"}
    exec(
        compile(
            _configure(cells[0], tmp_path / "data", tmp_path / "output", tmp_path / "runtime"),
            "standalone-setup",
            "exec",
        ),
        namespace,
    )
    split_cell = next(cell for cell in cells if "def make_split(" in cell)
    for node in ast.parse(split_cell).body:
        if isinstance(node, (ast.ImportFrom, ast.Import, ast.FunctionDef, ast.ClassDef)):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "inline-split", "exec"), namespace
            )
    frame = pd.DataFrame({"time": ["2026-01-01"] * 8 + ["2026-01-02"] * 2})
    split = namespace["make_split"](frame, time_column="time", valid_size=0.3)
    assert len(split.train) == 8 and len(split.valid) == 2
    assert split.train["time"].max() < split.valid["time"].min()
    with pytest.raises(ValueError, match="distinct times"):
        namespace["make_split"](frame.iloc[:8], time_column="time")
    with pytest.raises(ValueError, match="complete timestamps"):
        namespace["make_split"](pd.DataFrame({"time": [None, "2026-01-01"]}), time_column="time")
    with pytest.raises(RuntimeError, match="Group leakage"):
        namespace["make_split"](
            frame.assign(person="same"), time_column="time", group_column="person"
        )
    import zipfile

    archive = tmp_path / "escaping.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "must not escape")
    with pytest.raises(ValueError, match="outside the destination"):
        namespace["stage_data"](archive, tmp_path / "staged")
    assert not (tmp_path / "escaped.txt").exists()
    output = tmp_path / "output"
    sync_definition = next(
        node
        for code in cells
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "sync_artifacts"
    )
    exec(
        compile(ast.Module(body=[sync_definition], type_ignores=[]), "inline-sync", "exec"),
        namespace,
    )
    with pytest.raises(ValueError, match="separate"):
        namespace["sync_artifacts"](output, output / "nested")
    with pytest.raises(ValueError, match="reserve"):
        namespace["TimeBudget"](10, reserve_minutes=10)


def _download_lesson(tmp_path: Path, family: str) -> dict:
    pytest.importorskip("matplotlib")
    cells = _expected_code(_notebook(f"{family}_baseline"))
    namespace = {"__name__": "__main__"}
    exec(
        compile(
            _configure(cells[0], tmp_path / "dataset", tmp_path / "output", tmp_path / "runtime"),
            "download:setup",
            "exec",
        ),
        namespace,
    )
    exec(compile(cells[1], "download:disabled-by-default", "exec"), namespace)
    assert namespace["DOWNLOAD_DATA"] is False
    return namespace


@pytest.mark.parametrize(
    "family,slug",
    [
        ("01_credit", "home-credit-default-risk"),
        ("02_prices", "house-prices-advanced-regression-techniques"),
        ("03_toxic", "jigsaw-toxic-comment-classification-challenge"),
        ("04_leaves", "cassava-leaf-disease-classification"),
        ("05_salt", "tgs-salt-identification-challenge"),
        ("06_wheat", "global-wheat-detection"),
        ("07_matching", "shopee-product-matching"),
    ],
)
def test_optional_download_prepares_original_layout_and_reuses_data_and_cache(
    tmp_path, monkeypatch, family, slug
):
    calls = []

    def fake_cli(command, **options):
        calls.append(command)
        assert command[:6] == [sys.executable, "-m", "kaggle", "competitions", "download", slug]
        assert options["stdin"] == subprocess.DEVNULL and options["capture_output"]
        assert options["timeout"] == 17 and "shell" not in options
        cache = Path(command[command.index("--path") + 1])
        with zipfile.ZipFile(cache / f"{slug}.zip", "w") as outer:
            for path in pack.rglob("*"):
                if path.is_file() and "grading" not in path.relative_to(pack).parts:
                    relative = path.relative_to(pack).as_posix()
                    if family == "05_salt" and relative.startswith(("train/", "test/")):
                        continue
                    if family == "03_toxic" and relative.endswith(".csv"):
                        payload = io.BytesIO()
                        with zipfile.ZipFile(payload, "w") as inner:
                            inner.write(path, relative)
                        outer.writestr(relative + ".zip", payload.getvalue())
                    else:
                        outer.write(path, relative)
            if family == "05_salt":
                for split in ("train", "test"):
                    payload = io.BytesIO()
                    with zipfile.ZipFile(payload, "w") as inner:
                        for path in (pack / split).rglob("*"):
                            if path.is_file():
                                inner.write(path, path.relative_to(pack / split).as_posix())
                    outer.writestr(split + ".zip", payload.getvalue())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_cli)
    namespace = _download_lesson(tmp_path, family)
    assert not calls  # Disabled acquisition performs no credential access or subprocess call.
    assert namespace["KAGGLE_COMPETITION"] == slug
    for variant in ("baseline", "advanced"):
        code = _expected_code(_notebook(f"{family}_{variant}"))[1]
        constants = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(code).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"DOWNLOAD_DATA", "KAGGLE_COMPETITION", "EXPECTED_DATA_PATHS"}
        }
        assert constants["DOWNLOAD_DATA"] is False
        assert constants["KAGGLE_COMPETITION"] == slug
        assert constants["EXPECTED_DATA_PATHS"] == namespace["EXPECTED_DATA_PATHS"]
    monkeypatch.setitem(namespace, "find_spec", lambda name: object())
    pack = create_learning_pack(tmp_path / "fixture", family=family)
    download = namespace["download_practice_data"]
    destination, cache = namespace["DATA_DIR"], namespace["DOWNLOAD_CACHE"]
    result = download(
        slug, destination, cache, namespace["EXPECTED_DATA_PATHS"], timeout_seconds=17
    )
    assert result == destination
    assert len(calls) == 1
    for path in namespace["EXPECTED_DATA_PATHS"]:
        assert (destination / path).exists(), path
    marker = json.loads((destination / ".kaggle-source.json").read_text(encoding="utf-8"))
    assert marker["competition"] == slug
    assert not list(destination.rglob("*.zip"))
    download(slug, destination, cache, namespace["EXPECTED_DATA_PATHS"])
    assert len(calls) == 1
    restored = tmp_path / "restored-from-cache"
    download(slug, restored, cache, namespace["EXPECTED_DATA_PATHS"])
    assert len(calls) == 1  # A valid cached bundle needs neither credentials nor a client.
    pd.testing.assert_frame_equal(
        pd.read_csv(destination / "sample_submission.csv"),
        pd.read_csv(pack / "sample_submission.csv"),
    )
    (destination / "sentinel.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        download("wrong-competition", destination, tmp_path / "other-cache", ("train.csv",))
    assert (destination / "sentinel.txt").read_text(encoding="utf-8") == "preserve"


def test_optional_download_rejects_traversal_missing_layout_and_redacts_client_errors(
    tmp_path, monkeypatch, capsys
):
    namespace = _download_lesson(tmp_path, "02_prices")
    download = namespace["download_practice_data"]
    destination, cache = namespace["DATA_DIR"], namespace["DOWNLOAD_CACHE"]
    slug = namespace["KAGGLE_COMPETITION"]
    expected = namespace["EXPECTED_DATA_PATHS"]
    cache.mkdir(parents=True)
    bundle = cache / f"{slug}.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escaped.txt", "must not escape staging")
    with pytest.raises(RuntimeError, match="not replaced") as failure:
        download(slug, destination, cache, expected)
    assert isinstance(failure.value.__cause__, ValueError)
    assert not destination.exists() and not (destination.parent / "escaped.txt").exists()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("unrelated.csv", "not the task schema")
    with pytest.raises(RuntimeError) as failure:
        download(slug, destination, cache, expected)
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert not destination.exists()
    bundle.unlink()
    monkeypatch.setitem(namespace, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="uv sync"):
        download(slug, destination, cache, expected)
    monkeypatch.setitem(namespace, "find_spec", lambda name: object())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="secret-token", stderr="signed-url-with-secret"
        ),
    )
    with pytest.raises(RuntimeError, match="Check credentials") as failure:
        download(slug, destination, cache, expected)
    combined = str(failure.value) + str(capsys.readouterr())
    assert "secret-token" not in combined and "signed-url-with-secret" not in combined
    assert not destination.exists()
    with pytest.raises(ValueError, match="separate"):
        download(slug, destination, destination / "cache", expected)
    with pytest.raises(ValueError, match="slug"):
        download("invalid slug; command", destination, cache, expected)


def test_positive_mask_weight_changes_loss_and_keeps_ignored_gradient_zero():
    logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
    truth = torch.tensor([[[[1.0, 255.0]]]])
    weighted = DiceBCELoss(0, ignore_index=255, pos_weight=torch.tensor(3.0))
    loss = weighted(logits, truth)
    assert loss.item() == pytest.approx(3 * np.log(2))
    loss.backward()
    assert logits.grad[0, 0, 0, 0].item() == pytest.approx(-1.5)
    assert logits.grad[0, 0, 0, 1].item() == 0


def test_learning_feature_cache_and_visual_reports_survive_sync(tmp_path):
    source, destination = tmp_path / "run", tmp_path / "durable"
    source.mkdir()
    np.savez(source / "features.npz", image=np.ones((2, 3)))
    (source / "inspection.png").write_bytes(b"fixture")
    copied = sync_artifacts(source, destination)
    assert {path.name for path in copied} == {"features.npz", "inspection.png"}
    np.testing.assert_array_equal(np.load(destination / "features.npz")["image"], np.ones((2, 3)))


def test_learning_candidates_keep_separate_durable_bundles(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    pack = create_learning_pack(tmp_path / "pack", family="01_credit")
    persistent = tmp_path / "durable"
    folders = []
    for index in range(2):
        code = _expected_code(_notebook("01_credit_baseline"))
        code[0] = _configure(code[0], pack, tmp_path / f"run-{index}", tmp_path / "runtime")
        code[0] = code[0].replace(
            "DATA_ARCHIVE, PERSISTENT_DIR = None, None",
            f"DATA_ARCHIVE, PERSISTENT_DIR = None, Path({str(persistent)!r})",
        )
        namespace = {"__name__": "__main__"}
        for cell in code:
            exec(compile(cell, "durable-candidate", "exec"), namespace)
        destination = namespace["DURABLE_OUTPUT"]
        assert (destination / "submission.csv").read_bytes() == (
            namespace["OUTPUT"] / "submission.csv"
        ).read_bytes()
        assert (destination / "pipeline.pkl").is_file()
        folders.append(destination)
    assert folders[0] != folders[1]
    assert set(persistent.iterdir()) == set(folders)
