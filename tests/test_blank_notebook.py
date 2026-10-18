"""Exercise reconstruction answer keys without calling the reusable training kit."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from olp_ai_26.practice import create_practice_pack

REPOSITORY = Path(__file__).parents[1]
ANSWER_KEY = REPOSITORY / "scripts/blank_baseline.py"
SPEC = importlib.util.spec_from_file_location("blank_baseline", ANSWER_KEY)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


@pytest.mark.parametrize("task", ["text", "tabular"])
def test_standalone_answer_key_delivery_and_fresh_process_reload(tmp_path, task):
    pack = create_practice_pack(tmp_path / "pack", task=task)
    output = tmp_path / "candidate"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ANSWER_KEY),
            str(pack / "data"),
            str(output),
            "--task",
            task,
            "--metric",
            "macro_f1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    submission = pd.read_csv(output / "submission.csv", converters={"id": str, "label": str})
    sample = pd.read_csv(pack / "data/sample_submission.csv", converters={"id": str})
    answers = pd.read_csv(pack / "grading/answers.csv", converters={"id": str, "label": str})
    assert submission["id"].tolist() == sample["id"].tolist()
    truth = answers.set_index("id")["label"].reindex(submission["id"])
    assert (truth.to_numpy() == submission["label"].to_numpy()).mean() >= 0.8
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert not set(report["train_ids"]) & set(report["validation_ids"])
    # Isolated interpreter, outside the repository: infer using only a trusted saved pipeline.
    recovery = """
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
root, output, task = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
test = pd.read_csv(root / 'test.csv', converters={'id': str})
saved = pd.read_csv(output / 'submission.csv', converters={'id': str, 'label': str})
recipe = json.loads((output / 'run.json').read_text(encoding='utf-8'))
with (output / 'pipeline.pkl').open('rb') as handle:
    model = pickle.load(handle)
features = test['text'].fillna('') if task == 'text' else test[recipe['feature_columns']]
actual = pd.Series(model.predict(features), index=test['id']).reindex(saved['id'])
np.testing.assert_array_equal(actual.to_numpy(), saved['label'].to_numpy())
"""
    recovered = subprocess.run(
        [sys.executable, "-I", "-c", recovery, str(pack / "data"), str(output), task],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert recovered.returncode == 0, recovered.stderr


def test_answer_key_preserves_literal_ids_labels_and_prediction_order(tmp_path):
    (tmp_path / "train.csv").write_text(
        "id,text,label\n001,happy,NA\n002,sad,001\n003,happy,NA\n004,sad,001\n",
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text("id,text\nNA,happy\n010,sad\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,label\n010,\nNA,\n", encoding="utf-8")
    train, test, sample = baseline.read_tables(tmp_path)
    assert train["id"].iloc[0] == "001" and train["label"].iloc[0] == "NA"
    path = baseline.deliver(sample, test["id"], ["NA", "001"], tmp_path)
    restored = pd.read_csv(path, converters={"id": str, "label": str})
    assert restored["id"].tolist() == ["010", "NA"]
    assert restored["label"].tolist() == ["001", "NA"]


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "c"], ["a"]])
def test_answer_key_rejects_ambiguous_or_incomplete_predictions(tmp_path, ids):
    sample = pd.DataFrame({"id": ["b", "a"], "label": ["", ""]})
    with pytest.raises(ValueError):
        baseline.deliver(sample, pd.Series(ids), ["x"] * len(ids), tmp_path)
    assert not (tmp_path / "submission.csv").exists()


def test_answer_key_group_and_chronological_splits_respect_boundaries():
    train = pd.DataFrame(
        {
            "label": ["a", "b"] * 10,
            "person": np.repeat(np.arange(10), 2),
            "timestamp": np.repeat(pd.date_range("2026-01-01", periods=10), 2),
        }
    ).sample(frac=1, random_state=7)
    a, b = baseline.split_rows(train, "group", "person", 42)
    assert not set(train.iloc[a]["person"]) & set(train.iloc[b]["person"])
    a, b = baseline.split_rows(train, "time", "timestamp", 42)
    assert train.iloc[a]["timestamp"].max() < train.iloc[b]["timestamp"].min()


def test_image_document_cells_execute_through_checkpoint_reload_and_delivery(tmp_path):
    pack = create_practice_pack(tmp_path / "pack", task="image")
    source = (REPOSITORY / "docs/BLANK_NOTEBOOK_CV.md").read_text(encoding="utf-8")
    cells = re.findall(r"```python\n(.*?)\n```", source, flags=re.DOTALL)
    assert len(cells) == 4
    namespace = {
        "__name__": "__main__",
        "DATA_DIR": pack / "data",
        "OUTPUT_DIR": tmp_path / "candidate",
        "METRIC": "macro_f1",
        "SPLIT": "stratified",
        "SPLIT_COLUMN": None,
        "np": np,
        "json": json,
        "read_tables": baseline.read_tables,
        "split_rows": baseline.split_rows,
        "score_labels": baseline.score_labels,
        "deliver": baseline.deliver,
    }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for index, cell in enumerate(cells):
            exec(compile(cell, f"blank-cv:cell-{index}", "exec"), namespace)
    finally:
        torch.set_num_threads(previous_threads)
    output = tmp_path / "candidate"
    assert (output / "best.pt").is_file()
    sample = pd.read_csv(pack / "data/sample_submission.csv")
    submission = pd.read_csv(output / "submission.csv")
    assert submission["id"].tolist() == sample["id"].tolist()
    assert submission["label"].notna().all()
    answers = pd.read_csv(pack / "grading/answers.csv").set_index("id")["label"]
    truth = answers.reindex(submission["id"]).to_numpy()
    assert (submission["label"].to_numpy() == truth).mean() >= 0.8
    recipe = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert not set(recipe["train_ids"]) & set(recipe["validation_ids"])
    assert np.isfinite(recipe["validation_score"])
    assert not recipe["pretrained_allowed"]
