from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from olp_ai_26.core.config import CompetitionConfig, TimeBudget, TrainerConfig
from olp_ai_26.core.metrics import evaluate_metric, get_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, validate_submission
from olp_ai_26.validation.leakage import cross_split_overlap, duplicate_text_groups


def test_configs_validate_and_resolve_paths(tmp_path):
    config = CompetitionConfig(data_dir=tmp_path, output_dir=tmp_path / "out", device="cpu")
    config.prepare()
    assert config.output_dir.exists()
    assert config.device == "cpu"
    with pytest.raises(ValueError):
        TrainerConfig(epochs=0)


def test_time_budget_reserve():
    budget = TimeBudget(minutes=0.001, reserve_minutes=0.001)
    time.sleep(0.002)
    assert budget.should_stop
    with pytest.raises(TimeoutError):
        budget.checkpoint("training")


def test_stratified_split_is_deterministic():
    frame = pd.DataFrame({"id": range(40), "label": [0, 1] * 20})
    first = make_split(frame, target_columns="label", seed=7)
    second = make_split(frame, target_columns="label", seed=7)
    assert np.array_equal(first.valid_indices, second.valid_indices)
    assert set(first.valid["label"]) == {0, 1}


def test_group_split_has_no_group_overlap():
    frame = pd.DataFrame(
        {
            "id": range(24),
            "label": [0, 1] * 12,
            "person": [f"p{index // 3}" for index in range(24)],
        }
    )
    split = make_split(frame, target_columns="label", group_column="person", valid_size=0.25)
    assert set(split.train["person"]).isdisjoint(split.valid["person"])


def test_metrics_registry():
    assert evaluate_metric("accuracy", [0, 1], [0, 1]) == 1.0
    assert evaluate_metric("dice", [[1, 0]], [[0.9, 0.1]]) == pytest.approx(1.0)
    assert get_metric("rmse").maximize is False
    image = np.zeros((8, 8), dtype=float)
    assert evaluate_metric("ssim", image, image) == pytest.approx(1.0)


def test_submission_checks_schema_and_id_order():
    sample = pd.DataFrame({"id": [1, 2], "label": [0, 0]})
    submission = build_submission(sample, {"label": [1, 0]})
    assert validate_submission(submission, sample, id_columns="id").valid
    invalid = submission.iloc[::-1].reset_index(drop=True)
    result = validate_submission(invalid, sample, id_columns="id")
    assert not result.valid
    assert any("ID values/order" in error for error in result.errors)


def test_text_duplicate_and_overlap_detection():
    frame = pd.DataFrame({"text": ["Xin  chào", "xin chào", "khác"]})
    assert duplicate_text_groups(frame, "text") == [[0, 1]]
    overlap = cross_split_overlap(frame.iloc[:1], frame.iloc[1:], "text")
    assert overlap.index.tolist() == [1]
