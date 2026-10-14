"""Registry of common classification, regression, vision, and generation metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

MetricFunction = Callable[..., float]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Metric callable plus whether larger values are better."""

    function: MetricFunction
    maximize: bool


def _f1(average: str) -> MetricFunction:
    def score(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
        kwargs.setdefault("zero_division", 0)
        return float(f1_score(y_true, y_pred, average=average, **kwargs))

    return score


def binary_mask_score(
    y_true: Any,
    y_pred: Any,
    *,
    metric: str = "dice",
    threshold: float = 0.5,
    reduction: str = "global",
    empty_score: float = 1.0,
    ignore_index: int | None = None,
) -> float:
    """Score binary masks with explicit global or first-axis per-image averaging.

    Empty unions use ``empty_score``; ignored pixels are excluded from both masks. Inputs must
    have identical shapes. This is not a multiclass metric or a substitute for an official grader.
    """
    truth_values, scores = np.asarray(y_true), np.asarray(y_pred)
    if truth_values.shape != scores.shape or truth_values.size == 0:
        raise ValueError("Masks must be nonempty and have identical shapes")
    if not np.isfinite(scores).all():
        raise ValueError("Mask predictions must be finite")
    keep = np.ones(truth_values.shape, dtype=bool)
    if ignore_index is not None:
        keep = truth_values != ignore_index
    if not np.isin(truth_values[keep], [0, 1]).all():
        raise ValueError("Binary mask targets must contain only 0, 1, or ignore_index")
    truth = truth_values.astype(bool) & keep
    prediction = (scores >= threshold) & keep
    if reduction == "global":
        axes = None
    elif reduction == "mean_image" and truth.ndim >= 2:
        axes = tuple(range(1, truth.ndim))
    else:
        raise ValueError("reduction must be global or mean_image with a batch axis")
    intersection = np.logical_and(truth, prediction).sum(axis=axes)
    if metric == "dice":
        numerator = 2 * intersection
        denominator = truth.sum(axis=axes) + prediction.sum(axis=axes)
    elif metric == "iou":
        numerator = intersection
        denominator = np.logical_or(truth, prediction).sum(axis=axes)
    else:
        raise ValueError("metric must be dice or iou")
    values = np.full(np.shape(denominator), empty_score, dtype=float)
    np.divide(numerator, denominator, out=values, where=denominator != 0)
    return float(np.mean(values))


def _dice(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    return binary_mask_score(y_true, y_pred, metric="dice", **kwargs)


def _iou(y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    return binary_mask_score(y_true, y_pred, metric="iou", **kwargs)


def _bleu(y_true: list[str], y_pred: list[str], **_: Any) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(y_pred, [y_true]).score)


def _chrf(y_true: list[str], y_pred: list[str], **_: Any) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_chrf(y_pred, [y_true]).score)


def _rouge_l(y_true: list[str], y_pred: list[str], **_: Any) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for reference, prediction in zip(y_true, y_pred, strict=True)
    ]
    return float(np.mean(scores))


def _psnr(y_true: Any, y_pred: Any, data_range: float = 1.0, **_: Any) -> float:
    return float(
        peak_signal_noise_ratio(np.asarray(y_true), np.asarray(y_pred), data_range=data_range)
    )


def _ssim(y_true: Any, y_pred: Any, data_range: float = 1.0, **kwargs: Any) -> float:
    return float(
        structural_similarity(
            np.asarray(y_true),
            np.asarray(y_pred),
            data_range=data_range,
            **kwargs,
        )
    )


def _seqeval_f1(y_true: list[list[str]], y_pred: list[list[str]], **_: Any) -> float:
    from seqeval.metrics import f1_score as sequence_f1_score

    return float(sequence_f1_score(y_true, y_pred))


METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(lambda y, p, **_: float(accuracy_score(y, p)), True),
    "balanced_accuracy": MetricSpec(
        lambda y, p, **k: float(balanced_accuracy_score(y, p, **k)), True
    ),
    "roc_auc": MetricSpec(lambda y, p, **k: float(roc_auc_score(y, p, **k)), True),
    "average_precision": MetricSpec(
        lambda y, p, **k: float(average_precision_score(y, p, **k)), True
    ),
    "log_loss": MetricSpec(lambda y, p, **k: float(log_loss(y, p, **k)), False),
    "macro_f1": MetricSpec(_f1("macro"), True),
    "micro_f1": MetricSpec(_f1("micro"), True),
    "weighted_f1": MetricSpec(_f1("weighted"), True),
    "mse": MetricSpec(lambda y, p, **_: float(mean_squared_error(y, p)), False),
    "rmse": MetricSpec(lambda y, p, **_: float(mean_squared_error(y, p) ** 0.5), False),
    "mae": MetricSpec(lambda y, p, **_: float(mean_absolute_error(y, p)), False),
    "dice": MetricSpec(_dice, True),
    "iou": MetricSpec(_iou, True),
    "dice_mean_image": MetricSpec(lambda y, p, **k: _dice(y, p, reduction="mean_image", **k), True),
    "iou_mean_image": MetricSpec(lambda y, p, **k: _iou(y, p, reduction="mean_image", **k), True),
    "bleu": MetricSpec(_bleu, True),
    "chrf": MetricSpec(_chrf, True),
    "rouge_l": MetricSpec(_rouge_l, True),
    "psnr": MetricSpec(_psnr, True),
    "ssim": MetricSpec(_ssim, True),
    "seqeval_f1": MetricSpec(_seqeval_f1, True),
}


def get_metric(name: str) -> MetricSpec:
    """Resolve a case-insensitive metric name and explain valid alternatives on failure."""
    try:
        return METRICS[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(METRICS))
        raise KeyError(f"Unknown metric '{name}'. Available: {available}") from error


def evaluate_metric(name: str, y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    """Evaluate a registered metric with optional metric-specific keyword arguments."""
    return get_metric(name).function(y_true, y_pred, **kwargs)
