from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

MetricFunction = Callable[..., float]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    function: MetricFunction
    maximize: bool


def _f1(average: str) -> MetricFunction:
    return lambda y_true, y_pred, **_: float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )


def _dice(y_true: Any, y_pred: Any, threshold: float = 0.5, **_: Any) -> float:
    truth = np.asarray(y_true).astype(bool)
    prediction = np.asarray(y_pred) >= threshold
    intersection = np.logical_and(truth, prediction).sum()
    return float((2 * intersection + 1e-7) / (truth.sum() + prediction.sum() + 1e-7))


def _iou(y_true: Any, y_pred: Any, threshold: float = 0.5, **_: Any) -> float:
    truth = np.asarray(y_true).astype(bool)
    prediction = np.asarray(y_pred) >= threshold
    intersection = np.logical_and(truth, prediction).sum()
    union = np.logical_or(truth, prediction).sum()
    return float((intersection + 1e-7) / (union + 1e-7))


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
    "macro_f1": MetricSpec(_f1("macro"), True),
    "micro_f1": MetricSpec(_f1("micro"), True),
    "weighted_f1": MetricSpec(_f1("weighted"), True),
    "mse": MetricSpec(lambda y, p, **_: float(mean_squared_error(y, p)), False),
    "rmse": MetricSpec(lambda y, p, **_: float(mean_squared_error(y, p) ** 0.5), False),
    "mae": MetricSpec(lambda y, p, **_: float(mean_absolute_error(y, p)), False),
    "dice": MetricSpec(_dice, True),
    "iou": MetricSpec(_iou, True),
    "bleu": MetricSpec(_bleu, True),
    "chrf": MetricSpec(_chrf, True),
    "rouge_l": MetricSpec(_rouge_l, True),
    "psnr": MetricSpec(_psnr, True),
    "ssim": MetricSpec(_ssim, True),
    "seqeval_f1": MetricSpec(_seqeval_f1, True),
}


def get_metric(name: str) -> MetricSpec:
    try:
        return METRICS[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(METRICS))
        raise KeyError(f"Unknown metric '{name}'. Available: {available}") from error


def evaluate_metric(name: str, y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    return get_metric(name).function(y_true, y_pred, **kwargs)
