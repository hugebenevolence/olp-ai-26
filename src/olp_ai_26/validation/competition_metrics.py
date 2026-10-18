"""Explicit historical Kaggle contracts used in the learning notebooks, not COCO substitutes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def log_price_rmse(truth: ArrayLike, prediction: ArrayLike) -> float:
    """House Prices RMSE of natural logarithms; unlike RMSLE, there is no added one."""
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not y.size or not np.isfinite([y, p]).all():
        raise ValueError("Expected aligned nonempty finite prices")
    if np.any(y <= 0) or np.any(p <= 0):
        raise ValueError("Log-price evaluation requires strictly positive prices")
    return float(np.sqrt(np.mean((np.log(y) - np.log(p)) ** 2)))


def salt_score(truth: ArrayLike, prediction: ArrayLike) -> float:
    """TGS single binary mask per image: mean hits at strict IoU thresholds .50:.05:.95.

    Inputs are equally shaped [N,H,W] binary masks. Both empty scores one, exactly one empty
    scores zero. This is not pixel Dice, continuous mean IoU, or instance COCO average precision.
    """
    y, p = np.asarray(truth), np.asarray(prediction)
    if y.ndim != 3 or y.shape != p.shape or not y.size:
        raise ValueError("Expected aligned nonempty [N,H,W] masks at original resolution")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Threshold probability maps before scoring binary masks")
    intersection = np.logical_and(y, p).sum(axis=(1, 2))
    union = np.logical_or(y, p).sum(axis=(1, 2))
    iou = np.divide(intersection, union, out=np.ones(len(y)), where=union != 0)
    thresholds = np.arange(50, 100, 5) / 100
    return float((iou[:, None] > thresholds).mean())


def wheat_score(
    targets: Sequence[ArrayLike],
    boxes: Sequence[ArrayLike],
    scores: Sequence[ArrayLike],
    *,
    score_threshold: float = 0.5,
    inclusive_pixels: bool = True,
    strict_iou: bool = True,
) -> float:
    """Wheat reference convention: confidence-ordered greedy matching, mean .50:.05:.75.

    Absolute xyxy boxes; the linked winning implementation uses inclusive (+1) pixel areas.
    Expose that choice because other graders use continuous exclusive coordinates. Compare
    fixtures with the actual official grader before transferring this metric to a new contest.
    The task statement uses strict IoU comparisons and scores every no-target image zero.
    Set strict_iou=False only to compare with the linked winner's non-strict implementation.
    This averages TP/(TP+FP+FN) over thresholds then images, not confidence-integrated AP.
    """
    if not len(targets) or not (len(targets) == len(boxes) == len(scores)):
        raise ValueError("Need one target, prediction and confidence array per image")
    image_scores = []
    offset = 1.0 if inclusive_pixels else 0.0
    for references, predictions, confidences in zip(targets, boxes, scores, strict=True):
        gt = np.asarray(references, dtype=float).reshape(-1, 4)
        predicted = np.asarray(predictions, dtype=float).reshape(-1, 4)
        confidence = np.asarray(confidences, dtype=float)
        if confidence.shape != (len(predicted),) or not np.isfinite(confidence).all():
            raise ValueError("One finite confidence per predicted box is required")
        for rectangles in (gt, predicted):
            if not np.isfinite(rectangles).all() or np.any(rectangles[:, 2:] <= rectangles[:, :2]):
                raise ValueError("Boxes must be finite positive-area xyxy rectangles")
        if not len(gt):
            image_scores.append(0.0)
            continue
        keep = np.argsort(-confidence, kind="stable")
        predicted = predicted[keep[confidence[keep] > score_threshold]]
        low = np.maximum(predicted[:, None, :2], gt[None, :, :2])
        high = np.minimum(predicted[:, None, 2:], gt[None, :, 2:])
        intersection = np.maximum(high - low + offset, 0).prod(axis=2)
        area_p = (predicted[:, 2:] - predicted[:, :2] + offset).prod(axis=1)
        area_y = (gt[:, 2:] - gt[:, :2] + offset).prod(axis=1)
        iou = intersection / np.maximum(area_p[:, None] + area_y[None, :] - intersection, 1e-12)
        threshold_scores = []
        for threshold in np.arange(50, 80, 5) / 100:
            available = np.ones(len(gt), dtype=bool)
            matched = 0
            for row in iou:
                matched_iou = row > threshold if strict_iou else row >= threshold
                candidates = np.where(available & matched_iou)[0]
                if len(candidates):
                    best = candidates[np.argmax(row[candidates])]
                    available[best] = False
                    matched += 1
            denominator = len(gt) + len(predicted) - matched
            threshold_scores.append(matched / denominator if denominator else 1.0)
        image_scores.append(np.mean(threshold_scores))
    return float(np.mean(image_scores))


def matching_f1(groups: Sequence[object], matches: Sequence[Sequence[int]]) -> float:
    """Shopee mean per-query set F1, including the query itself in its group and prediction."""
    labels = np.asarray(groups)
    if not len(labels) or len(labels) != len(matches):
        raise ValueError("Provide one match set per gallery item")
    values = []
    for index, predictions in enumerate(matches):
        candidate = set(predictions)
        if not candidate or any(not 0 <= item < len(labels) for item in candidate):
            raise ValueError("Match sets must contain valid gallery positions")
        relevant = set(np.flatnonzero(labels == labels[index]).tolist())
        values.append(2 * len(candidate & relevant) / (len(candidate) + len(relevant)))
    return float(np.mean(values))
