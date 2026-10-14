"""Explicit integral AP for small practice tasks; official COCO/VOC graders remain authoritative."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _overlap(prediction: np.ndarray, truth: np.ndarray, masks: bool) -> float:
    if masks:
        if prediction.shape != truth.shape:
            raise ValueError("Predicted and reference masks must use the same image dimensions")
        intersection = np.logical_and(prediction, truth).sum()
        union = np.logical_or(prediction, truth).sum()
    else:
        intersection = np.maximum(
            0, np.minimum(prediction[2:], truth[2:]) - np.maximum(prediction[:2], truth[:2])
        ).prod()
        union = (
            np.maximum(0, prediction[2:] - prediction[:2]).prod()
            + np.maximum(0, truth[2:] - truth[:2]).prod()
            - intersection
        )
    return float(intersection / union) if union else 0.0


def practice_average_precision(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    masks: bool = False,
    mask_threshold: float = 0.5,
) -> float:
    """Mean all-points interpolated AP across classes present in practice ground truth.

    Uses exclusive xyxy boxes, stable descending confidence sorting, one detection per ground
    truth, and optional binary-mask IoU. No crowd/difficult/area/maxDet rules or COCO 101-point
    averaging are implemented. Never substitute this named practice metric for an official grader.
    """
    if len(predictions) != len(targets) or not targets or not 0 < iou_threshold <= 1:
        raise ValueError("Nonempty matching prediction/target lists and valid IoU are required")
    key = "masks" if masks else "boxes"
    truth_by_class: dict[int, dict[int, list[np.ndarray]]] = {}
    candidates: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    for image, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        if any(_array(target.get(flag, [0])).any() for flag in ("iscrowd", "difficult")):
            raise ValueError("Use the official evaluator for crowd or difficult ground truth")
        truth_labels = _array(target["labels"]).tolist()
        pred_labels = _array(prediction["labels"]).tolist()
        scores = _array(prediction["scores"])
        truth_shapes = _array(target[key])
        pred_shapes = _array(prediction[key])
        if masks:
            if pred_shapes.ndim == 4:
                pred_shapes = pred_shapes[:, 0]
            pred_shapes = pred_shapes >= mask_threshold
            truth_shapes = truth_shapes.astype(bool)
        if (
            len(truth_labels) != len(truth_shapes)
            or len(pred_labels) != len(scores)
            or len(scores) != len(pred_shapes)
        ):
            raise ValueError("Detection arrays must have matching instance counts")
        if not np.isfinite(scores).all():
            raise ValueError("Detection confidence must be finite")
        for label, shape in zip(truth_labels, truth_shapes, strict=True):
            truth_by_class.setdefault(int(label), {}).setdefault(image, []).append(shape)
        for label, score, shape in zip(pred_labels, scores, pred_shapes, strict=True):
            candidates.setdefault(int(label), []).append((float(score), image, shape))
    if not truth_by_class:
        raise ValueError("AP needs at least one labeled ground-truth instance")
    averages = []
    for label, images in truth_by_class.items():
        used = {image: set() for image in images}
        positives = sum(len(shapes) for shapes in images.values())
        matches = []
        for _, image, shape in sorted(candidates.get(label, []), key=lambda item: -item[0]):
            overlaps = [_overlap(shape, expected, masks) for expected in images.get(image, [])]
            best = int(np.argmax(overlaps)) if overlaps else -1
            matched = best >= 0 and overlaps[best] >= iou_threshold and best not in used[image]
            matches.append(int(matched))
            if matched:
                used[image].add(best)
        if not matches:
            averages.append(0.0)
            continue
        true_positive = np.cumsum(matches)
        recall = true_positive / positives
        precision = true_positive / np.arange(1, len(matches) + 1)
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        changes = np.flatnonzero(recall[1:] != recall[:-1])
        averages.append(
            float(((recall[changes + 1] - recall[changes]) * precision[changes + 1]).sum())
        )
    return float(np.mean(averages))
