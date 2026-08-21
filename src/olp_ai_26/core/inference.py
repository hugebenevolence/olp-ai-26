from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


def _inputs(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Mapping):
        values = {key: value for key, value in batch.items() if key != "labels"}
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in values.items()
        }
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    return batch.to(device) if hasattr(batch, "to") else batch


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    device: str = "cuda",
    transforms: list[Callable[[Any], Any]] | None = None,
) -> np.ndarray:
    resolved = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved).eval()
    transforms = transforms or [lambda value: value]
    outputs: list[np.ndarray] = []
    for batch in loader:
        inputs = _inputs(batch, resolved)
        augmented = []
        for transform in transforms:
            transformed = transform(inputs)
            output = (
                model(**transformed) if isinstance(transformed, Mapping) else model(transformed)
            )
            logits = output.logits if hasattr(output, "logits") else output
            augmented.append(logits.float())
        outputs.append(torch.stack(augmented).mean(dim=0).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0,))


def benchmark_inference(
    predict_one_batch: Callable[[], Any],
    *,
    total_batches: int,
    warmup_batches: int = 2,
) -> dict[str, float]:
    for _ in range(warmup_batches):
        predict_one_batch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    predict_one_batch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds_per_batch = time.perf_counter() - started
    return {
        "seconds_per_batch": seconds_per_batch,
        "estimated_total_seconds": seconds_per_batch * total_batches,
    }


def weighted_ensemble(
    predictions: list[np.ndarray], weights: list[float] | None = None
) -> np.ndarray:
    if not predictions:
        raise ValueError("At least one prediction array is required")
    if any(item.shape != predictions[0].shape for item in predictions):
        raise ValueError("All prediction arrays must have the same shape")
    weights_array = np.asarray(
        weights if weights is not None else [1.0] * len(predictions), dtype=float
    )
    if len(weights_array) != len(predictions) or weights_array.sum() <= 0:
        raise ValueError("Invalid ensemble weights")
    weights_array /= weights_array.sum()
    return np.average(np.stack(predictions), axis=0, weights=weights_array)
