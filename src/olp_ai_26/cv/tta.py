"""Named test-time augmentation (TTA) recipes for classification and segmentation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

CLASSIFICATION_TTA: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda value: value,
    "hflip": lambda value: torch.flip(value, dims=(-1,)),
    "vflip": lambda value: torch.flip(value, dims=(-2,)),
    "rot90": lambda value: torch.rot90(value, 1, dims=(-2, -1)),
}


def build_classification_tta(names: Iterable[str] = ("identity", "hflip")) -> list[Callable]:
    """Resolve named image transforms for ``core.inference.predict_logits``.

    Use ``("identity",)`` to disable TTA. Always compare the same validation split with and
    without TTA because flips and rotations can invalidate orientation-sensitive labels.
    """
    resolved = []
    for name in names:
        try:
            resolved.append(CLASSIFICATION_TTA[name.lower()])
        except KeyError as error:
            raise ValueError(
                f"Unknown classification TTA {name!r}; choose {sorted(CLASSIFICATION_TTA)}"
            ) from error
    if not resolved:
        raise ValueError("At least one TTA transform is required; use 'identity' for no TTA")
    return resolved


@dataclass(frozen=True)
class SegmentationTTA:
    """A spatial transform and the inverse used to align predicted masks."""

    name: str
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


SEGMENTATION_TTA: dict[str, SegmentationTTA] = {
    "identity": SegmentationTTA("identity", lambda x: x, lambda x: x),
    "hflip": SegmentationTTA(
        "hflip", lambda x: torch.flip(x, (-1,)), lambda x: torch.flip(x, (-1,))
    ),
    "vflip": SegmentationTTA(
        "vflip", lambda x: torch.flip(x, (-2,)), lambda x: torch.flip(x, (-2,))
    ),
    "rot90": SegmentationTTA(
        "rot90", lambda x: torch.rot90(x, 1, (-2, -1)), lambda x: torch.rot90(x, 3, (-2, -1))
    ),
}


def _images(batch: Any, device: torch.device) -> torch.Tensor:
    """Extract an image tensor from a common loader batch shape."""
    if isinstance(batch, dict):
        batch = batch.get("image", batch.get("pixel_values"))
    elif isinstance(batch, (tuple, list)):
        batch = batch[0]
    if not isinstance(batch, torch.Tensor):
        raise TypeError("Segmentation TTA expects a tensor, tuple first item, or image mapping")
    return batch.to(device)


@torch.inference_mode()
def predict_segmentation_tta(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    device: str = "auto",
    names: Iterable[str] = ("identity", "hflip"),
    mixed_precision: bool = True,
) -> torch.Tensor:
    """Average aligned segmentation logits over named spatial augmentations.

    Returns:
        A CPU tensor shaped ``[N, C, H, W]``. Apply ``sigmoid`` for binary/multilabel masks or
        ``softmax(dim=1)`` for mutually exclusive multiclass masks.
    """
    resolved_device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    recipes = []
    for name in names:
        if name not in SEGMENTATION_TTA:
            raise ValueError(
                f"Unknown segmentation TTA {name!r}; choose {sorted(SEGMENTATION_TTA)}"
            )
        recipes.append(SEGMENTATION_TTA[name])
    if not recipes:
        raise ValueError("At least one TTA recipe is required")
    model = model.to(resolved_device).eval()
    use_amp = mixed_precision and resolved_device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    outputs = []
    for batch in loader:
        images = _images(batch, resolved_device)
        predictions = []
        for recipe in recipes:
            with torch.autocast(resolved_device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(recipe.forward(images))
            predictions.append(recipe.inverse(logits).float())
        outputs.append(torch.stack(predictions).mean(0).cpu())
    return torch.cat(outputs) if outputs else torch.empty(0)
