"""Semantic-segmentation datasets, model factory, loss, and mask serialization helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.cv.weights import local_checkpoint

SEGMENTATION_ARCHITECTURES = (
    "unet",
    "unetplusplus",
    "fpn",
    "pspnet",
    "deeplabv3",
    "deeplabv3plus",
    "linknet",
    "manet",
    "pan",
)


class SegmentationTableDataset(Dataset[Any]):
    """Read paired RGB images and indexed/binary masks described by a table.

    The same random transform is applied to the image and mask. Without a mask column the dataset
    returns only an image tensor, which is suitable for test inference.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str,
        mask_column: str | None = None,
        root: Path | str = ".",
        size: int = 256,
        training: bool = False,
        binary_mask: bool = True,
        normalize: bool = True,
        foreground_values: Sequence[int] | None = None,
        ignore_index: int | None = None,
    ) -> None:
        """Store table/path settings and create paired torchvision transforms."""
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.mask_column = mask_column
        self.root = Path(root)
        self.binary_mask = binary_mask
        self.normalize = normalize
        self.foreground_values = foreground_values
        self.ignore_index = ignore_index
        operations: list[Callable] = [v2.Resize((size, size), antialias=True)]
        if training:
            operations += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(p=0.2)]
        self.spatial_transform = v2.Compose(operations)

    def __len__(self) -> int:
        """Return the number of rows/examples."""
        return len(self.frame)

    def __getitem__(self, index: int) -> Any:
        """Load one image and, when configured, its spatially aligned mask."""
        row = self.frame.iloc[index]
        with Image.open(self.root / str(row[self.image_column])) as source:
            image = v2.functional.to_image(source.convert("RGB"))
        if not self.mask_column:
            image = self.spatial_transform(image)
            image = v2.functional.to_dtype(image, torch.float32, scale=True)
            if self.normalize:
                image = v2.functional.normalize(
                    image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
            return image
        with Image.open(self.root / str(row[self.mask_column])) as source:
            mask = tv_tensors.Mask(v2.functional.to_image(source.convert("L")))
        image, mask = self.spatial_transform(image, mask)
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        if self.normalize:
            image = v2.functional.normalize(
                image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        if self.binary_mask:
            ignored = mask == self.ignore_index if self.ignore_index is not None else None
            mask = (
                torch.isin(mask, torch.tensor(self.foreground_values, dtype=mask.dtype))
                if self.foreground_values is not None
                else mask > 0
            ).float()
            if ignored is not None:
                mask[ignored] = self.ignore_index
        else:
            mask = mask.to(torch.long).squeeze(0)
        return image, mask


def build_segmentation_model(
    *,
    architecture: str = "unet",
    encoder_name: str = "resnet18",
    classes: int = 1,
    pretrained_allowed: bool = False,
    checkpoint_path: Path | str | None = None,
) -> nn.Module:
    """Build an SMP semantic-segmentation model without implicit weight downloads.

    ``encoder_name`` accepts any encoder printed by ``smp.encoders.get_encoder_names()``. Common
    choices include ``resnet18``, ``resnet50``, ``efficientnet-b0``, ``timm-convnext_tiny``, and
    ``tu-swin_tiny_patch4_window7_224`` (availability depends on the installed SMP/timm version).
    """
    import segmentation_models_pytorch as smp

    architectures = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "linknet": smp.Linknet,
        "manet": smp.MAnet,
        "pan": smp.PAN,
    }
    try:
        constructor = architectures[architecture.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown architecture {architecture!r}; choose {sorted(architectures)}"
        ) from error
    checkpoint = local_checkpoint(pretrained_allowed, checkpoint_path)
    model = constructor(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=classes,
    )
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


class DiceBCELoss(nn.Module):
    """Weighted sum of binary cross entropy and soft Dice loss."""

    def __init__(self, dice_weight: float = 0.5, *, ignore_index: int | None = None) -> None:
        """Create the loss; ``dice_weight=1`` selects pure soft Dice."""
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError("dice_weight must be between 0 and 1")
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss from raw logits and binary masks of the same shape."""
        if logits.shape != targets.shape:
            targets = F.interpolate(targets.float(), logits.shape[-2:], mode="nearest")
        keep = (
            torch.ones_like(targets, dtype=torch.bool)
            if self.ignore_index is None
            else targets != self.ignore_index
        )
        targets = targets.float().masked_fill(~keep, 0)
        pixel_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (pixel_loss * keep).sum() / keep.sum().clamp_min(1)
        probabilities = logits.sigmoid() * keep
        axes = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=axes)
        dice = (2 * intersection + 1e-7) / (
            probabilities.sum(dim=axes) + targets.sum(dim=axes) + 1e-7
        )
        return (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice.mean())


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask as one-indexed, column-major run-length encoding."""
    pixels = np.asarray(mask, dtype=np.uint8).T.reshape(-1)
    padded = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(value) for value in changes)


def rle_to_mask(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode one-indexed, column-major RLE into a ``uint8`` mask of ``(height, width)``."""
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or not str(rle).strip():
        return np.zeros(shape, dtype=np.uint8)
    values = np.asarray([int(value) for value in str(rle).split()], dtype=np.int64)
    if len(values) % 2:
        raise ValueError("RLE must contain start/length pairs")
    starts = values[::2] - 1
    lengths = values[1::2]
    if (starts < 0).any() or (lengths < 1).any() or (starts + lengths > shape[0] * shape[1]).any():
        raise ValueError("RLE runs exceed mask bounds or have invalid start/length")
    if len(starts) > 1 and (starts[1:] < (starts + lengths)[:-1]).any():
        raise ValueError("RLE runs must be ordered and nonoverlapping")
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths, strict=True):
        flat[start : start + length] = 1
    return flat.reshape((shape[1], shape[0])).T


def restore_binary_masks(
    logits: torch.Tensor,
    frame: pd.DataFrame,
    *,
    image_column: str,
    root: Path | str,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """Resize binary logits to each original image before thresholding and RLE serialization."""
    if len(logits) != len(frame):
        raise ValueError("Mask logits must match the manifest rows")
    result = []
    for logit, relative_path in zip(logits, frame[image_column], strict=True):
        with Image.open(Path(root) / str(relative_path)) as image:
            height, width = image.height, image.width
        resized = F.interpolate(
            logit.unsqueeze(0).float(), (height, width), mode="bilinear", align_corners=False
        )
        result.append((resized.sigmoid()[0, 0].cpu().numpy() >= threshold).astype(np.uint8))
    return result


def load_binary_masks(
    frame: pd.DataFrame,
    *,
    mask_column: str,
    root: Path | str,
    foreground_values: Sequence[int] | None = None,
    ignore_index: int | None = None,
) -> list[np.ndarray]:
    """Load original-resolution binary references with an explicit foreground-label policy."""
    result = []
    for relative_path in frame[mask_column]:
        with Image.open(Path(root) / str(relative_path)) as image:
            values = np.asarray(image.convert("L"))
        mask = np.isin(values, foreground_values) if foreground_values is not None else values > 0
        labels = mask.astype(np.int64)
        if ignore_index is not None:
            labels[values == ignore_index] = ignore_index
        result.append(labels)
    return result


def score_mask_pairs(
    metric: str, truth: list[np.ndarray], predictions: list[np.ndarray], **metric_options: Any
) -> float:
    """Score variable-size masks with explicit global or mean-per-image Dice/IoU conventions."""
    if not truth or len(truth) != len(predictions):
        raise ValueError("Nonempty matching reference/prediction lists are required")
    if any(
        target.shape != prediction.shape
        for target, prediction in zip(truth, predictions, strict=True)
    ):
        raise ValueError("Original mask and prediction dimensions differ")
    if metric in {"dice_mean_image", "iou_mean_image"}:
        base = metric.removesuffix("_mean_image")
        return float(
            np.mean(
                [
                    evaluate_metric(base, target, prediction, **metric_options)
                    for target, prediction in zip(truth, predictions, strict=True)
                ]
            )
        )
    if metric not in {"dice", "iou"}:
        raise ValueError("Wire the official evaluator for a different mask metric")
    return evaluate_metric(
        metric,
        np.concatenate([mask.reshape(-1) for mask in truth]),
        np.concatenate([mask.reshape(-1) for mask in predictions]),
        **metric_options,
    )
