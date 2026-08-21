"""Semantic-segmentation datasets, model factory, loss, and mask serialization helpers."""

from __future__ import annotations

from collections.abc import Callable
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
    ) -> None:
        """Store table/path settings and create paired torchvision transforms."""
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.mask_column = mask_column
        self.root = Path(root)
        self.binary_mask = binary_mask
        self.normalize = normalize
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
            return v2.functional.to_dtype(image, torch.float32, scale=True)
        with Image.open(self.root / str(row[self.mask_column])) as source:
            mask = tv_tensors.Mask(v2.functional.to_image(source.convert("L")))
        image, mask = self.spatial_transform(image, mask)
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        if self.normalize:
            image = v2.functional.normalize(
                image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        if self.binary_mask:
            mask = (mask > 0).float()
        else:
            mask = mask.to(torch.long).squeeze(0)
        return image, mask


def build_segmentation_model(
    *,
    architecture: str = "unet",
    encoder_name: str = "resnet18",
    classes: int = 1,
    pretrained_allowed: bool = False,
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
    return constructor(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained_allowed else None,
        in_channels=3,
        classes=classes,
    )


class DiceBCELoss(nn.Module):
    """Weighted sum of binary cross entropy and soft Dice loss."""

    def __init__(self, dice_weight: float = 0.5) -> None:
        """Create the loss; ``dice_weight=1`` selects pure soft Dice."""
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError("dice_weight must be between 0 and 1")
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss from raw logits and binary masks of the same shape."""
        if logits.shape != targets.shape:
            targets = F.interpolate(targets.float(), logits.shape[-2:], mode="nearest")
        bce = self.bce(logits, targets.float())
        probabilities = logits.sigmoid()
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
    starts = values[::2] - 1
    lengths = values[1::2]
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths, strict=True):
        flat[start : start + length] = 1
    return flat.reshape((shape[1], shape[0])).T
