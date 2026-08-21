from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_segmentation_model(
    *,
    architecture: str = "unet",
    encoder_name: str = "resnet18",
    classes: int = 1,
    pretrained_allowed: bool = False,
) -> nn.Module:
    architectures = {
        "unet": smp.Unet,
        "fpn": smp.FPN,
        "deeplabv3plus": smp.DeepLabV3Plus,
    }
    try:
        constructor = architectures[architecture.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown architecture: {architecture}") from error
    return constructor(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained_allowed else None,
        in_channels=3,
        classes=classes,
    )


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets.float())
        probabilities = logits.sigmoid()
        axes = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=axes)
        dice = (2 * intersection + 1e-7) / (
            probabilities.sum(dim=axes) + targets.sum(dim=axes) + 1e-7
        )
        return (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice.mean())
