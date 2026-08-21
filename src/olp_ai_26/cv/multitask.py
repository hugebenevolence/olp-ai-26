"""Image classification with an auxiliary semantic-segmentation head."""

from __future__ import annotations

from collections.abc import Mapping

import timm
import torch
from torch import nn
from torch.nn import functional as F

from olp_ai_26.cv.segmentation import DiceBCELoss


class ClassificationSegmentationModel(nn.Module):
    """Share a strong timm feature encoder between class and mask predictions.

    The auxiliary mask task can encourage spatially meaningful features when masks are provided.
    It is not instance segmentation; use Mask R-CNN for separate object instances.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        *,
        mask_classes: int = 1,
        decoder_channels: int = 128,
        pretrained_allowed: bool = False,
    ) -> None:
        """Build a feature-only timm encoder, classifier head, and lightweight FPN decoder."""
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained_allowed,
            features_only=True,
            out_indices=(-4, -3, -2, -1),
        )
        channels = self.encoder.feature_info.channels()
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.projections = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, mask_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``class_logits`` and input-resolution ``mask_logits``."""
        features = self.encoder(images)
        class_logits = self.classifier(features[-1].mean(dim=(-2, -1)))
        target_size = features[0].shape[-2:]
        pyramid = [
            F.interpolate(projection(feature), target_size, mode="bilinear", align_corners=False)
            for projection, feature in zip(self.projections, features, strict=True)
        ]
        mask_logits = self.mask_head(torch.stack(pyramid).sum(0))
        mask_logits = F.interpolate(
            mask_logits, images.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"class_logits": class_logits, "mask_logits": mask_logits}


class ClassificationSegmentationLoss(nn.Module):
    """Combine classification cross entropy with auxiliary Dice+BCE mask loss."""

    def __init__(self, *, class_weight: float = 1.0, mask_weight: float = 0.5) -> None:
        """Set the relative classification and segmentation contributions."""
        super().__init__()
        if class_weight < 0 or mask_weight < 0 or class_weight + mask_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.class_loss = nn.CrossEntropyLoss()
        self.mask_loss = DiceBCELoss()

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        class_targets: torch.Tensor,
        mask_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return total loss plus detached component values for logging."""
        classification = self.class_loss(outputs["class_logits"], class_targets)
        segmentation = self.mask_loss(outputs["mask_logits"], mask_targets)
        total = self.class_weight * classification + self.mask_weight * segmentation
        return total, {
            "classification_loss": float(classification.detach()),
            "segmentation_loss": float(segmentation.detach()),
        }
