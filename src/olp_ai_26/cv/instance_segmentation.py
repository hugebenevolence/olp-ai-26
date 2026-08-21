"""Instance-segmentation dataset and Mask R-CNN model factory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

INSTANCE_SEGMENTATION_ARCHITECTURES = (
    "maskrcnn_resnet50_fpn_v2",
    "maskrcnn_resnet50_fpn",
)


class InstanceMaskDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Load one image and one or more per-instance binary-mask files per example.

    The annotation table uses one row per instance. Rows with the same image path are grouped.
    Each mask must have the original image size and each foreground label must be at least 1.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str = "image",
        mask_column: str = "mask",
        label_column: str = "label",
        root: Path | str = ".",
    ) -> None:
        """Validate annotations and group instance rows by source image."""
        required = {image_column, mask_column, label_column}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Instance table is missing columns: {sorted(missing)}")
        self.groups = list(frame.groupby(image_column, sort=False))
        self.mask_column = mask_column
        self.label_column = label_column
        self.root = Path(root)

    def __len__(self) -> int:
        """Return the number of distinct images."""
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return an image and torchvision-compatible boxes, labels, and instance masks."""
        image_path, rows = self.groups[index]
        with Image.open(self.root / str(image_path)) as source:
            image = TF.pil_to_tensor(source.convert("RGB")).float() / 255
        masks = []
        for path in rows[self.mask_column]:
            with Image.open(self.root / str(path)) as source:
                masks.append((TF.pil_to_tensor(source.convert("L"))[0] > 0).to(torch.uint8))
        masks_tensor = torch.stack(masks)
        boxes = []
        for mask in masks_tensor:
            y, x = torch.where(mask > 0)
            if len(x) == 0:
                raise ValueError(f"Empty instance mask in image {image_path!r}")
            boxes.append(torch.tensor([x.min(), y.min(), x.max() + 1, y.max() + 1]))
        boxes_tensor = torch.stack(boxes).float()
        labels = torch.as_tensor(rows[self.label_column].to_numpy(), dtype=torch.int64)
        if (labels < 1).any():
            raise ValueError("Instance foreground labels must start at 1")
        target = {
            "boxes": boxes_tensor,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor([index]),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return image, target


def build_instance_segmentation_model(
    num_classes: int,
    *,
    architecture: str = "maskrcnn_resnet50_fpn_v2",
    pretrained_allowed: bool = False,
) -> nn.Module:
    """Build Mask R-CNN and replace its box and mask predictors.

    ``num_classes`` includes background. Default weights are downloaded only when
    ``pretrained_allowed=True``.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    factories = {
        "maskrcnn_resnet50_fpn_v2": maskrcnn_resnet50_fpn_v2,
        "maskrcnn_resnet50_fpn": maskrcnn_resnet50_fpn,
    }
    try:
        factory = factories[architecture]
    except KeyError as error:
        raise ValueError(
            f"Unknown instance model {architecture!r}; choose {sorted(factories)}"
        ) from error
    model = factory(
        weights="DEFAULT" if pretrained_allowed else None,
        weights_backbone="DEFAULT" if pretrained_allowed else None,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model
