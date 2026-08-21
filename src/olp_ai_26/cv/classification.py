from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ImageTableDataset(Dataset[Any]):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str,
        target_column: str | None = None,
        root: Path | str = ".",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        label_to_index: Mapping[object, int] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.target_column = target_column
        self.root = Path(root)
        self.transform = transform or build_image_transforms(training=False)
        self.label_to_index = dict(label_to_index or {})

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Any:
        row = self.frame.iloc[index]
        path = self.root / str(row[self.image_column])
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image)
        if self.target_column:
            label = row[self.target_column]
            target = torch.tensor(self.label_to_index.get(label, label), dtype=torch.long)
            return tensor, target
        return tensor


def build_image_transforms(
    *,
    size: int = 224,
    training: bool,
    normalize: bool = True,
) -> Callable[[Image.Image], torch.Tensor]:
    operations: list[Any]
    if training:
        operations = [
            v2.RandomResizedCrop((size, size), scale=(0.7, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ]
    else:
        operations = [v2.Resize((size, size))]
    operations.append(v2.ToImage())
    operations.append(v2.ToDtype(torch.float32, scale=True))
    if normalize:
        operations.append(v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return v2.Compose(operations)


def label_mapping(labels: pd.Series) -> tuple[dict[object, int], dict[int, object]]:
    classes = sorted(labels.dropna().unique().tolist(), key=str)
    forward = {label: index for index, label in enumerate(classes)}
    return forward, {index: label for label, index in forward.items()}


def build_image_classifier(
    model_name: str,
    num_classes: int,
    *,
    pretrained_allowed: bool = False,
    checkpoint_path: Path | str | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a timm classifier; remote pretrained weights are opt-in."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained_allowed and checkpoint_path is None,
        checkpoint_path=str(checkpoint_path or ""),
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def class_weights(labels: pd.Series, mapping: Mapping[object, int]) -> torch.Tensor:
    counts = labels.map(mapping).value_counts().sort_index()
    weights = len(labels) / (len(mapping) * counts.reindex(range(len(mapping)), fill_value=1))
    return torch.tensor(weights.to_numpy(dtype=np.float32))


def classification_loss(
    *,
    multilabel: bool = False,
    weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    if multilabel:
        return nn.BCEWithLogitsLoss(pos_weight=weights)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
