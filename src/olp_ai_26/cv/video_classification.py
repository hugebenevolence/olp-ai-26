from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from olp_ai_26.cv.classification import build_image_transforms


def sample_video_frames(path: Path | str, num_frames: int = 16) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {path}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    positions = torch.linspace(0, frame_count - 1, steps=num_frames).round().int().tolist()
    frames: list[Image.Image] = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise OSError(f"Cannot decode frame {position} from {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        capture.release()
    return frames


class VideoTableDataset(Dataset[Any]):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        video_column: str,
        target_column: str | None = None,
        root: Path | str = ".",
        num_frames: int = 16,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        label_to_index: Mapping[object, int] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.video_column = video_column
        self.target_column = target_column
        self.root = Path(root)
        self.num_frames = num_frames
        self.transform = transform or build_image_transforms(training=False)
        self.label_to_index = dict(label_to_index or {})

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Any:
        row = self.frame.iloc[index]
        path = self.root / str(row[self.video_column])
        video = torch.stack(
            [self.transform(image) for image in sample_video_frames(path, self.num_frames)]
        )
        if self.target_column:
            label = row[self.target_column]
            target = torch.tensor(self.label_to_index.get(label, label), dtype=torch.long)
            return video, target
        return video


class TemporalPoolingClassifier(nn.Module):
    """Fast action baseline: shared 2D backbone followed by temporal pooling."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        *,
        pretrained_allowed: bool = False,
        pooling: str = "mean",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained_allowed, num_classes=0, global_pool="avg"
        )
        self.pooling = pooling
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        features = self.backbone(video.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, -1)
        if self.pooling == "max":
            pooled = features.max(dim=1).values
        elif self.pooling == "mean":
            pooled = features.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal pooling: {self.pooling}")
        return self.head(pooled)
