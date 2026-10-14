"""Torchvision and optional YOLO object-detection building blocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from olp_ai_26.core.config import TimeBudget
from olp_ai_26.cv.segmentation import mask_to_rle
from olp_ai_26.cv.weights import local_checkpoint

DETECTION_ARCHITECTURES = (
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
)


class BoxDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Read detection annotations with one row per bounding box.

    Expected coordinates are absolute ``xmin, ymin, xmax, ymax`` values. Rows sharing an image
    path are grouped into one target dictionary in the format expected by torchvision detectors.
    Class index 0 is reserved for background, so foreground labels must start at 1.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str = "image",
        label_column: str = "label",
        box_columns: tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
        root: Path | str = ".",
    ) -> None:
        """Validate annotation columns and group rows by image path."""
        required = {image_column, label_column, *box_columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Detection table is missing columns: {sorted(missing)}")
        self.groups = list(frame.groupby(image_column, sort=False))
        self.label_column = label_column
        self.box_columns = box_columns
        self.root = Path(root)

    def __len__(self) -> int:
        """Return the number of distinct images."""
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load one RGB image and all of its boxes and labels."""
        relative_path, rows = self.groups[index]
        with Image.open(self.root / str(relative_path)) as source:
            image = TF.pil_to_tensor(source.convert("RGB")).float() / 255
        boxes = torch.as_tensor(rows.loc[:, self.box_columns].to_numpy(), dtype=torch.float32)
        labels = torch.as_tensor(rows[self.label_column].to_numpy(), dtype=torch.int64)
        if (labels < 1).any():
            raise ValueError("Torchvision foreground detection labels must start at 1")
        if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("Boxes must be finite exclusive xyxy rectangles with positive area")
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


def detection_collate(batch: list[tuple[Any, Any]]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Collate variable-size images and variable-length detection targets."""
    return tuple(zip(*batch, strict=True))


class DetectionInferenceDataset(Dataset[Any]):
    """Read an ordered image manifest without resizing or normalizing detector inputs."""

    def __init__(
        self, frame: pd.DataFrame, *, image_column: str = "image", root: Path | str = "."
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.root = Path(root)

    def __len__(self) -> int:
        """Return the number of inference images."""
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return an RGB tensor and empty target for the variable-size detection collator."""
        with Image.open(self.root / str(self.frame.iloc[index][self.image_column])) as image:
            return TF.pil_to_tensor(image.convert("RGB")).float() / 255, {}


def build_detection_model(
    num_classes: int,
    *,
    architecture: str = "fasterrcnn_resnet50_fpn_v2",
    pretrained_allowed: bool = False,
    checkpoint_path: Path | str | None = None,
) -> nn.Module:
    """Build a torchvision box detector and replace its classification head.

    Args:
        num_classes: Total number of labels including background class 0.
        architecture: ``fasterrcnn_resnet50_fpn_v2``, ``fasterrcnn_resnet50_fpn``, or
            ``fasterrcnn_mobilenet_v3_large_fpn``.
        pretrained_allowed: Require explicitly staged permitted weights when enabled.
        checkpoint_path: Original permitted COCO state for initialization; matching task state
            for reload.
    """
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    factories = {
        "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
        "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    }
    try:
        factory = factories[architecture]
    except KeyError as error:
        raise ValueError(
            f"Unknown detector {architecture!r}; choose {sorted(factories)}"
        ) from error
    checkpoint = local_checkpoint(pretrained_allowed, checkpoint_path)
    model = factory(weights=None, weights_backbone=None)
    if checkpoint and pretrained_allowed:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes)
    if checkpoint and not pretrained_allowed:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def train_detection_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "auto",
    time_budget: TimeBudget | None = None,
) -> float:
    """Train one epoch using the loss dictionary returned by torchvision detectors."""
    resolved = torch.device(
        device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(resolved).train()
    total = 0.0
    batches = 0
    for images, targets in loader:
        if time_budget and time_budget.should_stop:
            break
        images = [image.to(resolved) for image in images]
        targets = [{key: value.to(resolved) for key, value in target.items()} for target in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


@torch.inference_mode()
def predict_detections(
    model: nn.Module, loader: Any, *, device: str = "cpu"
) -> list[dict[str, torch.Tensor]]:
    """Return CPU detection dictionaries in the manifest/dataset order, fitting nothing."""
    model.to(device).eval()
    outputs = []
    for images, _ in loader:
        outputs.extend(
            {key: value.cpu() for key, value in prediction.items()}
            for prediction in model([image.to(device) for image in images])
        )
    return outputs


def serialize_detections(
    predictions: list[dict[str, torch.Tensor]],
    *,
    score_threshold: float = 0.0,
    mask_threshold: float = 0.5,
) -> list[str]:
    """Serialize one JSON list per image, optionally with original-size column-major mask RLE.

    This is the documented practice CSV contract; adapt serialization to the official sample.
    """
    rows = []
    for prediction in predictions:
        instances = []
        for index, score in enumerate(prediction["scores"]):
            if float(score) < score_threshold:
                continue
            instance = {
                "label": int(prediction["labels"][index]),
                "score": float(score),
                "box": prediction["boxes"][index].tolist(),
            }
            if "masks" in prediction:
                mask = prediction["masks"][index, 0] >= mask_threshold
                instance["rle"] = mask_to_rle(mask.numpy())
                instance["shape"] = list(mask.shape)
            instances.append(instance)
        rows.append(json.dumps(instances, allow_nan=False))
    return rows


def validate_yolo_data_config(path: Path | str) -> dict[str, Any]:
    """Load a YOLO dataset YAML and assert the required train/val/name keys."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    missing = {"train", "val", "names"} - set(payload)
    if missing:
        raise ValueError(f"YOLO data config is missing: {sorted(missing)}")
    return payload


def train_yolo_baseline(
    *,
    data_config: Path | str,
    model_path: Path | str,
    pretrained_allowed: bool = False,
    epochs: int = 20,
    image_size: int = 640,
    batch_size: int = 16,
    project: Path | str = "outputs/detection",
    **kwargs: Any,
) -> Any:
    """Train Ultralytics YOLO from an explicit local YAML or permitted local weight file."""
    from ultralytics import YOLO

    validate_yolo_data_config(data_config)
    model_path = Path(model_path)
    if not model_path.exists():
        policy = "allowed local weights" if pretrained_allowed else "a local model YAML"
        raise FileNotFoundError(f"Expected {policy}: {model_path}")
    model = YOLO(str(model_path))
    return model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(project),
        **kwargs,
    )
