from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_yolo_data_config(path: Path | str) -> dict[str, Any]:
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
    """Run a YOLO baseline only from an explicit local architecture/weight file."""
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
