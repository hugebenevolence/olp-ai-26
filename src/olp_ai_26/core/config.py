"""Reproducible competition, trainer, device, and wall-clock configuration objects."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


def resolve_device(requested: str = "auto") -> str:
    """Resolve a portable torch device without importing torch at module import time."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and, when installed, PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass(slots=True)
class CompetitionConfig:
    """Shared paths, hardware, seed, time budget, and pretrained-weight policy."""

    data_dir: Path | str = Path("data")
    output_dir: Path | str = Path("outputs")
    seed: int = 42
    device: str = "auto"
    time_budget_minutes: float = 330.0
    num_workers: int = 2
    pretrained_allowed: bool = False
    deterministic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.device = resolve_device(self.device)
        if self.time_budget_minutes <= 0:
            raise ValueError("time_budget_minutes must be positive")

    def prepare(self) -> None:
        """Create the output directory and seed supported random-number generators."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(self.seed, self.deterministic)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration with paths converted to JSON-compatible strings."""
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def save(self, path: Path | str | None = None) -> Path:
        """Write the configuration to JSON and return the destination path."""
        destination = Path(path) if path else self.output_dir / "config.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination


@dataclass(slots=True)
class TrainerConfig:
    """Optimizer, scheduler, precision, checkpoint, and early-stopping settings."""

    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    optimizer: Literal["adamw", "adam", "sgd", "adafactor"] = "adamw"
    scheduler: Literal["none", "cosine", "onecycle", "plateau"] = "cosine"
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    mixed_precision: Literal["auto", "fp16", "bf16", "none"] | bool = "auto"
    patience: int = 3
    metric_name: str = "macro_f1"
    maximize_metric: bool = True
    checkpoint_name: str = "best.pt"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.mixed_precision not in {True, False, "auto", "fp16", "bf16", "none"}:
            raise ValueError("mixed_precision must be auto, fp16, bf16, none, or a boolean")


class TimeBudget:
    """Monotonic wall-clock guard used by training and inference loops."""

    def __init__(self, minutes: float, reserve_minutes: float = 15.0) -> None:
        if minutes <= 0 or reserve_minutes < 0:
            raise ValueError("Invalid time budget")
        self.total_seconds = minutes * 60
        self.reserve_seconds = reserve_minutes * 60
        self.started_at = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        """Return monotonic seconds elapsed since budget creation."""
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        """Return non-negative wall-clock seconds remaining."""
        return max(0.0, self.total_seconds - self.elapsed_seconds)

    @property
    def should_stop(self) -> bool:
        """Return whether execution has entered the reserved shutdown window."""
        return self.remaining_seconds <= self.reserve_seconds

    def checkpoint(self, stage: str) -> None:
        """Raise ``TimeoutError`` if starting a named stage would consume the reserve."""
        if self.should_stop:
            remaining = self.remaining_seconds / 60
            raise TimeoutError(
                f"Time reserve reached before {stage}; {remaining:.1f} minutes remain"
            )
