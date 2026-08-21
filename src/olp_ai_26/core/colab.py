"""Google Colab runtime detection, fast local paths, precision, and artifact persistence."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def is_colab() -> bool:
    """Return whether execution appears to be inside a hosted Colab VM."""
    return "google.colab" in sys.modules or Path("/content").is_dir()


@dataclass(slots=True)
class ColabPaths:
    """Fast local paths plus an optional durable Google Drive destination."""

    runtime_dir: Path
    data_dir: Path
    output_dir: Path
    persistent_dir: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        runtime_dir: Path | str | None = None,
        persistent_dir: Path | str | None = None,
    ) -> ColabPaths:
        """Create and prepare fast runtime paths plus an optional persistent destination."""
        root = Path(runtime_dir or ("/content/olp_runtime" if is_colab() else "olp_runtime"))
        paths = cls(
            runtime_dir=root,
            data_dir=root / "data",
            output_dir=root / "outputs",
            persistent_dir=Path(persistent_dir) if persistent_dir else None,
        )
        paths.prepare()
        return paths

    def prepare(self) -> None:
        """Create configured runtime, data, output, and optional persistent directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.persistent_dir:
            self.persistent_dir.mkdir(parents=True, exist_ok=True)


def mount_google_drive(mount_point: Path | str = "/content/drive") -> Path:
    """Mount Google Drive in Colab and return the mount path."""
    if not is_colab():
        raise RuntimeError("Google Drive mounting is available only in a hosted Colab runtime")
    from google.colab import drive

    destination = Path(mount_point)
    drive.mount(str(destination))
    return destination


def stage_data(source: Path | str, destination: Path | str) -> Path:
    """Copy data to the Colab VM; archives are unpacked locally for fast training I/O."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_dir():
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        return destination_path
    local_archive = destination_path.parent / source_path.name
    if source_path.resolve() != local_archive.resolve():
        shutil.copy2(source_path, local_archive)
    try:
        shutil.unpack_archive(str(local_archive), str(destination_path))
    except shutil.ReadError as error:
        raise ValueError(f"Unsupported data archive: {source_path}") from error
    return destination_path


def sync_artifacts(
    source_dir: Path | str,
    destination_dir: Path | str | None,
    *,
    patterns: tuple[str, ...] = (
        "*.pt",
        "*.pth",
        "*.bin",
        "*.safetensors",
        "*.model",
        "*.csv",
        "*.json",
        "*.md",
    ),
) -> list[Path]:
    """Persist only small, important artifacts instead of training directly on Drive."""
    if destination_dir is None:
        return []
    source = Path(source_dir)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in source.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(target)
            copied.append(target)
    return copied


def gpu_report() -> dict[str, Any]:
    """Report torch, CUDA availability, GPU memory, and BF16 capability."""
    import torch

    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": properties.name,
                "gpu_memory_gb": round(properties.total_memory / 1024**3, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return report


def hf_precision_flags(device: str) -> dict[str, bool]:
    """Return mutually exclusive BF16/FP16 flags for Hugging Face training arguments."""
    import torch

    use_cuda = device == "cuda" and torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    return {"bf16": use_bf16, "fp16": use_cuda and not use_bf16}


def dataloader_kwargs(device: str, num_workers: int) -> dict[str, Any]:
    """Return safe DataLoader worker, pinned-memory, and persistence settings."""
    return {
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
        "persistent_workers": num_workers > 0,
    }
