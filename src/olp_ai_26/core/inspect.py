from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(slots=True)
class TableProfile:
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    unique: dict[str, int]
    duplicate_rows: int
    memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MediaProfile:
    scanned: int
    valid: int
    corrupt: list[str]
    dimensions: dict[str, int]
    duplicate_groups: list[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_files(root: Path | str) -> dict[str, list[Path]]:
    """Inventory common competition file types recursively."""
    root = Path(root)
    result: dict[str, list[Path]] = {"tables": [], "images": [], "videos": [], "other": []}
    if not root.exists():
        raise FileNotFoundError(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        suffix = path.suffix.lower()
        if suffix in TABLE_SUFFIXES:
            result["tables"].append(path)
        elif suffix in IMAGE_SUFFIXES:
            result["images"].append(path)
        elif suffix in VIDEO_SUFFIXES:
            result["videos"].append(path)
        else:
            result["other"].append(path)
    return result


def load_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported table format: {path.suffix}")


def profile_table(frame: pd.DataFrame) -> TableProfile:
    return TableProfile(
        rows=len(frame),
        columns=[str(column) for column in frame.columns],
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        missing={str(column): int(value) for column, value in frame.isna().sum().items()},
        unique={str(column): int(frame[column].nunique(dropna=False)) for column in frame},
        duplicate_rows=int(frame.duplicated().sum()),
        memory_mb=float(frame.memory_usage(deep=True).sum() / 1024**2),
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def profile_images(paths: Iterable[Path | str], limit: int | None = 5000) -> MediaProfile:
    selected = [Path(path) for path in paths]
    if limit is not None:
        selected = selected[:limit]
    corrupt: list[str] = []
    dimensions: Counter[str] = Counter()
    hashes: dict[str, list[str]] = {}
    for path in selected:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
            hashes.setdefault(_sha256(path), []).append(str(path))
        except (OSError, ValueError):
            corrupt.append(str(path))
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    return MediaProfile(
        scanned=len(selected),
        valid=len(selected) - len(corrupt),
        corrupt=corrupt,
        dimensions=dict(dimensions.most_common()),
        duplicate_groups=duplicate_groups,
    )


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        report["torch"] = None
    return report


def dataset_report(root: Path | str, image_limit: int = 1000) -> dict[str, Any]:
    inventory = discover_files(root)
    tables = {}
    for path in inventory["tables"]:
        try:
            tables[str(path)] = profile_table(load_table(path)).to_dict()
        except (OSError, ValueError, pd.errors.ParserError) as error:
            tables[str(path)] = {"error": str(error)}
    images = profile_images(inventory["images"], limit=image_limit).to_dict()
    return {
        "root": str(Path(root).resolve()),
        "counts": {name: len(paths) for name, paths in inventory.items()},
        "tables": tables,
        "images": images,
        "environment": environment_report(),
    }


def save_report(report: dict[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
