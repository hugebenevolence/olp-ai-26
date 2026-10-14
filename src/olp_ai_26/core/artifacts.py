"""Atomic persistence of local solution metadata, fitted pipelines, and torch state."""

from __future__ import annotations

import json
import pickle
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _write(value: Any, path: Path | str, kind: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        if kind == "json":
            temporary.write_text(
                json.dumps(
                    value, default=_json_default, indent=2, ensure_ascii=False, allow_nan=False
                ),
                encoding="utf-8",
            )
        elif kind == "pickle":
            with temporary.open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            import torch

            torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_json(value: Any, path: Path | str) -> Path:
    """Atomically save UTF-8 metadata, including paths, dataclasses, and NumPy values."""
    return _write(value, path, "json")


def save_pickle(value: Any, path: Path | str) -> Path:
    """Atomically save a fitted local pipeline with preprocessing and original label values."""
    return _write(value, path, "pickle")


def load_pickle(path: Path | str) -> Any:
    """Reload a trusted, locally produced pickle; never load untrusted downloaded pickle files."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_torch_state(value: Any, path: Path | str) -> Path:
    """Atomically save torch state without importing torch until this function is used."""
    return _write(value, path, "torch")


def write_single_file_zip(
    source: Path | str, destination: Path | str, *, arcname: str | None = None
) -> Path:
    """Atomically create a verified one-file archive, retaining the previous ZIP on error."""
    source_path, output = Path(source), Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    name = arcname or source_path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(source_path, arcname=name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.namelist() != [name] or archive.testzip() is not None:
                raise ValueError("Invalid single-file submission archive")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
