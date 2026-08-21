"""Offline JSON Lines experiment logging for contest-safe provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ExperimentLogger:
    """Small offline JSONL logger; safe when network tracking is unavailable."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        """Append a timestamped event and JSON-compatible payload atomically to the log."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def read_experiments(path: Path | str) -> list[dict[str, Any]]:
    """Read all non-empty JSON Lines experiment records from a file."""
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
