"""Generate a concise Markdown technical report from configs, metrics, and artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from olp_ai_26.core.artifacts import save_json
from olp_ai_26.core.split import SplitResult


def _mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return value
    raise TypeError("Expected a dataclass or mapping")


def technical_report(
    *,
    task: str,
    model: str,
    data_summary: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    training_config: Any,
    validation_metrics: Mapping[str, float],
    augmentations: list[str] | None = None,
    checkpoint: str | None = None,
    notes: list[str] | None = None,
) -> str:
    """Render the standard competition handoff report as Markdown text."""
    config = _mapping(training_config)
    lines = [
        "# Technical Report",
        "",
        f"- Task: {task}",
        f"- Model: {model}",
        f"- Checkpoint: {checkpoint or 'not recorded'}",
        "",
        "## Data and validation",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in data_summary.items())
    lines.extend(f"- Split {key}: {value}" for key, value in split_summary.items())
    lines.extend(["", "## Training", ""])
    lines.extend(f"- {key}: {value}" for key, value in config.items())
    lines.extend(["", "## Validation metrics", ""])
    lines.extend(f"- {key}: {value:.6f}" for key, value in validation_metrics.items())
    lines.extend(["", "## Augmentations", ""])
    lines.extend(f"- {item}" for item in (augmentations or ["None"]))
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {item}" for item in notes)
    return "\n".join(lines) + "\n"


def write_technical_report(content: str, path: Path | str) -> Path:
    """Write a UTF-8 technical report, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def record_run(
    output_dir: Path | str,
    *,
    task: str,
    model: str,
    configuration: Any,
    validation_metrics: Mapping[str, float],
    split: SplitResult | None = None,
    data_summary: Mapping[str, Any] | None = None,
    inference_options: Mapping[str, Any] | None = None,
    checkpoint: str | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Save configuration/fold positions and a matching human-readable technical report."""
    root = Path(output_dir)
    split_summary = (
        {"train_rows": len(split.train), "valid_rows": len(split.valid)} if split else {}
    )
    payload = {
        "task": task,
        "model": model,
        "configuration": _mapping(configuration),
        "validation_metrics": dict(validation_metrics),
        "data_summary": dict(data_summary or {}),
        "inference_options": dict(inference_options or {}),
        "checkpoint": checkpoint,
        "notes": notes or [],
        "split": {
            **split_summary,
            "train_indices": split.train_indices if split else [],
            "valid_indices": split.valid_indices if split else [],
        },
    }
    save_json(payload, root / "run.json")
    report = technical_report(
        task=task,
        model=model,
        data_summary=data_summary or {},
        split_summary=split_summary,
        training_config=configuration,
        validation_metrics=validation_metrics,
        checkpoint=checkpoint,
        notes=[*(notes or []), f"Inference options: {dict(inference_options or {})}"],
    )
    return write_technical_report(report, root / "technical_report.md")
