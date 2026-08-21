from __future__ import annotations

import json
from pathlib import Path


def _markdown_source(lines: list[str]) -> list[str]:
    converted = []
    for line in lines:
        converted.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
    return converted


def percent_script_to_notebook(source: Path | str, destination: Path | str) -> Path:
    """Convert a lightweight ``# %%`` Python template into a Jupyter notebook."""
    source_path = Path(source)
    cells: list[dict[str, object]] = []
    current: list[str] = []
    kind = "code"

    def flush() -> None:
        nonlocal current
        if not current:
            return
        content = _markdown_source(current) if kind == "markdown" else current.copy()
        cell: dict[str, object] = {
            "cell_type": kind,
            "metadata": {},
            "source": content,
        }
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
        current = []

    for line in source_path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            current.append(line)
    flush()
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return destination_path


def build_all_notebooks(root: Path | str = "notebooks") -> list[Path]:
    root_path = Path(root)
    outputs = []
    for source in sorted(root_path.glob("*_template.py")):
        outputs.append(percent_script_to_notebook(source, source.with_suffix(".ipynb")))
    return outputs
