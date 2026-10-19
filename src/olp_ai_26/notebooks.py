"""Convert readable percent-style Python templates into Colab-ready notebooks."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path


def _markdown_source(lines: list[str]) -> list[str]:
    converted = []
    for line in lines:
        converted.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
    return converted


def _exercise_source(lines: list[str]) -> list[str]:
    """Hide only explicitly marked answer blocks, preserving the surrounding student scaffold."""
    result: list[str] = []
    hiding = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# >>> exercise:"):
            if hiding:
                raise ValueError("Exercise answer blocks cannot be nested")
            hiding = True
            indent = line[: len(line) - len(line.lstrip())]
            hint = stripped.partition(":")[2].strip()
            result.append(f"{indent}raise NotImplementedError({('TODO: ' + hint)!r})\n")
        elif stripped == "# <<< exercise":
            if not hiding:
                raise ValueError("Exercise block ended without a start")
            hiding = False
        elif not hiding:
            result.append(line)
    if hiding:
        raise ValueError("Exercise block was not closed")
    return result


def percent_script_to_notebook(source: Path | str, destination: Path | str) -> Path:
    """Convert a lightweight ``# %%`` Python template into a Jupyter notebook."""
    source_path = Path(source)
    cells: list[dict[str, object]] = []
    current: list[str] = []
    kind = "code"
    exercise = False
    tags: list[str] = []
    baseline = source_path.parent.name == "learning" and "_baseline_" in source_path.name

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
            if tags:
                cell["metadata"] = {"tags": tags.copy()}
            if "provided" in tags:
                cell["metadata"]["jupyter"] = {"source_hidden": True}
            if exercise:
                cell["source"] = _exercise_source(current)
        cells.append(cell)
        if (exercise or baseline) and kind == "code":
            answer = "".join(
                line
                for line in current
                if not line.strip().startswith(("# >>> exercise:", "# <<< exercise"))
            )
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {"tags": ["solution"], "collapsed": True},
                    "source": [
                        "<details>\n",
                        "<summary>Reveal expected code — compare after your attempt</summary>\n\n",
                        f"<pre><code>{escape(answer)}</code></pre>\n",
                        "</details>\n",
                    ],
                }
            )
        current = []

    for line in source_path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            exercise = "[exercise]" in line
            tags = [tag for tag in re.findall(r"\[([^]]+)\]", line) if tag != "markdown"]
        else:
            current.append(line)
    flush()
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    if source_path.parent.name == "learning":
        for index, cell in enumerate(cells):
            cell["id"] = f"lesson-{index:03d}"
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return destination_path


def build_all_notebooks(root: Path | str = "notebooks") -> list[Path]:
    """Regenerate every ``*_template.ipynb`` from its canonical Python source."""
    root_path = Path(root)
    outputs = []
    for source in sorted(root_path.rglob("*_template.py")):
        outputs.append(percent_script_to_notebook(source, source.with_suffix(".ipynb")))
    return outputs
