from __future__ import annotations

import json
from pathlib import Path

from olp_ai_26.notebooks import percent_script_to_notebook


def test_percent_script_export(tmp_path):
    source = tmp_path / "template.py"
    source.write_text("# %% [markdown]\n# Title\n# %%\nx = 1\n", encoding="utf-8")
    output = percent_script_to_notebook(source, tmp_path / "template.ipynb")
    notebook = json.loads(output.read_text(encoding="utf-8"))
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code"]
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"


def test_generated_colab_notebooks_are_valid_and_compilable():
    notebook_dir = Path(__file__).parents[1] / "notebooks"
    notebooks = sorted(notebook_dir.glob("*_template.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["metadata"]["accelerator"] == "GPU"
        source = "".join(
            line
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )
        assert "/content/olp-ai-26" in source
        assert "ColabPaths.create" in source
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
