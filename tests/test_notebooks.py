from __future__ import annotations

import json

from olp_ai_26.notebooks import percent_script_to_notebook


def test_percent_script_export(tmp_path):
    source = tmp_path / "template.py"
    source.write_text("# %% [markdown]\n# Title\n# %%\nx = 1\n", encoding="utf-8")
    output = percent_script_to_notebook(source, tmp_path / "template.ipynb")
    notebook = json.loads(output.read_text(encoding="utf-8"))
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code"]
    assert notebook["nbformat"] == 4
