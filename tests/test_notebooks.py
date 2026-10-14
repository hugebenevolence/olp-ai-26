from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from olp_ai_26.notebooks import percent_script_to_notebook
from olp_ai_26.practice import create_practice_pack


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
    expected = {
        "adversarial_robustness_template.ipynb",
        "classification_segmentation_template.ipynb",
        "cv_classification_template.ipynb",
        "image_captioning_template.ipynb",
        "image_anomaly_detection_dino_augmented_template.ipynb",
        "image_anomaly_detection_template.ipynb",
        "instance_segmentation_template.ipynb",
        "nlp_classification_template.ipynb",
        "object_detection_template.ipynb",
        "retrieval_template.ipynb",
        "semantic_segmentation_template.ipynb",
        "tabular_template.ipynb",
        "seq2seq_template.ipynb",
        "token_classification_template.ipynb",
        "transformer_text_classification_template.ipynb",
        "video_classification_template.ipynb",
    }
    assert {path.name for path in notebooks} == expected
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


def test_generated_cells_match_canonical_sources(tmp_path):
    directory = Path(__file__).parents[1] / "notebooks"
    for source in directory.glob("*_template.py"):
        generated = percent_script_to_notebook(source, tmp_path / source.with_suffix(".ipynb").name)
        expected = json.loads(generated.read_text(encoding="utf-8"))
        actual = json.loads(source.with_suffix(".ipynb").read_text(encoding="utf-8"))
        assert actual == expected, f"Regenerate {source.name}; generated artifact has drifted"


@pytest.mark.parametrize("task,notebook", [("text", "nlp_classification"), ("tabular", "tabular")])
def test_cpu_notebooks_execute_through_delivery(tmp_path, monkeypatch, task, notebook):
    repository = Path(__file__).parents[1]
    pack = create_practice_pack(tmp_path / "pack", task=task)
    source = (repository / "notebooks" / f"{notebook}_template.py").read_text(encoding="utf-8")
    source = source.replace(
        'PROJECT_ROOT = Path("/content/olp-ai-26") '
        'if "google.colab" in sys.modules else Path.cwd()',
        f"PROJECT_ROOT = Path({str(repository)!r})",
    )
    source = source.replace("DATA_DIR = paths.data_dir", f"DATA_DIR = Path({str(pack / 'data')!r})")
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    exec(compile(source, f"{notebook}:full-cpu-practice", "exec"), namespace)
    submission = namespace["submission"]
    expected = pd.read_csv(pack / "grading/answers.csv").set_index("id")["label"]
    truth = expected.reindex(submission["id"]).to_numpy()
    assert (truth == submission["label"].to_numpy()).mean() >= 0.8
    output = (
        tmp_path
        / "olp_runtime"
        / "outputs"
        / ("nlp_classification" if task == "text" else "tabular")
    )
    assert (output / "pipeline.pkl").is_file()
    assert (output / "submission.csv").is_file()
    assert (output / "run.json").is_file()
    assert (output / "technical_report.md").is_file()
