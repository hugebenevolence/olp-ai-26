# Your first practical session

Goal: produce a scored, reloadable baseline and correctly aligned submission in 45–60 minutes.
Start with text or tabular data on CPU. Use a real dataset after the plumbing works.

[Guide index](README.md) · [15-day plan](15_DAY_PREPARATION.md) · [Notebook catalog](BASELINE_CATALOG.md)

## 1. Prepare the local environment and small offline packs

Run from the repository root:

If `data/practice` already contains the four packs, use them and skip the creation command.
Choose a new destination such as `data/practice_round2` for another fresh exercise.

```powershell
uv sync --group dev
uv run olp-ai doctor
uv run olp-ai practice-data data/practice --task all
```

If uv's global cache is inaccessible, add `--cache-dir .uv-cache` after `uv`. If uv is installed
but absent from PATH on Windows, invoke the executable in your installation directory explicitly.
Keep the Python version you selected for local work; separately verify the expected Python 3.12
Colab runtime before relying on GPU notebooks.

The command creates four packs: text, tabular, image classification, and binary segmentation.
Each has `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`, and separate
`grading/answers.csv`. The sample deliberately shuffles test IDs. The command refuses to overwrite
a nonempty pack; choose a new directory for another exercise.

**Keep grading answers closed until the exercise ends.** Synthetic scores verify a workflow and
do not establish real competition quality. These packs are preparation material, never additional
official contest training data.

## 2. Open one notebook and change the configuration

For text, use `notebooks/nlp_classification_template.ipynb`; replace its configuration line
`DATA_DIR = paths.data_dir` with:

```python
DATA_DIR = PROJECT_ROOT / "data" / "practice" / "text" / "data"
```

For tabular, use `notebooks/tabular_template.ipynb` and the analogous path ending in
`"tabular" / "data"`. Start with `TASK="classification"`, `MODEL="linear"`, and
`METRIC="macro_f1"`. Compare `MODEL="dummy"` and then `"hist_gradient_boosting"` on the same fold.

Run every stage: inspect, split, fit, validate, save/reload, infer, validate submission, and report.
The local output is under `olp_runtime/outputs/nlp_classification/` or `.../tabular/`.
Use a fresh output directory or copy each complete bundle before another experiment overwrites it.

In Colab, stage the repository at `/content/olp-ai-26` and the selected pack under
`/content/olp_runtime/data`. Select GPU only when your chosen task needs it. Use the notebook
bootstrap, not `uv sync` or a pip installation of the project. It installs only requested missing
dependencies with constraints on the existing numerical/CUDA stack. See the [Colab runbook](COLAB_RUNBOOK.md).

## 3. Check the complete result

You should have four artifacts:

| File | Purpose |
|---|---|
| `pipeline.pkl` | Fitted preprocessing, estimator, and label/feature handling |
| `submission.csv` | Predictions aligned to the sample IDs |
| `run.json` | Configuration, validation score, fold positions, and inference recipe |
| `technical_report.md` | Human-readable explanation for a teammate/reviewer |

Validate the text CSV, for example:

```powershell
uv run olp-ai validate-submission olp_runtime/outputs/nlp_classification/submission.csv data/practice/text/data/sample_submission.csv --id-column id
```

For a mask contract that permits empty RLE strings, add `--allow-empty-column rle` to CSV
validation. Leave it disabled for required labels or scores. ID readers preserve literal values
such as `001` and `NA`, which must not be converted into numbers or missing-value markers.

Now open the grading answers and score **once**. This is an exercise grader, not a new model
selection set. For the text pack:

```python
import pandas as pd
from olp_ai_26.core.metrics import evaluate_metric

answers = pd.read_csv(PROJECT_ROOT / "data/practice/text/grading/answers.csv", dtype={"id": str})
truth = answers.set_index("id")["label"].reindex(submission["id"]).to_numpy()
print(evaluate_metric("macro_f1", truth, submission["label"].to_numpy()))
```

The baseline fits preprocessing on its training fold. `REFIT_ALL_TRAIN=False` keeps the validation
model as the default saved model. A full-training refit is a separate final decision after model
selection; the earlier validation score must not be described as a score of that refitted model.

## 4. Run three failure drills

1. Shuffle `test.csv` and rerun inference. The ID-to-label result should remain the same.
2. Start a new process/runtime, load `pipeline.pkl`, and reproduce predictions without refitting.
3. Try one intentional feature/path mistake. Read the error and repair it within five minutes.

Only load pickle files produced by your trusted local workflow. GPU recipes also require their
architecture, preprocessing/tokenizer, label order, and exact checkpoint.

## 5. Move to realistic practice

Follow Day 2's metric/leakage exercises, then use Adult or UIT-VSFC with the same complete loop.
Record time to first submission, official metric convention, fold choice, training cost, inference
cost, and one error-based hypothesis. Do not add another architecture until you can deliver and
reload the first one.
