# OLP AI 2026 practical competition kit

A small toolkit for inspecting data, building a legal baseline, evaluating it correctly, and
producing a reproducible submission within a six-hour competition. Python 3.12 or newer.

**Start here: [seven problem families, fourteen learning notebooks](notebooks/learning/README.md).**
Each family has a fill-in baseline with collapsible solutions and an explained advanced study route.
All 14 learning notebooks run independently of the repository: copy one notebook, supply its
data and installed libraries, and edit its paths. The 16 older kit notebooks use the shared package.
Then use the [blank-notebook playbook](docs/BLANK_NOTEBOOK_PLAYBOOK.md) to assess reconstruction
when saved code and internet may be unavailable.
For a guided run with the repository available, use the [practice quickstart](docs/PRACTICE_QUICKSTART.md).
For the full learning sequence, follow the [15-day plan](docs/15_DAY_PREPARATION.md).
The [guide index](docs/README.md) is the main documentation entry point.

## Get a complete CPU baseline working

```powershell
uv sync --locked
uv run olp-ai doctor
uv run olp-ai practice-data data/practice --task all
```

Open `notebooks/nlp_classification_template.ipynb` or `notebooks/tabular_template.ipynb` and
point `DATA_DIR` at the selected practice pack's `data/` directory. Run through saving/reloading,
inference, submission validation, and report creation. The packs deliberately shuffle sample IDs.
Keep their grading answers closed until the exercise ends.

Synthetic packs teach the workflow; their scores do not predict contest performance. Use real
practice data next, and never add these packs to official contest training data.

## Choose a task

| Required output | Start with |
|---|---|
| Table class or numerical value | `tabular_template.ipynb` |
| Text class, fast CPU route | `nlp_classification_template.ipynb` |
| Image class | `cv_classification_template.ipynb` |
| Binary semantic mask | `semantic_segmentation_template.ipynb` |
| Other outputs | [Notebook catalog and exact default contracts](docs/BASELINE_CATALOG.md) |

There are **16 competition-kit notebooks plus 14 learning notebooks**, including specialized
Task 2 profiles and a robustness
validation drill. Complete practice workflows still require adaptation to the released task's
metric, legal resources, schema, and output contract. Detection's default AP50 is explicitly a
practice evaluator; it is not COCO mAP.

## Use Colab correctly

For the self-contained learning notebooks, upload just the notebook and configure its data paths;
no repository bootstrap is needed. The following setup applies to the older competition-kit notebooks.

1. Put this repository at `/content/olp-ai-26` and open a generated notebook.
2. Use the bootstrap; do not run `uv sync` or install the project with pip in Colab.
3. Inspect `gpu_report()` before choosing batch size or precision.
4. Stage media under `/content/olp_runtime`; use Drive for input archives and durable output copies.
5. Save the complete solution recipe, including tokenizer/preprocessing, class order, and thresholds.

The bootstrap installs only requested missing dependencies, with constraints on the installed
numerical/CUDA stack. Precision is BF16 when supported, FP16 otherwise, and off on CPU.
See the [Colab runbook](docs/COLAB_RUNBOOK.md) for setup and recovery. Verify the organizer's
actual platform separately; prepared-code, internet, Drive, and personal Colab access are not assumed.

## Weight policy and correctness

Generic pretrained defaults are disabled. CV models use explicitly staged local files; HF helpers
are local-only by default. Task 2's specific permitted-weight profiles must not be reused as general
permission. A local checkpoint or tokenizer is legal only when its provenance is allowed.

Keep prediction IDs through inference and use ID-aware `build_submission`. Verify the exact
metric averaging, label set, empty/ignored cases, and required prediction type. Fit preprocessing
on training only; use grouped/time splits when identities or acquisition time can leak.

The shared trainer supports hardware-aware AMP, sample-weighted accumulation, clipping,
scheduling, early stopping, time reserves, and atomic best checkpoints. It reloads the best model
for inference; it does not promise exact training continuation with all scheduler/scaler/random state.

## Find a recipe or inspect readiness

- [Learning notebook pairs](notebooks/learning/README.md): representative datasets and exercises.
- [Method evidence](docs/KAGGLE_METHOD_EVIDENCE.md): documented results versus the reduced teaching code.
- [Learning scorecard](docs/LEARNING_SCORECARD.md): track independent implementation and transfer.
- [Guide index](docs/README.md): choose a document by what you want to do.
- [Customization cookbook](docs/CUSTOMIZATION_GUIDE.md): splits, metrics, weights, models, losses, TTA.
- [Implementation status](docs/IMPLEMENTATION_STATUS.md): repairs and verification boundaries.
- [Model catalog](docs/MODEL_CATALOG.md): supported model identifiers.
- [Archived Task 2 runbook](docs/TASK2_ANOMALY_DETECTION.md): normal-only anomaly case study.

## Develop and verify

Canonical notebook sources are `notebooks/*_template.py`. Regenerate artifacts after template edits:

```powershell
uv run ruff format .
uv run ruff check .
uv run olp-ai build-notebooks
uv run pytest
uv run python -m compileall -q src tests scripts notebooks
uv lock --check
git diff --check
```

Plain `uv sync` installs the core dependencies plus the default `dev`, `notebook`, and
`kaggle-extra` groups: local checks, Jupyter/plotting, and LightGBM for all 14 learning notebooks.
Use `uv sync --locked` to require the committed lockfile. For a smaller core/dev environment,
use `uv sync --locked --no-default-groups --group dev`.

Install extended task groups when needed:
`uv sync --locked --group cv-extra --group nlp-extra`.
To install every declared optional group too, use `uv sync --locked --all-groups`; this also
includes tracking, audio, and fast-data packages, which the learning notebooks do not require.
Tracking, audio, and fast-data packages remain separate. If the Windows cache is inaccessible,
use `uv --cache-dir .uv-cache ...`.

Keep `core/` task-agnostic and task adapters small. Follow [AGENTS.md](AGENTS.md) when changing code.
