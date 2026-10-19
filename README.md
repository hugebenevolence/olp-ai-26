# OLP AI 2026 practical preparation

Practice inspecting data, building a legal baseline, measuring the exact metric and producing a
valid submission within the first 45–60 minutes of a six-hour contest. Python 3.12 or newer.

**Start with [seven problem families and fourteen learning notebooks](notebooks/learning/README.md).**
Each pair has a fill-in baseline with collapsible answers and an explained advanced approach.
The learning notebooks are self-contained: copy one file, install its libraries and configure
its paths. The 16 competition-kit notebooks in `notebooks/` use the shared repository package.

| Next step | Guide |
|---|---|
| Follow a daily sequence | [15-day training plan](docs/15_DAY_PREPARATION.md) |
| Download and stage the real datasets | [Dataset guide](docs/LEARNING_DATASETS.md) |
| Rebuild a baseline without saved code | [Blank-notebook playbook and 2k assistant strategy](docs/BLANK_NOTEBOOK_PLAYBOOK.md) |
| Run a worked example with the repository | [Practice quickstart](docs/PRACTICE_QUICKSTART.md) |
| Find a task, model or adaptation recipe | [Documentation index](docs/README.md) |

## Local setup

```powershell
uv sync --locked
uv run olp-ai doctor
```

The default groups include checks, Jupyter/plotting, LightGBM and the official Kaggle client.
For extended CV/NLP tools, use `uv sync --locked --group cv-extra --group nlp-extra`.
Use `uv sync --locked --all-groups` only when you also need every optional group, including
tracking, audio and fast-data. For a smaller core/dev environment:
`uv sync --locked --no-default-groups --group dev`.
If the Windows cache is inaccessible, use `uv --cache-dir .uv-cache ...`.

Create small offline fixtures before downloading real data:

```powershell
uv run olp-ai practice-data data/learning --task learning
```

For the repository-assisted quickstart instead, create `data/practice` with
`uv run olp-ai practice-data data/practice --task all`. Fixture scores test the workflow only;
keep grading answers closed and never add fixtures to official contest training data.

## Colab and contest use

Upload a learning notebook directly; kit notebooks require the checkout at `/content/olp-ai-26`.
Follow the [Colab runbook](docs/COLAB_RUNBOOK.md): preserve Colab's numerical/CUDA stack, inspect
its actual GPU, train from `/content/olp_runtime`, and sync complete run bundles to durable storage.
Verify the organizer's allowed environment separately; internet, prepared code, Drive and personal
Colab access are not assumed on contest day.

Pretrained defaults are disabled. Local checkpoints and tokenizers still require explicit task
permission; the [archived Task 2 case study](docs/archive/README.md) grants no general permission.
Fit preprocessing on training only, use leakage-aware splits, retain prediction IDs and verify
the organizer's exact metric and output contract before tuning. Generic detection AP50 is a
practice scorer, not COCO mAP. See [current readiness and verification limits](docs/IMPLEMENTATION_STATUS.md).

Competition data, credentials and personal notebooks are ignored by Git. Outputs embedded in
tracked notebook templates still need clearing before a commit. Work in a personal copy such as
`01_credit_attempt.ipynb`; rebuilding replaces generated templates.

## Develop and verify

Edit canonical `notebooks/**/*_template.py` sources, then regenerate their `.ipynb` artifacts.
The adjacent Python source is never needed to execute a self-contained learning notebook.

```powershell
uv run ruff format .
uv run ruff check .
uv run olp-ai build-notebooks
uv run pytest
uv run python -m compileall -q src tests scripts notebooks
uv lock --check
git diff --check
```

Keep task adapters small and `core/` task-agnostic. Follow [AGENTS.md](AGENTS.md) when changing code.
