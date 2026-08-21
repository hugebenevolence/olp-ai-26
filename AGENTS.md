# Repository instructions

## Purpose

This repository is a six-hour OlympicAI Vietnam competition kit for CV, NLP, and multimodal
tasks. Optimize for a correct local metric and valid submission within the first 45–60 minutes.
Keep task adapters replaceable; do not turn the project into a large training framework.

## Package and environment workflow

- Use `uv` for local dependency management and command execution.
- Local setup: `uv sync --group dev`.
- Install optional groups only when needed:
  `uv sync --group dev --group cv-extra --group nlp-extra --group notebook`.
- Keep heavyweight tracking, audio, and fast-data packages outside the default environment.
- Update and commit `uv.lock` whenever dependency declarations change.
- If the global uv cache is inaccessible on Windows, use `uv --cache-dir .uv-cache ...`.

## Google Colab Pro rules

- Hosted Colab GPU is the target execution environment.
- Do not run `uv sync` in Colab. Do not replace Colab's CUDA-matched `torch`, `torchvision`,
  NumPy, or CUDA packages.
- Do not install this project with pip. The notebook bootstrap adds `src/` to `sys.path`.
- Install only missing small dependencies from `requirements-colab.txt`.
- Expect Python 3.12, but do not assume a particular GPU model or amount of GPU memory.
- Use `gpu_report()` and runtime device detection before selecting batch size or precision.
- Mixed precision must remain hardware-aware: BF16 when supported, FP16 otherwise, and off on CPU.
- Train and repeatedly read media under `/content/olp_runtime`, not mounted Google Drive.
- Use Drive only for the input archive and durable copies of checkpoints, weights, submissions,
  configurations, and reports.
- Keep Colab paths and persistence behavior in `src/olp_ai_26/core/colab.py`.

## Notebook contract

- Files matching `notebooks/*_template.py` are the canonical notebook sources.
- Generated `notebooks/*_template.ipynb` files are artifacts; never edit them manually.
- Regenerate notebooks after any template or exporter change:
  `uv run olp-ai build-notebooks`.
- Every competition notebook must retain these stages:
  bootstrap, configuration, environment/data inspection, leakage-aware split, model/loss/training,
  validation, inference, submission validation, and artifact synchronization.
- Keep competition-specific values in the drag-and-plug configuration cell.
- The repository must be available at `/content/olp-ai-26` in Colab unless the notebook's
  `PROJECT_ROOT` is explicitly changed.

## Architecture boundaries

- `core/` must remain task-agnostic.
- `cv/`, `nlp/`, and `multimodal/` contain task-specific datasets, models, losses, and decoders.
- `validation/` contains leakage and integrity checks.
- Optional dependencies must be imported lazily when practical so core and tests run without all
  optional groups installed.
- Reuse `CompetitionConfig`, `TrainerConfig`, `Trainer`, metric registry, split utilities, and
  submission validation instead of creating parallel implementations in notebooks.
- Keep CLI behavior in `src/olp_ai_26/cli.py`.

## Competition integrity and model policy

- Default `pretrained_allowed` to `False`.
- Never silently download model weights. Hugging Face loaders should remain local-only by default.
- Use pretrained weights only when the task explicitly permits the exact model or checkpoint.
- Do not add external training data unless the task explicitly permits it.
- Do not train on test data or pseudo-label the test set unless the task explicitly authorizes it.
- Reproduce the official metric locally; do not substitute a similar metric.
- Prefer grouped or chronological splits whenever identities, specimens, speakers, videos, or time
  can leak between train and validation.
- Validate submission columns, row count, ID order, missing values, encoding, and allowed labels.

## Coding conventions

- Support Python 3.12 or newer and use `pathlib.Path` for paths.
- Add type hints to public functions and use dataclasses for configuration/state objects.
- Keep random seeds, device, paths, time budget, and weight policy explicit.
- Use atomic checkpoint writes and preserve the best checkpoint.
- Avoid network-dependent logging in the baseline path; prefer local JSONL and Markdown reports.
- Avoid hard-coding Windows paths in package code or notebook templates.
- Do not optimize or ensemble without validation evidence.
- Preserve user changes and avoid unrelated rewrites.

## Required verification

Run these after implementation changes:

```powershell
uv run ruff format .
uv run ruff check .
uv run olp-ai build-notebooks
uv run pytest
uv run python -m compileall -q src tests scripts notebooks
uv lock --check
git diff --check
```

When only documentation changes, use proportionate checks. When notebook templates change, always
regenerate the `.ipynb` files and run the notebook tests. When trainer, precision, data staging, or
checkpoint code changes, run the full suite.

## Reporting expectations

- Lead with what changed and whether it works.
- State exact commands and test counts.
- Distinguish local CPU verification from live Colab GPU verification.
- Call out mistakes directly, especially metric mismatch, data leakage, invalid submissions,
  training directly from Drive, replacing Colab's PyTorch stack, and assuming pretrained weights
  are legal.
