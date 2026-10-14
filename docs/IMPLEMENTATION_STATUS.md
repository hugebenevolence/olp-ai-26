# Current implementation status

Updated 14 October 2026. This page describes the repairs after the
[historical audit](READINESS_AUDIT_2026-10-14.md); it is not live Colab certification.

[Guide index](README.md) · [Start a practice session](PRACTICE_QUICKSTART.md)

## Correctness and runtime repairs

- Submissions can align unique single/composite prediction IDs to sample order. Missing, duplicate,
  or extra keys fail before writing. CSV writes are atomic and reject non-finite numeric values.
  Literal CSV IDs such as `001` and `NA` are preserved; CLI validation has an explicit opt-in for
  task-permitted empty text/RLE fields.
- Accumulation groups, including uneven final batches and interrupted groups, use sample-weighted
  averaging, clipping, and consistent optimizer/scheduler updates. Empty validation fails clearly;
  an expired run cannot silently reload an old checkpoint. AMP overflow skips scheduler advancement.
- Dice/IoU distinguish global and mean-per-image averaging and expose empty-mask/ignored-pixel
  conventions. Probability metrics include AUROC, average precision, and log loss.
- Segmentation uses identical train/validation/test normalization. Checkpoint selection and final
  TTA inference use original-resolution masks and the chosen foreground-label policy.
- Generic pretrained defaults are disabled. CV weight loading requires staged local files; timm
  can adapt the classifier head from the permitted original checkpoint. HF loaders stay local-only.
- Colab installs requested missing dependencies with constraints on installed torch, torchvision,
  NumPy, Triton, and NVIDIA CUDA distributions. No live resolver/install on Colab was performed.
- Artifact sync includes fitted pipelines, tokenizer text files, ZIPs, notebooks, and source files;
  it rejects a destination nested under its own source.
- HF best-model reload uses the model's local key-conversion loader and fails on missing or
  unexpected state. This fixes a ViT encoder key-renaming problem found during the caption smoke run.

## Added or completed workflows

- CPU tabular classification/regression: dummy, linear, and bounded histogram boosting, with
  train-only preprocessing, unknown-category handling, saved pipeline, and explicit prediction type.
- Text/CV/video classification: ID alignment, saved model/label recipe, validation record, and report.
- NER: shared group/time split, entity-F1 checkpoint selection, strict training truncation checks,
  overflow-aware word-level inference, and JSON-list output.
- Seq2seq/captioning: generated-text validation, complete test output, saved model/tokenizer/processor,
  explicit decoding settings, and wall-clock callback. Caption processors can pair separate image
  and text processors. Exact multi-reference/CIDEr evaluation remains a task-specific adapter.
- Retrieval: bounded query batches, saved sparse index, validation MRR@K, and query/rank ID alignment.
- Detection/instances: evaluation-selected atomic checkpoints and full test JSON output. The default
  integral AP50 is **only a practice scorer**; replace it with the official evaluator for contest use.
- Robustness drill: uses a saved classifier and its real validation fold instead of random images
  and labels; normalization sits inside the model so attacks remain in bounded pixel space.
- `olp-ai practice-data`: reproducible offline text/tabular/image/mask packs with deliberately shuffled
  sample IDs and separate grading answers.

## Notebook reconciliation

All 16 notebooks are generated from canonical sources. The two old anomaly artifacts were preserved
in [the backup directory](../artifacts/notebook_backups/2026-10-14/README.md) before regeneration,
including their additional setup cell, settings, and older implementation. No direct `.ipynb` edits
were used to produce the current artifacts. Tests now enforce source/artifact equality.

Task 2 profiles retain their task-specific pretrained permission. They now require each selected
model's permitted weight file via the existing checkpoint configuration. A pretrained flag with
no local file raises a clear error; it does not initiate a download.

## Verification boundaries

The CPU suite includes behavioral regression tests and full synthetic text/tabular notebook runs
through persistence, inference, CSV validation, and report creation. Other notebook cells are
compiled, and their adapters have targeted component tests. Compilation is not whole-notebook
execution. Synthetic-pack scores are not evidence of real-dataset performance.

Separately, tiny local BERT NER, encoder-decoder seq2seq, and ViT+BERT captioning models completed
training, validation, best-checkpoint reload, full test inference, and reporting on CPU without
downloads. The generation checks exposed and fixed unlabeled trainer inference and processor
selection issues. These small smoke runs test compatibility and delivery, not model quality.

Still required during your preparation: clean Colab GPU execution, actual optional dependency
installation, realistic-dataset training, restart/persistence on Drive, and a team rehearsal in the
organizer's permitted environment. Exact COCO, CIDEr, multiclass segmentation, unusual ranking,
and task-specific attack/output rules must use their official adapters.

## Required local checks

```powershell
uv run ruff format .
uv run ruff check .
uv run olp-ai build-notebooks
uv run pytest
uv run python -m compileall -q src tests scripts notebooks
uv lock --check
git diff --check
```

For this Windows sandbox, tests use an explicit unused workspace temporary directory:
`uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/preparation-delivery --tb=short`.
The existing environment is retained with `--no-sync`; no dependency declarations or lockfile
changes were needed. Other checks use the same cache/no-sync options. The final test count and
results are recorded below.

## Final verification — 14 October 2026

| Exact command | Result |
|---|---|
| `uv --cache-dir .uv-cache run --no-sync ruff format .` | Passed; 86 files formatted. |
| `uv --cache-dir .uv-cache run --no-sync ruff check .` | Passed. |
| `uv --cache-dir .uv-cache run --no-sync olp-ai build-notebooks` | Regenerated all 16 notebooks. |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/preparation-delivery --tb=short` | **69 passed**, one Windows CPU-core discovery fallback warning, 27.58 seconds. |
| `uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests scripts notebooks` | Passed. |
| `uv --cache-dir .uv-cache lock --check` | Passed; 254 packages resolved. |
| `git diff --check` | Passed. |
| `uv --cache-dir .uv-cache run --no-sync olp-ai practice-data data/practice --task all` | Created all four local practice packs. |

The shell invoked the installed uv executable by its absolute installation path because it was
not on PATH. The commands above retain their exact arguments in a portable form.
Fourteen Markdown documents passed local-link, fence-balance, and whitespace checks. The manual
offline HF smoke runs used small random local models and are separate from the 69-test count.
No live Colab GPU verification or real-dataset score is claimed.
