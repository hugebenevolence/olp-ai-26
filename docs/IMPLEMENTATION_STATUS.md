# Current implementation status

Updated 18 October 2026. This page describes the repairs after the
[historical audit](READINESS_AUDIT_2026-10-14.md); it is not live Colab certification.

[Guide index](README.md) · [Start a practice session](PRACTICE_QUICKSTART.md)

## Paired learning curriculum — 18 October 2026

Start with [the seven notebook pairs](../notebooks/learning/README.md). There are **14 new learning
notebooks**, generated from adjacent canonical sources, alongside the 16 existing competition-kit
notebooks. Each baseline has 11 competition stages and one optional dataset acquisition cell
(12 code cells), with a closed HTML-details solution immediately after every code cell:
84 solution cells across seven baselines. Marked answer blocks become small
`NotImplementedError` exercises while the supplied setup/inspection scaffold remains executable.
Advanced notebooks are worked component studies with primary evidence and explicit omissions;
they are not claimed SOTA reproductions or universally fastest methods.

Coverage: Home Credit classification, House Prices regression, Jigsaw multilabel text, Cassava
image classes, TGS masks, Wheat detection and Shopee multimodal matching. Each route includes
inspection/plots, training-fold imbalance diagnosis and response, leakage checks, saved candidates,
validation/error plots, reload, ID-aware inference, checked output and a configuration/report bundle.
Local data defaults to the seven staged schema fixtures. Colab uses fast runtime storage and the
notebook's inline Drive mount function; each durable candidate gets a separate run-ID folder.
Trainer best checkpoints back up to that same unique directory.

All 14 learning notebooks are now **self-contained**. Project imports, repository-root discovery,
search-path changes and external bootstrap execution were removed. Storage/configuration,
leakage-aware splitting, datasets, model/loss construction, hardware-aware training, exact task
metrics, ID alignment, CSV validation and reporting are defined visibly in the lesson itself.
Only installed third-party libraries, staged data and explicitly permitted local assets are
required. Nothing installs packages or fetches code/weights while a learning notebook runs.
The older 16 competition-kit notebooks remain package-based references, clearly distinguished
in the learning index and root README.

Each learning notebook now includes **stage 1a**, an optional original-competition Kaggle
download. It is disabled by default, uses the official CLI through the notebook's interpreter,
reuses complete data/caches, extracts nested image/CSV ZIPs and checks the expected layout
before publishing the folder. Existing unmarked data folders are preserved. Paths escaping an
archive's extraction root and overlapping cache/data folders are rejected. Failed preparation
leaves a temporary folder for inspection and never replaces the data directory. Credentials
come from Kaggle's configuration/environment or hidden input; client output is withheld from
saved notebook output. No competition rules are accepted and no submissions or weights are
uploaded/downloaded. Acquisition happens before the training timer restarts. Offline execution
continues to use staged data with `DOWNLOAD_DATA=False`.

Download tests simulate the official client's response for all seven raw dataset layouts,
including TGS train/test ZIPs and separately zipped CSVs. They verify disabled behavior,
cache/data reuse, layout failures, traversal rejection, preservation of existing folders, and
client-error redaction. They do not access a Kaggle account or fetch the real datasets.

The inline torch loop deliberately supports AdamW, one update per batch and no scheduler;
unsupported precision/scheduler settings raise an error. It uses BF16 when supported, FP16
otherwise on CUDA, FP32 on CPU, gradient clipping, global validation, a delivery reserve,
atomic best-checkpoint writes and best-state reload. Detection keeps its separate visible FP32
loop. A final boundary check caught a chronological-split bug: a large equal-timestamp block
could otherwise produce an empty training fold. All standalone copies now choose the nearest
legal boundary between distinct timestamps and reject overlapping group/time splits.

The implementation also adds positive-pixel weighting to `DiceBCELoss`, explicit log-price/TGS/
Wheat/matching metric helpers, and feature-cache/plot synchronization. Wheat's public statement
and winner reference differ at boundaries; the helper exposes the convention and tests it rather
than silently using COCO AP. The matching execution test caught and fixed an ambiguous sparse
matrix length call. No test pseudo-labeling or new external training data path was introduced.

The `kaggle-extra` group contains LightGBM and now the official Kaggle client (`kaggle>=2.2,<3`).
The notebook group supplies plotting and Jupyter tools.
`pyproject.toml` now selects `dev`, `notebook`, and `kaggle-extra` by default, so plain `uv sync`
prepares all 14 learning notebooks. Previously, a core/dev sync omitted those tools and removed
them from the local environment. Extended CV/NLP, tracking, audio and fast-data remain opt-in;
`uv sync --locked --all-groups` selects every declared group. The earlier default-group selection
introduced no package constraints or versions; its default sync and compatibility check passed
with 161 packages. The download client subsequently added six locked packages (261 in the full
lockfile), retaining all existing versions. It is also listed in `requirements-colab.txt` for
manual installation in a copied notebook runtime when missing. The initial sandboxed installation hit
a Windows launcher-resource permission error; completing the installation with normal local
permissions succeeded.

Dependency declarations and `uv.lock` were committed separately as `77df1a3`
for the original paired lessons. The download client's dependency declarations, Colab list and
updated lockfile were committed as `e60b4b7`, as required by AGENTS.md. Notebook code and
documentation remain available as working-tree changes. Existing optional local tools were
retained with an all-groups sync using their cached packages; no existing package version changed.

| Exact command | Final result |
|---|---|
| `uv --cache-dir .uv-cache sync --group dev --group notebook --group kaggle-extra --locked` | Installed the declared optional tools locally; existing torch stack retained. |
| `uv --cache-dir .uv-cache sync --locked --link-mode copy` | Installed the default preparation groups, including notebook tools and LightGBM. |
| `uv --cache-dir .uv-cache sync --locked` | Second sync passed with no missing packages. |
| `uv --cache-dir .uv-cache pip check` | All 161 installed packages compatible. |
| `uv --cache-dir .uv-cache run --no-sync ruff format .` | Passed; final run left 111 files unchanged. |
| `uv --cache-dir .uv-cache run --no-sync ruff check .` | Passed. |
| `uv --cache-dir .uv-cache run --no-sync olp-ai build-notebooks` | Generated all 30 notebooks. |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/standalone-final --tb=short` | **105 passed**, one Windows CPU-core fallback warning, 91.51 seconds. |
| `uv --cache-dir .uv-cache run pytest --basetemp .pytest_cache/default-sync-final --tb=short` | **105 passed**, one Windows CPU-core fallback warning, 172.15 seconds, after the default sync. |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/download-final --tb=short` | **113 passed**, one Windows CPU-core fallback warning, 246.18 seconds, with the download cells and optional local tools installed. Optional wandb also printed ignored temporary-folder cleanup permission errors at process exit in the Windows sandbox. |
| `uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests scripts notebooks` | Passed. |
| `uv --cache-dir .uv-cache lock --check` | Passed; 261 packages resolved after the Kaggle client addition. |
| `uv --cache-dir .uv-cache pip check` (after download-client setup) | All 238 installed packages compatible, including the user's existing optional groups. |
| `uv --cache-dir .uv-cache sync --locked --all-groups --dry-run` | Passed; would make no changes. |
| `git diff --check` | Passed. |
| `uv --cache-dir .uv-cache run --no-sync olp-ai practice-data data/learning --task learning` | Created all seven offline schema fixtures. |

Additional verification validates all 14 learning artifacts against the Jupyter schema and checks
canonical/artifact equality, 11-stage plus acquisition exercise/solution placement, malformed exercise boundaries,
execution in separate working folders with both network connections and project imports blocked,
inline metric fixtures, positive-mask-weight gradients, feature/plot copying,
and two independent durable bundles. Fresh isolated Python processes (`python -I`) also complete
the table and image baseline answers with project imports/network blocked. Inline split copies
are checked for identical guards; storage tests reject escaping ZIP members and nested backups.
The matching route executes both its feature-cache path and
its local-encoder path using tiny random BERT and ResNet models, with no downloads. Two rendered
inspection/segmentation plots were visually checked.

All training/execution checks above are **local CPU**, with small synthetic data and short epochs.
No real Kaggle dataset performance, leaderboard placement, Colab GPU run, organizer upload,
DeepSeek interface or live notebook-viewer collapse behavior is certified. HTML details start
closed structurally; rendering depends on the viewer. The fixtures cannot establish the advanced
models' superiority. Follow the [data guide](LEARNING_DATASETS.md),
[evidence ledger](KAGGLE_METHOD_EVIDENCE.md) and [learning scorecard](LEARNING_SCORECARD.md) for
the actual preparation exercises.

## Blank-notebook preparation — 18 October 2026

The [blank-notebook playbook](BLANK_NOTEBOOK_PLAYBOOK.md) remains the reconstruction assessment path.
The 15-day program starts reconstruction on Day 1, uses the stated 2k-context DeepSeek deployment
for individual functions, and includes two assistant-free assessment sessions. The standalone
learning notebooks remain the untimed study and comparison path; the older kit is a separate
repository-assisted reference.

The standalone [text/table answer key](../scripts/blank_baseline.py) uses no package imports from
this project. Isolated Python processes outside the repository complete fitting, validation,
persistence, shuffled-ID delivery, and fresh-process inference from the saved pipeline. Additional
checks cover literal IDs/labels, invalid prediction keys, grouped splits, and equal-timestamp
boundaries. All four executable [image exercise](BLANK_NOTEBOOK_CV.md) cells run through real
training, best-checkpoint reload, decoding, CSV alignment and saved recipe on the small image pack.
These are intentionally independent educational reconstructions, not replacements for the shared
production trainer or its richer task adapters.

Final local verification for this addition:

| Exact command | Result |
|---|---|
| `uv --cache-dir .uv-cache run --no-sync ruff format .` | Passed; final run left all 90 files unchanged. |
| `uv --cache-dir .uv-cache run --no-sync ruff check .` | Passed. |
| `uv --cache-dir .uv-cache run --no-sync olp-ai build-notebooks` | Regenerated all 16 artifacts; no notebook diff. |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/blank-notebook-delivery --tb=short` | **77 passed**, one Windows CPU-core discovery fallback warning, 67.37 seconds. |
| `uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests scripts notebooks` | Passed. |
| `uv --cache-dir .uv-cache lock --check` | Passed; 254 packages resolved. No dependency changes. |
| `git diff --check` | Passed. |

The shell used the absolute installed uv executable because uv is not on PATH. A separate local
Markdown check verified 15 documents, 74 local links, code-fence balance and whitespace.
All execution was local **CPU** (PyTorch 2.13.0+cpu, sklearn 1.9.0). No live Colab GPU, real-dataset
benchmark, organizer upload, or supplied DeepSeek interface was tested. Prompt sizing and coding
time goals are practice guidance, not measured guarantees for that assistant or contest hardware.

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
