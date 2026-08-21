# OLP AI 2026 competition kit

Reusable, offline-first baselines for a six-hour OlympicAI Vietnam workflow. The package is
organized around a common inspection, validation, training, inference, and submission layer so
task-specific code stays replaceable.

## Quick start

```powershell
uv sync --group dev
uv run olp-ai doctor
uv run olp-ai inspect data
uv run olp-ai build-notebooks
uv run pytest
```

Generated notebooks are written next to the `# %%` templates in `notebooks/`.

Install task-specific extras only when the released problem needs them:

```powershell
uv sync --group dev --group cv-extra --group nlp-extra --group notebook
```

`tracking`, `fast-data`, and `audio` are separate groups because they are not required by the
baseline path and materially increase setup time.

## Google Colab Pro

The generated notebooks target the hosted Colab GPU runtime. Current pin-able 2026 Colab
runtimes use Python 3.12 and PyTorch 2.9-2.11, so do **not** run `uv sync` inside Colab: the local
lock may replace Google's CUDA-matched PyTorch build. See the official
[runtime-version list](https://research.google.com/colaboratory/runtime-version-faq.html).

1. Select **Runtime → Change runtime type → GPU**.
2. Upload or clone this repository to `/content/olp-ai-26`.
3. Open one of the generated `.ipynb` files and run its Colab bootstrap cell.
4. Set `DRIVE_DATA_ARCHIVE` to a `.zip`/`.tar.gz` on Drive, or upload the extracted data directly
   to `/content/olp_runtime/data`.
5. Optionally set `PERSISTENT_DIR` to a Drive folder for best-checkpoint and submission backups.

The bootstrap adds the repository's `src/` directory directly to `sys.path`; it does not build or
install the project. This preserves Colab's PyTorch, torchvision, NumPy, and CUDA stack and avoids
downloading the `uv_build` backend. It installs only the small packages in
`requirements-colab.txt` when they are missing.

Training and repeated media reads happen under `/content`, not Google Drive. Only checkpoints,
submissions, configurations, and reports are copied back to `PERSISTENT_DIR`. This avoids the
Drive latency and I/O failure mode described in the official
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

Mixed precision is hardware-aware:

- BF16 when `torch.cuda.is_bf16_supported()` is true.
- FP16 plus gradient scaling on older GPUs such as T4.
- Disabled automatically on CPU.

Always inspect the output of `gpu_report()` at the beginning of the session. Colab Pro improves
access, but Google does not guarantee a particular GPU type or fixed resource limits.

## First 45 minutes of a contest

1. Read the task, metric, model allowlist, submission limit, and private-test procedure.
2. Run `olp-ai doctor` and `olp-ai inspect data`.
3. Identify train, test, sample-submission, input, target, ID, and grouping columns.
4. Reproduce the official metric locally.
5. Check duplicate examples and group leakage before choosing a split.
6. Change only the configuration cell in the nearest notebook template.
7. Train the cheapest baseline and validate the resulting submission file.
8. Submit once before changing models or augmentations.

## Available components

### Shared core

- `core.config`: reproducibility, device resolution, trainer settings, and wall-clock budget.
- `core.colab`: `/content` staging, GPU reporting, precision flags, Drive backup, and loaders.
- `core.inspect`: table/media inventory, corrupt-image detection, duplicate hashes, environment report.
- `core.split`: random, stratified, grouped, stratified-grouped, and chronological holdouts.
- `core.metrics`: accuracy/F1, seqeval F1, regression, Dice/IoU, BLEU/chrF, ROUGE-L,
  PSNR, and SSIM.
- `core.trainer`: AdamW/Adam/SGD/Adafactor, AMP, accumulation, clipping, schedulers, checkpoints.
- `core.inference`: batch inference, TTA averaging, timing, and weighted ensembles.
- `core.submission`: row, column, ID-order, missing-value, and allowed-label validation.
- `core.experiment` and `core.report`: offline JSONL tracking and technical-report generation.
- `validation.leakage`: normalized-text duplicates, file hashes, and cross-split overlap.

### CV

- timm image classification and reusable torchvision transforms.
- Frame-sampled video/activity classification with a 2D backbone and temporal pooling.
- SMP U-Net/FPN/DeepLabV3+ factories and Dice+BCE loss.
- Local-only Ultralytics YOLO training adapter.
- FGSM and PGD attacks with perturbation auditing.

### NLP and multimodal

- CPU-first word+character TF-IDF logistic-regression classifier.
- Local-only Hugging Face sequence and token classifiers.
- Translation/seq2seq tokenization and model factory.
- TF-IDF retrieval plus reciprocal-rank evaluation.
- Local image-captioning model factory and generation helper.

## Notebook templates

- `notebooks/cv_classification_template.py`
- `notebooks/video_classification_template.py`
- `notebooks/nlp_classification_template.py`
- `notebooks/seq2seq_template.py`
- `notebooks/image_captioning_template.py`

They are source-controlled as Python because diffs remain readable. Export them with:

```powershell
uv run olp-ai build-notebooks
```

The resulting `.ipynb` files assume this package is available. Before the event, test notebook
execution against the exact official Colab/Kaggle image. Do not assume the local lock file or
private repository will be available in the final room.

## Weight and data policy

Pretrained models are opt-in. Hugging Face helpers default to `local_files_only=True`, and the
YOLO adapter requires an explicit local file. Set `pretrained_allowed=True` only when the task's
model allowlist permits the exact weights. Do not pseudo-label or otherwise train on the test set
unless the task explicitly authorizes it.

## Six-hour operating plan

| Time | Outcome |
|---|---|
| 00:00–00:30 | Read rules, inspect data, reproduce metric, assign one owner per task |
| 00:30–01:00 | Valid baseline and submission for both tasks |
| 01:00–03:30 | Controlled model, split, loss, and augmentation experiments |
| 03:30–04:45 | Error analysis, stronger run, inference timing |
| 04:45–05:30 | TTA/ensemble only when validation evidence supports it |
| 05:30–06:00 | Final inference, validated files, weights, notebook, technical report |

Never spend the final 30 minutes starting a new training run.

## Adding a task

Keep the core unchanged. A new adapter should define only:

1. Dataset and preprocessing.
2. Model construction under the explicit pretrained-weight policy.
3. Loss and prediction decoding.
4. Official metric inputs.
5. Submission-column mapping.
