# Google Colab Pro competition runbook

[Guide index](README.md) · [First practice session](PRACTICE_QUICKSTART.md)

## Choose the notebook route

**Self-contained learning notebooks:** upload one of the [14 lessons](../notebooks/learning/README.md)
directly. Install only missing small libraries from `requirements-colab.txt` before practice;
never replace Colab's torch, torchvision, NumPy or CUDA packages. The notebook installs nothing.
Configure `DATA_DIR`, `DATA_ARCHIVE` and `PERSISTENT_DIR` in stage 1, then inspect `gpu_report()`.
Use [stage 1a](LEARNING_DATASETS.md#download-from-a-learning-notebook) only when online download
is intended; it is disabled by default. No checkout or project bootstrap is needed.

**Repository-assisted kit notebooks:** the setup below applies to the 16 notebooks in
`notebooks/` outside `learning/`. Keep the checkout at `/content/olp-ai-26`.

## Repository-assisted setup

- Generate every notebook with `uv run olp-ai build-notebooks` and upload/clone the repository to
  `/content/olp-ai-26`.
- Put immutable input archives on Drive; train from extracted files under `/content/olp_runtime`.
- Cache only competition-permitted model/tokenizer files and record their licenses and hashes.
- Execute one small end-to-end run: split, train, validate, infer, validate submission, sync.

1. Select a GPU runtime.
2. Run the bootstrap cell. It adds `src/` to `sys.path` and installs only the notebook's requested
   missing dependencies. Constraints pin the existing torch/torchvision/NumPy/CUDA packages.
   A resolver conflict fails clearly; do not remove the constraints to force installation.
3. Read `gpu_report()`; do not assume T4, L4, A100, memory size, or BF16 support.
4. Set `PERSISTENT_DIR` for checkpoints/submissions and `DRIVE_DATA_ARCHIVE` for the input archive.
5. Call `stage_data()` once, then repeatedly read from `/content`, not mounted Drive.

Record package versions before and after first bootstrap. A live GPU install remains a required
rehearsal; local tests only inspect the bootstrap's behavior with a simulated resolver.

## Cold reload and durable artifacts

Each kit task writes into its own output subdirectory; learning candidates use unique run IDs.
Preserve a whole run before changing
configuration: fitted pipeline or checkpoint, tokenizer/processor files, label order, thresholds,
`run.json`, report, submission, and executable notebook/source. Sync the task directory to a
durable destination outside that directory. Include the final ZIP when the task requests one.
Save a personal copy of your edited notebook too; an unsaved browser document cannot be captured
by artifact synchronization. Learning checkpoints back up to their configured durable run folder.

The CPU text and tabular models have `save`/`load` methods. For torch/HF models, construct the
recorded architecture and preprocessing first, then load the matching state. Reproducing inference
is separate from exact training continuation; the common trainer does not restore all scheduler,
scaler, and random-generator state for an exact resume.

## Six-hour safeguards

- Produce the first syntactically valid submission in 45-60 minutes.
- Save the best checkpoint atomically and sync it after each meaningful run.
- Keep the official metric, split seed/grouping, model, resolution, optimizer, and weights policy in
  the experiment record.
- Reserve the final 30 minutes for inference and submission validation; start no new training.

## Common mistakes

- A random row split can leak subject, video, document, or timestamp identity.
- Dice and IoU are not interchangeable with the organizer's exact metric implementation.
- Detection class 0 is background in torchvision.
- Local weight files still need exact task permission. CV helpers require staged files and HF
  loaders are local-only; do not bypass these checks to download a model during a restricted run.
- TTA and ensembles increase inference time and are not automatically beneficial.
- A notebook that runs locally is not evidence that the clean Colab runtime is ready.
