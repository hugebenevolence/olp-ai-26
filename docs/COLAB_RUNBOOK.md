# Google Colab Pro competition runbook

## Before the event

- Generate every notebook with `uv run olp-ai build-notebooks` and upload/clone the repository to
  `/content/olp-ai-26`.
- Put immutable input archives on Drive; train from extracted files under `/content/olp_runtime`.
- Cache only competition-permitted model/tokenizer files and record their licenses and hashes.
- Execute one small end-to-end run: split, train, validate, infer, validate submission, sync.

## Runtime setup

1. Select a GPU runtime.
2. Run the bootstrap cell. It adds `src/` to `sys.path` and installs only missing small packages.
3. Read `gpu_report()`; do not assume T4, L4, A100, memory size, or BF16 support.
4. Set `PERSISTENT_DIR` for checkpoints/submissions and `DRIVE_DATA_ARCHIVE` for the input archive.
5. Call `stage_data()` once, then repeatedly read from `/content`, not mounted Drive.

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
- `pretrained=True` may trigger a network download and violate the rules.
- TTA and ensembles increase inference time and are not automatically beneficial.
- A notebook that runs locally is not evidence that the clean Colab runtime is ready.
