# Repository readiness audit — 14 October 2026

**Historical snapshot before repairs.** For the current code and verification boundaries, use
[implementation status](../IMPLEMENTATION_STATUS.md) and the [guide index](../README.md).

**Verdict: useful competition foundation, with incomplete task workflows and confirmed correctness risks.**
The repository is ready to support deliberate practice. It is not yet demonstrated ready for an
unseen six-hour contest. This assessment covers commit `7515cbc`, source inspection, local CPU
checks, and three focused probes. It does not certify live Colab GPU execution or organizer-cloud
compatibility. No training implementation or existing generated notebook was changed during this audit.

The personal preparation plan is in [15_DAY_PREPARATION.md](../15_DAY_PREPARATION.md).

## What already helps

- Shared configuration, deterministic splits, leakage checks, local experiment logging, reports,
  inference helpers, and submission checks reduce duplicated work.
- The common trainer saves its best checkpoint atomically, supports accumulation and hardware-aware
  precision, and can back up checkpoints.
- Colab staging supports fast local media access and durable artifact copies.
- TF-IDF classification is a useful CPU fallback. CV classifiers, masks, retrieval, and task-specific
  anomaly detection offer several starting points.
- There are **15** canonical notebook templates. The README's count of 14 is stale.
- The prior Task 2 runbooks document experiment evidence and the limits of synthetic proxy metrics.
  Those records are useful case studies; their reported scores were not independently reproduced
  here and do not establish the scope or rules of OlpAI'26.

## Findings and acceptance criteria

Priorities reflect preparation value and failure consequences, not official task weights.

| Priority | Finding and evidence | Consequence | Acceptance criterion |
|---|---|---|---|
| P0 | Both generated anomaly notebooks have cell-source drift from their canonical `.py` files. The DINO notebook contains `!git clone` in a code cell, failing the repository's Python-cell compilation test. | The checked-in artifact does not reproduce the canonical source. Colab shell syntax itself can be valid, but it violates the current exporter/test contract. | Reconcile useful setup instructions into canonical sources, regenerate, and pass notebook tests. Preserve the existing notebook edits before reconciliation. |
| P0 | Generic transformer and NER templates set `PRETRAINED_ALLOWED=True`; both anomaly extractor constructors default `pretrained_allowed=True` and can download weights. | Reusing a helper/template can enable weights before task permission is established. | Generic defaults deny pretrained weights; an explicit task profile records exact permitted weights and local availability. Legal, local, and cached are three separate checks. |
| P0 | Generic classification notebooks predict in test-table order and then copy the sample's IDs. The shared validator only compares the resulting IDs with the sample. | Correct predictions can be assigned to the wrong IDs while the CSV passes validation. | Carry prediction IDs through inference; join/reindex to the sample with the task's correct key, then assert cardinality and key equality. A deliberately permuted test manifest must still produce the correct ID-to-prediction mapping. |
| P0 | `core.metrics._dice` and `_iou` aggregate over the entire array. The registry has no generic AUROC, average precision, log loss, or detection mAP entry. | A plausible score can mismatch the official averaging, label policy, empty-mask convention, or prediction type. | Compare the organizer evaluator against hand-calculated fixtures; register only the exact needed adapter. Do not interpret absent metrics as unavailable task-specific metrics elsewhere in the repo. |
| P0 | Detection and instance-segmentation templates show training/prediction syntax without a complete evaluator and test submission. Seq2seq, captioning, and NER also stop before a complete test submission. Retrieval writes a task-assumed long-format CSV. | A template filename overstates how much work remains. | For each selected practice task, complete validation, full test inference, exact output serialization, official evaluation, and cold reload before calling it ready. |
| P1 | The final partial accumulation group in `Trainer._train_epoch` skips gradient clipping and scheduler advancement; loss is divided by the full accumulation factor even for a short group. | Update scale and scheduling depend on loader length or stopping time. | Full and partial groups use a defined averaging policy, the same clipping/update path, and one scheduler step per optimizer update. Verify partial groups and interruption behavior. |
| P1 | There is no dedicated tabular classification/regression adapter or notebook. | A common ML practice family has no fast prepared path. ML is broader than tabular prediction; this is a coverage hedge, not a prediction of contest tasks. | Add one compact sklearn pipeline workflow with train-only preprocessing, official metric, persistence, inference, and validated submission. Keep the shared core unchanged. |
| P1 | Several generic notebooks omit durable model/config/report output. The TF-IDF notebook does not serialize its fitted pipeline. `sync_artifacts` defaults omit `.joblib`, `.pkl`, `.txt`, `.zip`, and `.ipynb`. | A CSV may survive while the exact generating pipeline, tokenizer files, or deliverable archive does not. | Inventory and persist the entire selected solution bundle. Reload in a fresh process/runtime and reproduce predictions without refitting. Do not assume suffix-based sync saves every required artifact. |
| P1 | CV classification defaults to identity+hflip at test time, while validation uses plain inference. Generic split calls do not automatically select task grouping. | The submitted prediction path is not the one that was evaluated, and a random split can leak task identities. | Validate the identical preprocessing/TTA/decoder path used for submission. Set group/time fields from task evidence and assert overlap is zero. |
| P1 | The bootstrap checks whether any package is missing, then installs the entire requirements file. No live install was tested. | The missing-only intent is not implemented at package granularity; transitive changes to the runtime stack have not been ruled out. | Install only needed missing dependencies and verify torch/torchvision/NumPy/CUDA versions before and after a clean Colab bootstrap. |
| P2 | `write_submission` advertises an atomic write but calls `to_csv` directly on the destination. Some task-specific notebooks also save weights directly. | Interrupted writes can leave a corrupt final artifact. | Write temporary files in the destination directory, validate/read back, then replace; retain a previous known-good bundle. |
| P2 | Local Python is 3.14.8, whereas the preparation target expects Colab Python 3.12. `uv` exists but is absent from this shell's PATH. `.python-version` was already untracked. | Local success is not cross-runtime compatibility evidence, and the README commands need a shell setup adjustment here. | Use the existing uv executable or correct PATH, preserve user runtime choices, and perform clean target-runtime checks. |

The generic pretrained defaults above conflict with `AGENTS.md`. The task-specific anomaly
notebooks explicitly claim their old task permitted pretrained weights; that permission must not
be transferred to an unrelated task. A local checkpoint can also be the team's own trained model,
so checkpoint provenance must be recorded rather than inferred from its path.

## Focused CPU evidence

These were audit probes, not newly added repository tests.

1. **ID alignment:** sample IDs `[A, B]`, test IDs `[B, A]`, predictions `[1, 0]` in test order.
   Copying those predictions into the sample passes `validate_submission`, but both predictions
   are attached to the wrong IDs. This demonstrates a mapping blind spot, not an incorrect schema check.
2. **Metric reduction:** two 100-pixel images, one perfectly segmented foreground image and one
   missed single-pixel foreground. The registry reports global Dice **0.995025**; mean-per-image
   Dice is **0.500000**. Either convention can be correct for a particular task. Substituting one
   for the organizer's other convention is a metric mismatch.
3. **Accumulation tail:** one microbatch, accumulation factor four, zero-initialized scalar linear
   model, MSE, SGD learning rate 0.1, and clipping limit 0.01. The current tail produces an update
   of **0.05** and **zero** scheduler steps. A mean-gradient update with the configured clipping
   would be approximately **0.001** and advance the scheduler once.

## Verification record

Commands below used the existing environment with `--no-sync`; dependencies and the lockfile
were not rewritten. In this shell, replace `uv` with
`& 'C:/Users/DELL/.local/bin/uv.exe'`. `--cache-dir .uv-cache` was used throughout.

| Command | Result |
|---|---|
| `uv --cache-dir .uv-cache run --no-sync olp-ai doctor` | Python 3.14.8; torch 2.13.0+cpu; CUDA unavailable; torchvision 0.28.0; transformers 5.15.1. |
| `uv --cache-dir .uv-cache run --no-sync pytest` | 20 passed, 1 failed, 10 errors. The 10 errors came from sandbox temporary-directory permissions. |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/readiness-20261014-audit --tb=short` | **30 passed, 1 failed** in 7.67 seconds. Remaining failure: generated DINO notebook Python-cell compilation. |
| `uv --cache-dir .uv-cache run --no-sync ruff check .` | Passed. |
| `uv --cache-dir .uv-cache run --no-sync ruff format --check .` | Failed: one existing Python code fence in `docs/TASK2_ANOMALY_DETECTION.md` would be reformatted; 69 files already formatted. |
| `uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests scripts notebooks` | Passed for Python sources; this does not compile `.ipynb` cells. |
| `uv --cache-dir .uv-cache lock --check` | Passed; 254 packages resolved. This is lock consistency, not a fresh environment installation test. |
| `git diff --check` | Passed for tracked changes. Newly created preparation documents were also checked for trailing whitespace and Markdown fence balance. |
| Local Markdown link check | Three documents checked; no broken local file links. |
| Canonical export comparison in an ignored workspace directory | 15 templates examined; two generated notebook source mismatches confirmed. |
| `uv --cache-dir .uv-cache run --no-sync python -` with the three probes described above | Completed successfully after directing probe output into the workspace. |

Existing generated notebooks were not overwritten by `olp-ai build-notebooks`: the audit found
useful-looking noncanonical edits, and preserving them is part of the reconciliation work. No live
Colab GPU session, full official-data training, hidden-test result, or organizer-cloud access was
verified. The actual notebook tests compile cells; they do not run full training workflows.

## Suggested repair order

1. Reconcile notebook drift and weight-policy defaults.
2. Fix ID alignment and bind each selected task to its exact evaluator.
3. Correct accumulation tails before using accumulation in practice.
4. Make the text and image classification bundles survive a cold reload.
5. Add the small tabular workflow.
6. Complete only the next selected task family; defer unused adapters.
7. Run the full required checks after implementation changes, then rehearse on a clean GPU runtime.

An unseen task, a teammate executing the bundle, and a restart drill are stronger readiness
evidence than increasing the number of model options.
