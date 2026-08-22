# Task 2 image anomaly-detection runbook

## Exact task contract

- Training contains 3,160 normal images across six independent categories.
- No labeled anomaly image or anomaly mask is supplied.
- Synthetic anomalies made only from official training images are allowed.
- Pretrained weights are allowed; external images, labels, and model outputs are forbidden.
- Output columns are exactly `sample_id,category,label`, where 0 is normal and 1 is anomaly.
- Official score is 100 times the mean of six per-category balanced accuracies.
- Tie-breaks are macro anomaly F1, then minimum per-category balanced accuracy.

The problem statement contains contradictory private-test wording: the boxed rule says model,
weights, thresholds, and calibration must be frozen and must not be refit from private test, while
the following numbered item appears to say threshold refitting is possible. The notebook enforces
the stricter boxed rule. Seek an organizer clarification rather than weakening this safeguard.

## Why this baseline

The supplied training data has no positive labels, so a conventional binary classifier cannot be
trained honestly. The notebook uses a compact PatchCore-style approach:

1. A permitted pretrained timm encoder extracts multiscale patches.
2. A deterministic random projection reduces patch dimension.
3. A bounded normal memory bank is fitted independently for each category.
4. An image score is the mean of its largest nearest-memory patch distances.
5. Held-out normal images and allowed CutPaste corruptions provide a proxy calibration set.
6. Six thresholds and all model/memory state are frozen in one artifact before private inference.

Synthetic proxy balanced accuracy is a debugging/calibration signal, not the hidden official
metric. Public aggregate feedback may tune an algorithmic threshold scale, but per-image manual
labeling is forbidden.

## First public run

Open `notebooks/image_anomaly_detection_template.ipynb` and change only the configuration cell:

```python
TEAM_NAME = "your_team"
PHASE = "public"
RUN_TRAINING = True
DRIVE_DATA_ARCHIVE = Path("/content/drive/MyDrive/olpai26/task2.zip")
PERSISTENT_DIR = Path("/content/drive/MyDrive/olpai26/task2_artifacts")
```

Run every cell. Verify the printed train counts match `664, 664, 302, 660, 660, 210`; inspect the
predicted anomaly count per category; then submit the generated ZIP. Do not assume a high CutPaste
proxy score implies a high PublicScore.

## Speed and memory controls

Change these in order if inference is too slow or GPU memory is insufficient:

```python
MODEL_NAME = "resnet18"  # fastest correctness baseline
IMAGE_SIZE = 224  # fewer patches
BATCH_SIZE = 8  # GPU activation memory
MAX_MEMORY_PATCHES = 2048  # distance-computation cost
PROJECTION_DIM = 64  # memory and distance cost
```

Reducing `MAX_MEMORY_PATCHES` has the largest effect on nearest-neighbor inference time. Reducing
image size may hide small defects, so validate that change rather than treating it as free speed.

## Public threshold experiments

The notebook can emit candidates at threshold scales 0.90, 1.00, and 1.10 without retraining.
Submit deliberate experiments only; the public limit is 20. Lower scale predicts more anomalies.
After selecting the scale from aggregate PublicScore, set `THRESHOLD_SCALE` and rerun the freeze
cell so the value is stored in `task2_anomaly_bundle.pt`.

## Private final

Before private data is released, ensure the frozen bundle is in durable storage. Then change:

```python
PHASE = "private"
RUN_TRAINING = False
BUNDLE_PATH = PERSISTENT_DIR / "task2_anomaly_bundle.pt"
```

The notebook rebuilds the architecture without downloading weights, restores the exact frozen
encoder, memories, thresholds, and scale, and performs inference only. It refuses private-mode
training. Validate that the final ZIP contains exactly `task2_private_output.csv`.
