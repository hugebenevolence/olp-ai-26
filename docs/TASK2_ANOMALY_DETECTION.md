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

The extracted `public_test/README.txt` also asks for a finite real-valued anomaly score, while the
official PDF explicitly requires binary values in a `label` column. The notebook follows the PDF's
binary CSV contract. Confirm any later organizer announcement before changing this behavior.

## Actual released archive layout

The official `ThiChinhThucData.zip` is a nested archive:

```text
ThiChinhThucData.zip
└── CV_Data
    ├── training_dataset/dataset_train.zip
    ├── public_test/public_test.zip
    └── private_test/private_test.zip  # encrypted until final release
```

After inner extraction, the usable roots are:

```text
dataset_train/
├── train1_6.csv
├── train2_5.csv
├── train3_4.csv
└── train/category_01/...category_06/...
public_test/
├── test.csv
└── images/category_01/...category_06/...
```

The notebook's `stage_official_task2_data()` helper extracts only the CV train archive and selected
test phase into fast `/content` storage. It does not unpack the unrelated NLP archive. The private
ZIP requires `PRIVATE_ZIP_PASSWORD` after the organizer releases it.
The helper uses `pyzipper` because the supplied private archive uses AES ZIP compression, which
Python's standard `zipfile` module cannot decrypt.

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

The supplied 0.577 run and the reasons for each revision are recorded in
[`TASK2_0577_AUDIT.md`](TASK2_0577_AUDIT.md). Read that evidence table before changing multiple
experimental axes at once.

## First public run

Open `notebooks/image_anomaly_detection_template.ipynb` and change only the configuration cell:

```python
TEAM_NAME = "your_team"
PHASE = "public"
RUN_TRAINING = True
EXPERIMENT_PRESET = "category_augmented"
OFFICIAL_DATA_SOURCE = Path("/content/drive/MyDrive/olpai26/ThiChinhThucData.zip")
PERSISTENT_DIR = Path("/content/drive/MyDrive/olpai26/task2_artifacts")
```

Run every cell. Verify the printed train counts match `664, 664, 302, 660, 660, 210`; inspect the
predicted anomaly count per category; then submit the generated ZIP. Do not assume a high CutPaste
proxy score implies a high PublicScore.

## Experiment presets

The configuration cell exposes three drag-and-plug presets:

```python
EXPERIMENT_PRESET = "baseline_0577"
EXPERIMENT_PRESET = "category_models_only"
EXPERIMENT_PRESET = "category_augmented"
```

`baseline_0577` reproduces the submitted design. `category_models_only` isolates per-category
model/resolution/memory choices. `category_augmented` adds limited normal-memory invariance and
separate synthetic anomaly families (`cutpaste`, `mixup`, `blur`, and `dark_curve`). Never add a
suspected defect transform to `normal_augmentations`.

Every category configuration stores `model_name`, `image_size`, `batch_size`,
`max_memory_patches`, `top_k`, `normal_augmentations`, `synthetic_anomalies`, `normal_quantile`,
`threshold_mode`, and `threshold_scale`. Edit one field or one preset at a time.

## Speed and memory controls

If Colab runs out of memory, reduce the affected category's `batch_size` first. If nearest-neighbor
scoring is too slow, reduce `max_memory_patches`; this has the largest direct effect. Reducing
`image_size` can hide small defects and should be treated as a measured model change.

## Public threshold experiments

The notebook emits candidates at threshold scales 0.90, 0.95, 1.00, and 1.05 without retraining
(1.00 is the main candidate).
Submit deliberate experiments only; the public limit is 20. Lower scale predicts more anomalies.
After selecting the scale from aggregate PublicScore, set that category configuration's
`threshold_scale` and rerun training so the exact choice is stored in the experiment bundle.

## Private final

Before private data is released, ensure the frozen bundle is in durable storage. Then change:

```python
PHASE = "private"
RUN_TRAINING = False
EXPERIMENT_PRESET = "category_augmented"  # must match the selected public run
```

The notebook rebuilds the architecture without downloading weights, restores the exact frozen
encoders, memories, thresholds, and scales from
`PERSISTENT_DIR / EXPERIMENT_PRESET / task2_<preset>_bundle.pt`, then performs inference only. It
refuses private-mode training. Validate that the final ZIP contains exactly
`task2_private_output.csv`.
