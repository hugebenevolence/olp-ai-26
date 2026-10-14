# %% [markdown]
# # Instance segmentation baseline (Mask R-CNN)
# Use this when touching objects must be returned as separate masks. Label 0 is background.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
import torch
from torch.utils.data import DataLoader

from olp_ai_26.core.artifacts import save_torch_state
from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.config import CompetitionConfig, TimeBudget
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.cv.detection import (
    DetectionInferenceDataset,
    detection_collate,
    predict_detections,
    serialize_detections,
    train_detection_epoch,
)
from olp_ai_26.cv.evaluation import practice_average_precision
from olp_ai_26.cv.instance_segmentation import (
    InstanceMaskDataset,
    build_instance_segmentation_model,
)

# %% [markdown]
# ## 0. Explicit practice contract and evaluator
# Default CSV: one image ID plus a JSON prediction list. Adapt to the official sample.
# Practice AP50 is not COCO. Replace EVALUATOR with the exact official callable.

# %%
DRIVE_DATA_ARCHIVE = None
PERSISTENT_DIR = None
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "instance_segmentation"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
ID_COLUMN, IMAGE_COLUMN, LABEL_COLUMN = "id", "image", "label_index"
PREDICTION_COLUMN = "predictions"
GROUP_COLUMN, TIME_COLUMN = None, None
MASK_COLUMN = "mask"
BOX_COLUMNS = ("xmin", "ymin", "xmax", "ymax")
ARCHITECTURE = "maskrcnn_resnet50_fpn_v2"
NUM_CLASSES = 2  # includes background; map organizer foreground classes to indices >= 1
PRETRAINED_ALLOWED = False
MODEL_CHECKPOINT = None  # staged full-model state with matching architecture and classes
BATCH_SIZE, EPOCHS, LEARNING_RATE = 2, 5, 5e-4
SCORE_THRESHOLD, MASK_THRESHOLD = 0.0, 0.5
METRIC_NAME = "practice_mask_ap50"


def EVALUATOR(predictions, targets):
    """Replace this practice scorer with the exact official evaluation function."""
    return practice_average_precision(
        predictions, targets, masks=True, mask_threshold=MASK_THRESHOLD
    )


competition = CompetitionConfig(
    data_dir=DATA_DIR, output_dir=paths.output_dir, pretrained_allowed=PRETRAINED_ALLOWED
)
competition.prepare()
budget = TimeBudget(competition.time_budget_minutes)
print(gpu_report())

# %% [markdown]
# ## 1. Inspect and split on image identity, optionally subject or time

# %%
annotations = pd.read_csv(DATA_DIR / "train_instances.csv", converters={ID_COLUMN: str})
metadata_columns = [ID_COLUMN, IMAGE_COLUMN] + [
    column for column in (GROUP_COLUMN, TIME_COLUMN) if column
]
images = annotations[metadata_columns].drop_duplicates()
if images[IMAGE_COLUMN].duplicated().any() or images[ID_COLUMN].duplicated().any():
    raise ValueError("One consistent ID and group/time record is required per image")
print(annotations.head())
split = make_split(
    images, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, seed=competition.seed
)
train_rows = annotations[annotations[IMAGE_COLUMN].isin(split.train[IMAGE_COLUMN])]
valid_rows = annotations[annotations[IMAGE_COLUMN].isin(split.valid[IMAGE_COLUMN])]
train_ds = InstanceMaskDataset(
    train_rows,
    image_column=IMAGE_COLUMN,
    label_column=LABEL_COLUMN,
    mask_column=MASK_COLUMN,
    root=DATA_DIR,
)
valid_ds = InstanceMaskDataset(
    valid_rows,
    image_column=IMAGE_COLUMN,
    label_column=LABEL_COLUMN,
    mask_column=MASK_COLUMN,
    root=DATA_DIR,
)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=competition.num_workers,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=detection_collate,
    num_workers=competition.num_workers,
)
valid_targets = [target for _, target in valid_ds]

# %% [markdown]
# ## 2. Train, evaluate the actual prediction path, and atomically preserve the best state

# %%
model = build_instance_segmentation_model(
    NUM_CLASSES,
    architecture=ARCHITECTURE,
    pretrained_allowed=PRETRAINED_ALLOWED,
    checkpoint_path=MODEL_CHECKPOINT,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def filter_predictions(predictions):
    """Apply the identical confidence cutoff before validation and submission."""
    return [
        {key: value[prediction["scores"] >= SCORE_THRESHOLD] for key, value in prediction.items()}
        for prediction in predictions
    ]


best_score = float("-inf")
checkpoint = paths.output_dir / "best.pt"
saved_this_run = False
for epoch in range(EPOCHS):
    if budget.should_stop:
        break
    loss = train_detection_epoch(
        model, train_loader, optimizer, device=competition.device, time_budget=budget
    )
    if budget.should_stop:
        break
    validation_predictions = filter_predictions(
        predict_detections(model, valid_loader, device=competition.device)
    )
    score = float(EVALUATOR(validation_predictions, valid_targets))
    print({"epoch": epoch + 1, "loss": loss, METRIC_NAME: score})
    if score > best_score:
        best_score = score
        save_torch_state(model.state_dict(), checkpoint)
        saved_this_run = True
        sync_artifacts(paths.output_dir, paths.persistent_dir)
if not saved_this_run:
    raise RuntimeError("No evaluated checkpoint was produced; shorten the baseline")
model.load_state_dict(torch.load(checkpoint, map_location=competition.device, weights_only=True))

# %% [markdown]
# ## 3. Full test inference, exact ID alignment, and delivery bundle

# %%
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
test_ds = DetectionInferenceDataset(test, image_column=IMAGE_COLUMN, root=DATA_DIR)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=detection_collate,
    num_workers=competition.num_workers,
)
predictions = filter_predictions(predict_detections(model, test_loader, device=competition.device))
payloads = serialize_detections(
    predictions, score_threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD
)
submission = build_submission(
    sample, {PREDICTION_COLUMN: payloads}, prediction_ids=test[[ID_COLUMN]], id_columns=ID_COLUMN
)
write_submission(
    submission, paths.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
record_run(
    paths.output_dir,
    task="instance_segmentation",
    model=ARCHITECTURE,
    configuration={
        "competition": competition.to_dict(),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    },
    validation_metrics={METRIC_NAME: best_score},
    split=split,
    inference_options={
        "num_classes": NUM_CLASSES,
        "score_threshold": SCORE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "serialization": "one JSON list per image",
    },
    checkpoint=str(checkpoint),
    notes=["Practice AP is not a substitute for the organizer evaluator."],
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
