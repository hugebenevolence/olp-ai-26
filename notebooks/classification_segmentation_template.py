# %% [markdown]
# # Strong classification with an auxiliary segmentation head
# Use this when the leaderboard target is an image class but training masks are available.
# The mask head regularizes spatial features; final test prediction uses only `class_logits`.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ("timm",)
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from olp_ai_26.core.colab import (
    ColabPaths,
    dataloader_kwargs,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.config import CompetitionConfig, TimeBudget, TrainerConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import Trainer
from olp_ai_26.cv.classification import label_mapping
from olp_ai_26.cv.multitask import ClassificationSegmentationLoss, ClassificationSegmentationModel
from olp_ai_26.cv.segmentation import SegmentationTableDataset

# %% [markdown]
# ## 0. Drag-and-plug configuration
# Any timm backbone supporting `features_only=True` can work. Good first choices are
# `resnet50`, `efficientnet_b3`, and `convnext_tiny`. Reduce batch/image size after OOM.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/multitask")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "classification_segmentation"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train.csv"
IMAGE_COLUMN, MASK_COLUMN, TARGET_COLUMN, ID_COLUMN = "image", "mask", "label", "id"
GROUP_COLUMN, TIME_COLUMN = None, None
MODEL_CHECKPOINT = None
METRIC = "macro_f1"
BACKBONE = "convnext_tiny"
IMAGE_SIZE, BATCH_SIZE, EPOCHS = 256, 16, 10
CLASS_LOSS_WEIGHT, MASK_LOSS_WEIGHT = 1.0, 0.4
PRETRAINED_ALLOWED = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
competition = CompetitionConfig(
    data_dir=DATA_DIR, output_dir=paths.output_dir, pretrained_allowed=PRETRAINED_ALLOWED
)
competition.prepare()
training = TrainerConfig(epochs=EPOCHS, metric_name=METRIC)
budget = TimeBudget(competition.time_budget_minutes)
print(gpu_report())

# %% [markdown]
# ## 1. Wrap paired image/mask data with classification targets

# %%
frame = pd.read_csv(TRAIN_CSV, converters={ID_COLUMN: str})
label_to_index, index_to_label = label_mapping(frame[TARGET_COLUMN])
split = make_split(
    frame,
    target_columns=TARGET_COLUMN,
    group_column=GROUP_COLUMN,
    time_column=TIME_COLUMN,
    valid_size=0.2,
    seed=competition.seed,
)


class MultitaskDataset(Dataset):
    """Attach a class index to the paired image/mask dataset."""

    def __init__(self, split_frame, training):
        """Create paired transforms and retain labels in matching row order."""
        self.frame = split_frame.reset_index(drop=True)
        self.paired = SegmentationTableDataset(
            self.frame,
            image_column=IMAGE_COLUMN,
            mask_column=MASK_COLUMN,
            root=DATA_DIR,
            size=IMAGE_SIZE,
            training=training,
        )

    def __len__(self):
        """Return the number of labeled images."""
        return len(self.paired)

    def __getitem__(self, index):
        """Return image tensor, class index, and aligned binary mask."""
        image, mask = self.paired[index]
        label = torch.tensor(label_to_index[self.frame.iloc[index][TARGET_COLUMN]])
        return image, {"class": label, "mask": mask}


options = dataloader_kwargs(str(DEVICE), num_workers=2)
train_loader = DataLoader(
    MultitaskDataset(split.train, training=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **options,
)
valid_loader = DataLoader(
    MultitaskDataset(split.valid, training=False),
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    **options,
)

# %% [markdown]
# ## 2. Shared encoder, two heads, and explicit combined-loss loop

# %%
model = ClassificationSegmentationModel(
    BACKBONE,
    len(label_to_index),
    pretrained_allowed=PRETRAINED_ALLOWED,
    checkpoint_path=MODEL_CHECKPOINT,
).to(DEVICE)
criterion = ClassificationSegmentationLoss(
    class_weight=CLASS_LOSS_WEIGHT, mask_weight=MASK_LOSS_WEIGHT
)


def combined_loss(outputs, targets):
    """Adapt the two-head loss to the shared trainer's scalar-loss contract."""
    return criterion(outputs, targets["class"], targets["mask"])[0]


trainer = Trainer(
    model,
    combined_loss,
    training,
    output_dir=competition.output_dir,
    device=competition.device,
    prediction_decoder=lambda outputs: outputs["class_logits"].argmax(1),
    target_decoder=lambda targets: targets["class"],
    metric_fn=lambda y, p: evaluate_metric(METRIC, y, p),
    time_budget=budget,
    backup_dir=paths.persistent_dir,
)
history = trainer.fit(train_loader, valid_loader)
score = trainer.evaluate(valid_loader)["metric"]
print(history[-1])

# %% [markdown]
# ## 3. Complete classification-only test inference and validated submission

# %%
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
test_ds = SegmentationTableDataset(test, image_column=IMAGE_COLUMN, root=DATA_DIR, size=IMAGE_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **options)
model.eval()
predicted_labels = []
with torch.inference_mode():
    for images in test_loader:
        indexes = model(images.to(DEVICE))["class_logits"].argmax(1).cpu().tolist()
        predicted_labels.extend(index_to_label[index] for index in indexes)
submission = build_submission(
    sample,
    {TARGET_COLUMN: predicted_labels},
    prediction_ids=test[[ID_COLUMN]],
    id_columns=ID_COLUMN,
)
write_submission(
    submission,
    competition.output_dir / "submission.csv",
    sample=sample,
    id_columns=ID_COLUMN,
    allowed_labels={TARGET_COLUMN: label_to_index},
)
record_run(
    competition.output_dir,
    task="classification_segmentation",
    model=BACKBONE,
    configuration={
        "competition": competition.to_dict(),
        "training": training,
        "class_loss_weight": CLASS_LOSS_WEIGHT,
        "mask_loss_weight": MASK_LOSS_WEIGHT,
    },
    validation_metrics={METRIC: score},
    split=split,
    inference_options={
        "image_size": IMAGE_SIZE,
        "label_values": [index_to_label[i] for i in range(len(index_to_label))],
    },
    checkpoint=str(trainer.checkpoint_path),
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
