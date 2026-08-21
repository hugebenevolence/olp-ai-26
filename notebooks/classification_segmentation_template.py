# %% [markdown]
# # Strong classification with an auxiliary segmentation head
# Use this when the leaderboard target is an image class but training masks are available.
# The mask head regularizes spatial features; final test prediction uses only `class_logits`.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
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
from olp_ai_26.core.split import make_split
from olp_ai_26.core.trainer import resolve_amp
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
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train.csv"
IMAGE_COLUMN, MASK_COLUMN, TARGET_COLUMN = "image", "mask", "label"
BACKBONE = "convnext_tiny"
IMAGE_SIZE, BATCH_SIZE, EPOCHS = 256, 16, 10
CLASS_LOSS_WEIGHT, MASK_LOSS_WEIGHT = 1.0, 0.4
PRETRAINED_ALLOWED = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(gpu_report())

# %% [markdown]
# ## 1. Wrap paired image/mask data with classification targets

# %%
frame = pd.read_csv(TRAIN_CSV)
label_to_index, index_to_label = label_mapping(frame[TARGET_COLUMN])
split = make_split(frame, target_columns=TARGET_COLUMN, valid_size=0.2, seed=42)


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
        return image, label, mask


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
    BACKBONE, len(label_to_index), pretrained_allowed=PRETRAINED_ALLOWED
).to(DEVICE)
criterion = ClassificationSegmentationLoss(
    class_weight=CLASS_LOSS_WEIGHT, mask_weight=MASK_LOSS_WEIGHT
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
amp_enabled, amp_dtype, scaler_enabled = resolve_amp(DEVICE, "auto")
scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for images, labels, masks in train_loader:
        images, labels, masks = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(images)
            loss, parts = criterion(outputs, labels, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels, _ in valid_loader:
            predicted = model(images.to(DEVICE))["class_logits"].argmax(1).cpu()
            correct += int((predicted == labels).sum())
            total += len(labels)
    print(
        {
            "epoch": epoch + 1,
            "loss": running / len(train_loader),
            "valid_accuracy": correct / total,
            **parts,
        }
    )
torch.save(model.state_dict(), paths.output_dir / "classification_segmentation.pt")
sync_artifacts(paths.output_dir, paths.persistent_dir)

# %% [markdown]
# ## 3. Classification-only inference syntax

# %%
model.eval()
images, _, _ = next(iter(valid_loader))
with torch.inference_mode():
    outputs = model(images.to(DEVICE))
predicted_labels = [
    index_to_label[index] for index in outputs["class_logits"].argmax(1).cpu().tolist()
]
predicted_labels[:5]
