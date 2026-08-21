# %% [markdown]
# # Instance segmentation baseline (Mask R-CNN)
# Use this when touching objects must be returned as separate masks. Label 0 is background.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
import torch
from torch.utils.data import DataLoader

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.split import make_split
from olp_ai_26.cv.detection import detection_collate, train_detection_epoch
from olp_ai_26.cv.instance_segmentation import (
    InstanceMaskDataset,
    build_instance_segmentation_model,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/instance_segmentation")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train_instances.csv"  # one row per object instance
IMAGE_COLUMN, MASK_COLUMN, LABEL_COLUMN = "image", "mask", "label_index"
ARCHITECTURE = "maskrcnn_resnet50_fpn_v2"  # or maskrcnn_resnet50_fpn
NUM_CLASSES = 2  # background + foreground classes
BATCH_SIZE, EPOCHS, LEARNING_RATE = 2, 10, 3e-4
PRETRAINED_ALLOWED = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_THRESHOLD, MASK_THRESHOLD = 0.25, 0.5
print(gpu_report())

# %% [markdown]
# ## 1. Dataset, model, and training
# Each mask file is a binary image for exactly one instance. Rows sharing an image are grouped.

# %%
annotations = pd.read_csv(TRAIN_CSV)
images = annotations[[IMAGE_COLUMN]].drop_duplicates()
split = make_split(images, valid_size=0.2, seed=42)
train_rows = annotations[annotations[IMAGE_COLUMN].isin(split.train[IMAGE_COLUMN])]
valid_rows = annotations[annotations[IMAGE_COLUMN].isin(split.valid[IMAGE_COLUMN])]
train_dataset = InstanceMaskDataset(
    train_rows,
    image_column=IMAGE_COLUMN,
    mask_column=MASK_COLUMN,
    label_column=LABEL_COLUMN,
    root=DATA_DIR,
)
valid_dataset = InstanceMaskDataset(
    valid_rows,
    image_column=IMAGE_COLUMN,
    mask_column=MASK_COLUMN,
    label_column=LABEL_COLUMN,
    root=DATA_DIR,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=2,
    pin_memory=DEVICE == "cuda",
)
model = build_instance_segmentation_model(
    NUM_CLASSES, architecture=ARCHITECTURE, pretrained_allowed=PRETRAINED_ALLOWED
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
for epoch in range(EPOCHS):
    print(
        {
            "epoch": epoch + 1,
            "loss": train_detection_epoch(model, train_loader, optimizer, device=DEVICE),
        }
    )
torch.save(model.state_dict(), paths.output_dir / "mask_rcnn.pt")
sync_artifacts(paths.output_dir, paths.persistent_dir)

# %% [markdown]
# ## 2. Prediction syntax
# A result contains per-instance `boxes`, `labels`, `scores`, and soft `masks` shaped N x 1 x H x W.

# %%
model.eval().to(DEVICE)
valid_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=detection_collate)
images, _ = next(iter(valid_loader))
with torch.inference_mode():
    prediction = model([images[0].to(DEVICE)])[0]
keep = prediction["scores"] >= SCORE_THRESHOLD
# Torchvision inference already returns mask probabilities, so do not apply sigmoid again.
instance_masks = (prediction["masks"][keep, 0] >= MASK_THRESHOLD).cpu()
print(prediction["boxes"][keep].shape, instance_masks.shape)
