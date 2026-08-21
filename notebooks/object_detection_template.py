# %% [markdown]
# # Object detection baseline (Faster R-CNN)
# Use this when the output is a class plus bounding box per object. Label 0 is background.

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
from olp_ai_26.cv.detection import (
    BoxDetectionDataset,
    build_detection_model,
    detection_collate,
    train_detection_epoch,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration
# Supported torchvision architectures are listed here. For YOLO-format datasets, see
# `train_yolo_baseline` and point it at an explicit local model YAML/weight file.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/detection")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train.csv"
IMAGE_COLUMN, LABEL_COLUMN = "image", "label_index"
BOX_COLUMNS = ("xmin", "ymin", "xmax", "ymax")
ARCHITECTURE = "fasterrcnn_resnet50_fpn_v2"
# alternatives: fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
NUM_CLASSES = 2  # background + foreground classes
BATCH_SIZE, EPOCHS, LEARNING_RATE = 4, 10, 5e-4
PRETRAINED_ALLOWED = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(gpu_report())

# %% [markdown]
# ## 1. Image-level split and variable-size loader
# If several rows describe one image, split on image identity to prevent the same image appearing
# in both folds. Replace this with grouped splitting when subject/video identity is available.

# %%
annotations = pd.read_csv(TRAIN_CSV)
images = annotations[[IMAGE_COLUMN]].drop_duplicates()
split = make_split(images, valid_size=0.2, seed=42)
train_rows = annotations[annotations[IMAGE_COLUMN].isin(split.train[IMAGE_COLUMN])]
valid_rows = annotations[annotations[IMAGE_COLUMN].isin(split.valid[IMAGE_COLUMN])]
train_ds = BoxDetectionDataset(
    train_rows,
    image_column=IMAGE_COLUMN,
    label_column=LABEL_COLUMN,
    box_columns=BOX_COLUMNS,
    root=DATA_DIR,
)
valid_ds = BoxDetectionDataset(
    valid_rows,
    image_column=IMAGE_COLUMN,
    label_column=LABEL_COLUMN,
    box_columns=BOX_COLUMNS,
    root=DATA_DIR,
)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=2,
    pin_memory=DEVICE == "cuda",
)

# %% [markdown]
# ## 2. Model and training
# Detection evaluation is task-specific: wire the official mAP/IoU implementation before tuning.

# %%
model = build_detection_model(
    NUM_CLASSES, architecture=ARCHITECTURE, pretrained_allowed=PRETRAINED_ALLOWED
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
for epoch in range(EPOCHS):
    loss = train_detection_epoch(model, train_loader, optimizer, device=DEVICE)
    print({"epoch": epoch + 1, "train_loss": loss})
checkpoint = paths.output_dir / "detector.pt"
torch.save(model.state_dict(), checkpoint)
sync_artifacts(paths.output_dir, paths.persistent_dir)

# %% [markdown]
# ## 3. Prediction syntax
# `predictions[i]` contains `boxes`, `labels`, and `scores`. Filter by `SCORE_THRESHOLD`, then
# serialize exactly as the sample submission specifies.

# %%
SCORE_THRESHOLD = 0.25
model.eval().to(DEVICE)
images, _ = next(iter(DataLoader(valid_ds, batch_size=2, collate_fn=detection_collate)))
with torch.inference_mode():
    predictions = model([image.to(DEVICE) for image in images])
filtered = [
    {key: value[prediction["scores"] >= SCORE_THRESHOLD].cpu() for key, value in prediction.items()}
    for prediction in predictions
]
filtered[0]
