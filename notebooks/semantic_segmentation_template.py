# %% [markdown]
# # Semantic segmentation baseline (SMP + TTA + RLE)
# Use this when every pixel receives a class and touching objects need not be separated.
# Change the configuration cell first; architecture and encoder discovery syntax is included.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from olp_ai_26.core.colab import (
    ColabPaths,
    dataloader_kwargs,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.config import CompetitionConfig, TrainerConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import Trainer
from olp_ai_26.cv.segmentation import (
    SEGMENTATION_ARCHITECTURES,
    DiceBCELoss,
    SegmentationTableDataset,
    build_segmentation_model,
    mask_to_rle,
)
from olp_ai_26.cv.tta import predict_segmentation_tta

# %% [markdown]
# ## 0. Drag-and-plug configuration
# `ARCHITECTURE` accepts every value printed below. `ENCODER_NAME` accepts every value from
# `smp.encoders.get_encoder_names()`. Start small, verify the metric/submission, then scale.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/semantic_segmentation")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"
IMAGE_COLUMN, MASK_COLUMN, ID_COLUMN, TARGET_COLUMN = "image", "mask", "id", "rle"
ARCHITECTURE = "unet"  # unet, unetplusplus, fpn, pspnet, deeplabv3(+), linknet, manet, pan
ENCODER_NAME = "resnet18"  # try resnet50, efficientnet-b0, timm-convnext_tiny
IMAGE_SIZE, BATCH_SIZE = 256, 16
MASK_THRESHOLD = 0.5
TTA_NAMES = ("identity", "hflip")
competition = CompetitionConfig(data_dir=DATA_DIR, output_dir=paths.output_dir)
training = TrainerConfig(epochs=12, learning_rate=3e-4, metric_name="dice")
competition.prepare()
print(gpu_report())
print("Architectures:", SEGMENTATION_ARCHITECTURES)
print("Encoder examples:", smp.encoders.get_encoder_names()[:40])

# %% [markdown]
# ## 1. Split, paired transforms, and loaders

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample = pd.read_csv(SAMPLE_CSV)
split = make_split(train, valid_size=0.2, seed=competition.seed)
train_ds = SegmentationTableDataset(
    split.train,
    image_column=IMAGE_COLUMN,
    mask_column=MASK_COLUMN,
    root=DATA_DIR,
    size=IMAGE_SIZE,
    training=True,
)
valid_ds = SegmentationTableDataset(
    split.valid,
    image_column=IMAGE_COLUMN,
    mask_column=MASK_COLUMN,
    root=DATA_DIR,
    size=IMAGE_SIZE,
    training=False,
)
options = dataloader_kwargs(competition.device, competition.num_workers)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **options)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **options)

# %% [markdown]
# ## 2. Model and training

# %%
model = build_segmentation_model(
    architecture=ARCHITECTURE,
    encoder_name=ENCODER_NAME,
    classes=1,
    pretrained_allowed=competition.pretrained_allowed,
)
trainer = Trainer(
    model,
    DiceBCELoss(dice_weight=0.5),
    training,
    output_dir=competition.output_dir,
    device=competition.device,
    prediction_decoder=lambda logits: (logits.sigmoid() >= MASK_THRESHOLD).long(),
    metric_fn=lambda target, pred: evaluate_metric("dice", target, pred),
    backup_dir=paths.persistent_dir,
)
history = trainer.fit(train_loader, valid_loader)
print(history[-1])

# %% [markdown]
# ## 3. TTA inference and RLE submission
# Verify the official RLE convention. `mask_to_rle` uses one-indexed, column-major ordering.

# %%
test_ds = SegmentationTableDataset(test, image_column=IMAGE_COLUMN, root=DATA_DIR, size=IMAGE_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **options)
logits = predict_segmentation_tta(model, test_loader, device=competition.device, names=TTA_NAMES)
masks = (logits.sigmoid()[:, 0].numpy() >= MASK_THRESHOLD).astype("uint8")
submission = build_submission(sample, {TARGET_COLUMN: [mask_to_rle(mask) for mask in masks]})
write_submission(
    submission, competition.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
