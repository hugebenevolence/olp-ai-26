# %% [markdown]
# # Semantic segmentation baseline (SMP + TTA + RLE)
# Use this when every pixel receives a class and touching objects need not be separated.
# Change the configuration cell first; architecture and encoder discovery syntax is included.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ("segmentation_models_pytorch",)
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
from olp_ai_26.core.config import CompetitionConfig, TimeBudget, TrainerConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import Trainer
from olp_ai_26.cv.segmentation import (
    SEGMENTATION_ARCHITECTURES,
    DiceBCELoss,
    SegmentationTableDataset,
    build_segmentation_model,
    load_binary_masks,
    mask_to_rle,
    restore_binary_masks,
    score_mask_pairs,
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
paths.output_dir = paths.output_dir / "semantic_segmentation"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
GROUP_COLUMN, TIME_COLUMN = None, None
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
FOREGROUND_VALUES = None  # e.g. (1,) for Oxford trimaps
IGNORE_INDEX = None  # e.g. 2 for an ignored trimap boundary
METRIC_OPTIONS = {"empty_score": 1.0, "ignore_index": IGNORE_INDEX}
TTA_NAMES = ("identity",)
MODEL_CHECKPOINT = None
METRIC = "dice_mean_image"
competition = CompetitionConfig(data_dir=DATA_DIR, output_dir=paths.output_dir)
training = TrainerConfig(epochs=12, learning_rate=3e-4, metric_name=METRIC)
competition.prepare()
budget = TimeBudget(competition.time_budget_minutes)
print(gpu_report())
print("Architectures:", SEGMENTATION_ARCHITECTURES)
print("Encoder examples:", smp.encoders.get_encoder_names()[:40])

# %% [markdown]
# ## 1. Split, paired transforms, and loaders

# %%
train = pd.read_csv(TRAIN_CSV, converters={ID_COLUMN: str})
test = pd.read_csv(TEST_CSV, converters={ID_COLUMN: str})
sample = pd.read_csv(SAMPLE_CSV, converters={ID_COLUMN: str})
split = make_split(
    train, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, valid_size=0.2, seed=competition.seed
)
train_ds = SegmentationTableDataset(
    split.train,
    image_column=IMAGE_COLUMN,
    mask_column=MASK_COLUMN,
    foreground_values=FOREGROUND_VALUES,
    ignore_index=IGNORE_INDEX,
    root=DATA_DIR,
    size=IMAGE_SIZE,
    training=True,
)
valid_ds = SegmentationTableDataset(
    split.valid,
    image_column=IMAGE_COLUMN,
    mask_column=MASK_COLUMN,
    foreground_values=FOREGROUND_VALUES,
    ignore_index=IGNORE_INDEX,
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
    checkpoint_path=MODEL_CHECKPOINT,
)


class OriginalResolutionTrainer(Trainer):
    """Select checkpoints using the same original-size masks and TTA as final submission."""

    def evaluate(self, loader):
        metrics = super().evaluate(loader)
        logits = predict_segmentation_tta(
            self.model, loader, device=str(self.device), names=TTA_NAMES
        )
        masks = restore_binary_masks(
            logits, split.valid, image_column=IMAGE_COLUMN, root=DATA_DIR, threshold=MASK_THRESHOLD
        )
        references = load_binary_masks(
            split.valid,
            mask_column=MASK_COLUMN,
            root=DATA_DIR,
            foreground_values=FOREGROUND_VALUES,
            ignore_index=IGNORE_INDEX,
        )
        metrics["metric"] = score_mask_pairs(METRIC, references, masks, **METRIC_OPTIONS)
        return metrics


trainer = OriginalResolutionTrainer(
    model,
    DiceBCELoss(dice_weight=0.5, ignore_index=IGNORE_INDEX),
    training,
    output_dir=competition.output_dir,
    device=competition.device,
    prediction_decoder=lambda logits: (logits.sigmoid() >= MASK_THRESHOLD).long(),
    metric_fn=lambda target, pred: evaluate_metric(METRIC, target, pred, **METRIC_OPTIONS),
    backup_dir=paths.persistent_dir,
    time_budget=budget,
)
history = trainer.fit(train_loader, valid_loader)
print(history[-1])
score = trainer.evaluate(valid_loader)["metric"]

# %% [markdown]
# ## 3. TTA inference and RLE submission
# Verify the official RLE convention. `mask_to_rle` uses one-indexed, column-major ordering.

# %%
test_ds = SegmentationTableDataset(test, image_column=IMAGE_COLUMN, root=DATA_DIR, size=IMAGE_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **options)
logits = predict_segmentation_tta(model, test_loader, device=competition.device, names=TTA_NAMES)
masks = restore_binary_masks(
    logits, test, image_column=IMAGE_COLUMN, root=DATA_DIR, threshold=MASK_THRESHOLD
)
submission = build_submission(
    sample,
    {TARGET_COLUMN: [mask_to_rle(mask) for mask in masks]},
    prediction_ids=test[[ID_COLUMN]],
    id_columns=ID_COLUMN,
)
write_submission(
    submission, competition.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
record_run(
    competition.output_dir,
    task="semantic_segmentation",
    model=ARCHITECTURE,
    configuration={"competition": competition.to_dict(), "training": training},
    validation_metrics={training.metric_name: score},
    split=split,
    data_summary={"train_rows": len(train), "test_rows": len(test)},
    inference_options={
        "image_size": IMAGE_SIZE,
        "id_column": ID_COLUMN,
        "tta_names": TTA_NAMES,
        "mask_threshold": MASK_THRESHOLD,
        "foreground_values": FOREGROUND_VALUES,
        "ignore_index": IGNORE_INDEX,
        "metric_options": METRIC_OPTIONS,
        "output_resolution": "original image",
    },
    checkpoint=str(trainer.checkpoint_path),
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
