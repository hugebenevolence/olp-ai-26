# %% [markdown]
# # Video / sign-language / activity-recognition baseline
# Start with sampled frames and a shared 2D CNN. Move to a 3D model only after this submits.

# %% [markdown]
# ## Colab bootstrap

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ("timm",)
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
from torch import nn
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
from olp_ai_26.core.inference import predict_logits
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import Trainer
from olp_ai_26.cv.classification import build_image_transforms, label_mapping
from olp_ai_26.cv.video_classification import TemporalPoolingClassifier, VideoTableDataset

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/video")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "video_classification"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
GROUP_COLUMN, TIME_COLUMN = None, None
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
VIDEO_COLUMN = "video"
TARGET_COLUMN = "label"
ID_COLUMN = "id"
BACKBONE = "resnet18"
NUM_FRAMES = 12
IMAGE_SIZE = 160
BATCH_SIZE = 4
MODEL_CHECKPOINT = None

competition = CompetitionConfig(
    data_dir=DATA_DIR,
    output_dir=paths.output_dir,
    pretrained_allowed=False,
    num_workers=2,
)
training = TrainerConfig(
    epochs=8,
    learning_rate=3e-4,
    optimizer="adamw",
    scheduler="onecycle",
)
competition.prepare()
budget = TimeBudget(competition.time_budget_minutes)
print(gpu_report())

# %% [markdown]
# ## 1. Data, group-aware split, and decoding sanity check

# %%
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
split = make_split(
    train,
    target_columns=TARGET_COLUMN,
    # Set group_column="person_id" when multiple clips belong to one person.
    group_column=GROUP_COLUMN,
    time_column=TIME_COLUMN,
    seed=competition.seed,
)
label_to_index, index_to_label = label_mapping(train[TARGET_COLUMN])
transform = build_image_transforms(size=IMAGE_SIZE, training=False)
train_ds = VideoTableDataset(
    split.train,
    video_column=VIDEO_COLUMN,
    target_column=TARGET_COLUMN,
    root=DATA_DIR,
    num_frames=NUM_FRAMES,
    transform=transform,
    label_to_index=label_to_index,
)
valid_ds = VideoTableDataset(
    split.valid,
    video_column=VIDEO_COLUMN,
    target_column=TARGET_COLUMN,
    root=DATA_DIR,
    num_frames=NUM_FRAMES,
    transform=transform,
    label_to_index=label_to_index,
)
loader_options = dataloader_kwargs(competition.device, competition.num_workers)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_options)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **loader_options)

# %% [markdown]
# ## 2. Baseline training

# %%
model = TemporalPoolingClassifier(
    BACKBONE,
    len(label_to_index),
    pretrained_allowed=competition.pretrained_allowed,
    checkpoint_path=MODEL_CHECKPOINT,
)
trainer = Trainer(
    model,
    nn.CrossEntropyLoss(),
    training,
    output_dir=competition.output_dir,
    device=competition.device,
    metric_fn=lambda y, p: evaluate_metric("macro_f1", y, p),
    time_budget=budget,
    backup_dir=paths.persistent_dir,
)
history = trainer.fit(train_loader, valid_loader)
print(history[-1])
score = trainer.evaluate(valid_loader)["metric"]

# %% [markdown]
# ## 3. Inference and submission

# %%
test_ds = VideoTableDataset(
    test,
    video_column=VIDEO_COLUMN,
    root=DATA_DIR,
    num_frames=NUM_FRAMES,
    transform=transform,
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **loader_options)
logits = predict_logits(model, test_loader, device=competition.device)
predictions = [index_to_label[index] for index in logits.argmax(axis=1)]
submission = build_submission(
    sample, {TARGET_COLUMN: predictions}, prediction_ids=test[[ID_COLUMN]], id_columns=ID_COLUMN
)
write_submission(
    submission,
    competition.output_dir / "submission.csv",
    sample=sample,
    id_columns=ID_COLUMN,
)
record_run(
    competition.output_dir,
    task="video_classification",
    model=BACKBONE,
    configuration={"competition": competition.to_dict(), "training": training},
    validation_metrics={training.metric_name: score},
    split=split,
    data_summary={"train_rows": len(train), "test_rows": len(test)},
    inference_options={
        "label_values": [index_to_label[i] for i in range(len(index_to_label))],
        "image_size": IMAGE_SIZE,
        "id_column": ID_COLUMN,
        "num_frames": NUM_FRAMES,
    },
    checkpoint=str(trainer.checkpoint_path),
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
