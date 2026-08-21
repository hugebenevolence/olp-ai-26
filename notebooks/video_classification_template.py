# %% [markdown]
# # Video / sign-language / activity-recognition baseline
# Start with sampled frames and a shared 2D CNN. Move to a 3D model only after this submits.

# %%
from pathlib import Path

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from olp_ai_26.core.config import CompetitionConfig, TimeBudget, TrainerConfig
from olp_ai_26.core.inference import predict_logits
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import Trainer
from olp_ai_26.cv.classification import build_image_transforms, label_mapping
from olp_ai_26.cv.video_classification import TemporalPoolingClassifier, VideoTableDataset

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DATA_DIR = Path("data")
VIDEO_COLUMN = "video"
TARGET_COLUMN = "label"
ID_COLUMN = "id"
BACKBONE = "resnet18"
NUM_FRAMES = 12
IMAGE_SIZE = 160
BATCH_SIZE = 4

competition = CompetitionConfig(data_dir=DATA_DIR, pretrained_allowed=False, num_workers=2)
training = TrainerConfig(
    epochs=8,
    learning_rate=3e-4,
    optimizer="adamw",
    scheduler="onecycle",
)
competition.prepare()
budget = TimeBudget(competition.time_budget_minutes)

# %% [markdown]
# ## 1. Data, group-aware split, and decoding sanity check

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
split = make_split(
    train,
    target_columns=TARGET_COLUMN,
    # Set group_column="person_id" when multiple clips belong to one person.
    group_column=None,
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
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

# %% [markdown]
# ## 2. Baseline training

# %%
model = TemporalPoolingClassifier(
    BACKBONE,
    len(label_to_index),
    pretrained_allowed=competition.pretrained_allowed,
)
trainer = Trainer(
    model,
    nn.CrossEntropyLoss(),
    training,
    output_dir=competition.output_dir,
    device=competition.device,
    metric_fn=lambda y, p: evaluate_metric("macro_f1", y, p),
    time_budget=budget,
)
history = trainer.fit(train_loader, valid_loader)
print(history[-1])

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
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
logits = predict_logits(model, test_loader, device=competition.device)
predictions = [index_to_label[index] for index in logits.argmax(axis=1)]
submission = build_submission(sample, {TARGET_COLUMN: predictions})
write_submission(
    submission,
    competition.output_dir / "submission.csv",
    sample=sample,
    id_columns=ID_COLUMN,
)
