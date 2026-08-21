# %% [markdown]
# # CV classification baseline
# Change only the configuration cell first.
# Obtain one valid local score and submission before tuning.

# %%
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from olp_ai_26.core.config import CompetitionConfig, TimeBudget, TrainerConfig
from olp_ai_26.core.inference import predict_logits
from olp_ai_26.core.inspect import dataset_report
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import Trainer
from olp_ai_26.cv.classification import (
    ImageTableDataset,
    build_image_classifier,
    build_image_transforms,
    classification_loss,
    label_mapping,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"
IMAGE_COLUMN = "image"
TARGET_COLUMN = "label"
ID_COLUMN = "id"
MODEL_NAME = "resnet18"
IMAGE_SIZE = 224
BATCH_SIZE = 32

competition = CompetitionConfig(data_dir=DATA_DIR, pretrained_allowed=False)
training = TrainerConfig(epochs=8, learning_rate=3e-4, metric_name="macro_f1")
competition.prepare()
budget = TimeBudget(competition.time_budget_minutes)

# %% [markdown]
# ## 1. Inspect before modeling

# %%
report = dataset_report(DATA_DIR, image_limit=500)
print(report["counts"])
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample = pd.read_csv(SAMPLE_CSV)
print(train.head())
print(train[TARGET_COLUMN].value_counts(dropna=False))

# %% [markdown]
# ## 2. Leakage-aware split and loaders

# %%
split = make_split(train, target_columns=TARGET_COLUMN, valid_size=0.2, seed=competition.seed)
label_to_index, index_to_label = label_mapping(train[TARGET_COLUMN])
train_ds = ImageTableDataset(
    split.train,
    image_column=IMAGE_COLUMN,
    target_column=TARGET_COLUMN,
    root=DATA_DIR,
    transform=build_image_transforms(size=IMAGE_SIZE, training=True),
    label_to_index=label_to_index,
)
valid_ds = ImageTableDataset(
    split.valid,
    image_column=IMAGE_COLUMN,
    target_column=TARGET_COLUMN,
    root=DATA_DIR,
    transform=build_image_transforms(size=IMAGE_SIZE, training=False),
    label_to_index=label_to_index,
)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=competition.num_workers
)
valid_loader = DataLoader(
    valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=competition.num_workers
)

# %% [markdown]
# ## 3. Model, optimizer, scheduler, checkpoint

# %%
model = build_image_classifier(
    MODEL_NAME,
    len(label_to_index),
    pretrained_allowed=competition.pretrained_allowed,
)
trainer = Trainer(
    model,
    classification_loss(label_smoothing=0.05),
    training,
    output_dir=competition.output_dir,
    device=competition.device,
    metric_fn=lambda y, p: evaluate_metric("macro_f1", y, p),
    time_budget=budget,
)
history = trainer.fit(train_loader, valid_loader)
print(history[-1])

# %% [markdown]
# ## 4. Test inference and validated submission

# %%
test_ds = ImageTableDataset(
    test,
    image_column=IMAGE_COLUMN,
    root=DATA_DIR,
    transform=build_image_transforms(size=IMAGE_SIZE, training=False),
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=competition.num_workers
)
logits = predict_logits(model, test_loader, device=competition.device)
labels = [index_to_label[index] for index in logits.argmax(axis=1)]
submission = build_submission(sample, {TARGET_COLUMN: labels})
write_submission(
    submission, competition.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
submission.head()
