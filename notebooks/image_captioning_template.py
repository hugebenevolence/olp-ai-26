# %% [markdown]
# # Image captioning baseline
# Use only a model directory explicitly permitted by the task's model allowlist.

# %% [markdown]
# ## Colab bootstrap

# %%
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
from transformers import Trainer, TrainingArguments

from olp_ai_26.core.colab import (
    ColabPaths,
    hf_precision_flags,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.split import make_split
from olp_ai_26.multimodal.image_to_text import ImageCaptionDataset, build_image_captioner

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/captioning")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
MODEL_DIR = PROJECT_ROOT / "models" / "allowed_caption_model"
IMAGE_COLUMN = "image"
CAPTION_COLUMN = "caption"
BATCH_SIZE = 8
config = CompetitionConfig(
    data_dir=DATA_DIR,
    output_dir=paths.output_dir,
    pretrained_allowed=False,
)
config.prepare()

# %% [markdown]
# ## 1. Data and model policy

# %%
frame = pd.read_csv(DATA_DIR / "train.csv")
split = make_split(frame, valid_size=0.1, seed=config.seed)
processor, model = build_image_captioner(
    MODEL_DIR,
    pretrained_allowed=config.pretrained_allowed,
    local_files_only=True,
)
train_ds = ImageCaptionDataset(
    split.train,
    processor=processor,
    image_column=IMAGE_COLUMN,
    caption_column=CAPTION_COLUMN,
    root=DATA_DIR,
)
valid_ds = ImageCaptionDataset(
    split.valid,
    processor=processor,
    image_column=IMAGE_COLUMN,
    caption_column=CAPTION_COLUMN,
    root=DATA_DIR,
)

# %% [markdown]
# ## 2. Fine-tune the smallest valid baseline first

# %%
arguments = TrainingArguments(
    output_dir=str(config.output_dir / "captioning"),
    learning_rate=3e-4,
    num_train_epochs=6,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    report_to="none",
    **hf_precision_flags(config.device),
)
processor_parameter = (
    "processing_class"
    if "processing_class" in inspect.signature(Trainer.__init__).parameters
    else "tokenizer"
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    **{processor_parameter: processor},
)
trainer.train()
sync_artifacts(config.output_dir, paths.persistent_dir)

# %% [markdown]
# ## 3. Before test inference
# Generate validation captions and reproduce the official BLEU/CIDEr/ROUGE implementation.
# Benchmark beam-search runtime before selecting `num_beams`.
