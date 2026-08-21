# %% [markdown]
# # Image captioning baseline
# Use only a model directory explicitly permitted by the task's model allowlist.

# %%
from pathlib import Path

import pandas as pd
from transformers import Trainer, TrainingArguments

from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.split import make_split
from olp_ai_26.multimodal.image_to_text import ImageCaptionDataset, build_image_captioner

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DATA_DIR = Path("data")
MODEL_DIR = Path("models/allowed_caption_model")
IMAGE_COLUMN = "image"
CAPTION_COLUMN = "caption"
BATCH_SIZE = 8
config = CompetitionConfig(data_dir=DATA_DIR, pretrained_allowed=False)
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
    fp16=config.device == "cuda",
    remove_unused_columns=False,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=processor,
)
trainer.train()

# %% [markdown]
# ## 3. Before test inference
# Generate validation captions and reproduce the official BLEU/CIDEr/ROUGE implementation.
# Benchmark beam-search runtime before selecting `num_beams`.
