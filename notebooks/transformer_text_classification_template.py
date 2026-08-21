# %% [markdown]
# # Transformer text classification baseline
# Use TF-IDF first; move here when contextual language understanding improves validation.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    hf_precision_flags,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.cv.classification import label_mapping
from olp_ai_26.nlp.classification import build_hf_text_classifier

# %% [markdown]
# ## 0. Drag-and-plug configuration
# Point `MODEL_PATH` at a local permitted model/tokenizer directory. Set `LOCAL_FILES_ONLY=False`
# only when downloading that exact model is legal. Common families: BERT, RoBERTa, DeBERTa.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/transformer_text")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV, TEST_CSV = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"
TEXT_COLUMN, TARGET_COLUMN, ID_COLUMN = "text", "label", "id"
MODEL_PATH = DATA_DIR / "models" / "text_encoder"
LOCAL_FILES_ONLY = True
PRETRAINED_ALLOWED = True  # false initializes only from the local architecture config
MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE = 256, 16, 4, 2e-5
print(gpu_report())

# %% [markdown]
# ## 1. Split, tokenize, and construct labels

# %%
train, test, sample = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV), pd.read_csv(SAMPLE_CSV)
split = make_split(train, target_columns=TARGET_COLUMN, valid_size=0.2, seed=42)
label_to_index, index_to_label = label_mapping(train[TARGET_COLUMN])
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=LOCAL_FILES_ONLY)


def tokenize(batch):
    """Tokenize a batch with truncation; dynamic padding happens in the data collator."""
    return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)


def as_dataset(frame, with_labels=True):
    """Convert a pandas split into a tokenized Hugging Face Dataset."""
    payload = {TEXT_COLUMN: frame[TEXT_COLUMN].fillna("").tolist()}
    if with_labels:
        payload["labels"] = frame[TARGET_COLUMN].map(label_to_index).tolist()
    return Dataset.from_dict(payload).map(tokenize, batched=True, remove_columns=[TEXT_COLUMN])


train_ds, valid_ds = as_dataset(split.train), as_dataset(split.valid)
test_ds = as_dataset(test, with_labels=False)

# %% [markdown]
# ## 2. Local model, official metric callback, and training

# %%
model = build_hf_text_classifier(
    MODEL_PATH,
    num_labels=len(label_to_index),
    pretrained_allowed=PRETRAINED_ALLOWED,
    local_files_only=LOCAL_FILES_ONLY,
)
arguments = TrainingArguments(
    output_dir=str(paths.output_dir / "hf_text"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none",
    **hf_precision_flags("cuda" if torch.cuda.is_available() else "cpu"),
)


def compute_metrics(result):
    """Convert logits to class indexes and compute the selected official metric."""
    predictions = np.asarray(result.predictions).argmax(1)
    return {"macro_f1": evaluate_metric("macro_f1", result.label_ids, predictions)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()

# %% [markdown]
# ## 3. Inference and validated submission

# %%
logits = trainer.predict(test_ds).predictions
labels = [index_to_label[index] for index in np.asarray(logits).argmax(1)]
submission = build_submission(sample, {TARGET_COLUMN: labels})
write_submission(
    submission, paths.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
