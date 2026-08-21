# %% [markdown]
# # Token classification / named-entity recognition baseline
# Use this when each token receives a BIO/BILOU-style label. The official entity-level F1 matters.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import ast

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    hf_precision_flags,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.nlp.token_classification import build_token_classifier, tokenize_and_align_labels

# %% [markdown]
# ## 0. Drag-and-plug configuration
# CSV cells below contain Python/JSON-like lists, e.g. `["Ha", "Noi"]` and `[1, 2]`.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/token_classification")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train.csv"
TOKENS_COLUMN, LABELS_COLUMN = "tokens", "ner_tags"
MODEL_PATH = DATA_DIR / "models" / "token_encoder"
LABEL_NAMES = ["O", "B-ENT", "I-ENT"]  # replace in exact organizer index order
LOCAL_FILES_ONLY = True
PRETRAINED_ALLOWED = True  # false initializes only from the local architecture config
BATCH_SIZE, EPOCHS, LEARNING_RATE = 16, 5, 3e-5
print(gpu_report())

# %% [markdown]
# ## 1. Parse list columns and align word labels to subword tokens

# %%
frame = pd.read_csv(TRAIN_CSV)
for column in (TOKENS_COLUMN, LABELS_COLUMN):
    frame[column] = frame[column].map(ast.literal_eval)
dataset = Dataset.from_pandas(frame[[TOKENS_COLUMN, LABELS_COLUMN]], preserve_index=False)
split = dataset.train_test_split(test_size=0.2, seed=42)
tokenizer, model = build_token_classifier(
    MODEL_PATH,
    num_labels=len(LABEL_NAMES),
    pretrained_allowed=PRETRAINED_ALLOWED,
    local_files_only=LOCAL_FILES_ONLY,
)


def align(batch):
    """Tokenize split words and assign -100 to ignored special/subword positions."""
    return tokenize_and_align_labels(
        batch, tokenizer, token_column=TOKENS_COLUMN, label_column=LABELS_COLUMN
    )


tokenized = split.map(align, batched=True, remove_columns=[TOKENS_COLUMN, LABELS_COLUMN])

# %% [markdown]
# ## 2. Train with dynamic padding

# %%
arguments = TrainingArguments(
    output_dir=str(paths.output_dir / "ner"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    **hf_precision_flags("cuda" if torch.cuda.is_available() else "cpu"),
)


def compute_metrics(result):
    """Remove ignored tokens and compute entity-level seqeval F1."""
    predicted = result.predictions.argmax(-1)
    references, hypotheses = [], []
    for truth_row, pred_row in zip(result.label_ids, predicted, strict=True):
        keep = truth_row != -100
        references.append([LABEL_NAMES[index] for index in truth_row[keep]])
        hypotheses.append([LABEL_NAMES[index] for index in pred_row[keep]])
    return {"entity_f1": evaluate_metric("seqeval_f1", references, hypotheses)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(paths.output_dir / "best_ner")
sync_artifacts(paths.output_dir, paths.persistent_dir)
