# %% [markdown]
# # Token classification / named-entity recognition baseline
# Use this when each token receives a BIO/BILOU-style label. The official entity-level F1 matters.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ("transformers", "datasets", "accelerate", "seqeval")
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import ast
import json

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, TrainingArguments

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    hf_precision_flags,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.config import CompetitionConfig, TimeBudget
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import hf_time_budget_callback, local_hf_trainer_class
from olp_ai_26.nlp.token_classification import (
    build_token_classifier,
    predict_word_tags,
    tokenize_and_align_labels,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration
# CSV cells below contain Python/JSON-like lists, e.g. `["Ha", "Noi"]` and `[1, 2]`.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/token_classification")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "token_classification"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TRAIN_CSV = DATA_DIR / "train.csv"
TOKENS_COLUMN, LABELS_COLUMN = "tokens", "ner_tags"
MODEL_PATH = DATA_DIR / "models" / "token_encoder"
LABEL_NAMES = ["O", "B-ENT", "I-ENT"]  # replace in exact organizer index order
LOCAL_FILES_ONLY = True
PRETRAINED_ALLOWED = False  # opt in only for the exact permitted local checkpoint
BATCH_SIZE, EPOCHS, LEARNING_RATE = 16, 5, 3e-5
ID_COLUMN = "id"
GROUP_COLUMN, TIME_COLUMN = None, None
MAX_LENGTH = 256
OUTPUT_LABEL_NAMES = False  # JSON integer tags by default; match the official sample
config = CompetitionConfig(
    data_dir=DATA_DIR, output_dir=paths.output_dir, pretrained_allowed=PRETRAINED_ALLOWED
)
config.prepare()
budget = TimeBudget(config.time_budget_minutes)
print(gpu_report())

# %% [markdown]
# ## 1. Parse list columns and align word labels to subword tokens

# %%
frame = pd.read_csv(TRAIN_CSV, converters={ID_COLUMN: str})
print(frame.head())
for column in (TOKENS_COLUMN, LABELS_COLUMN):
    frame[column] = frame[column].map(ast.literal_eval)
split = make_split(frame, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, seed=config.seed)
train_data = Dataset.from_pandas(split.train[[TOKENS_COLUMN, LABELS_COLUMN]], preserve_index=False)
valid_data = Dataset.from_pandas(split.valid[[TOKENS_COLUMN, LABELS_COLUMN]], preserve_index=False)
tokenizer, model = build_token_classifier(
    MODEL_PATH,
    num_labels=len(LABEL_NAMES),
    pretrained_allowed=PRETRAINED_ALLOWED,
    local_files_only=LOCAL_FILES_ONLY,
)


def align(batch):
    """Tokenize split words and assign -100 to ignored special/subword positions."""
    return tokenize_and_align_labels(
        batch,
        tokenizer,
        token_column=TOKENS_COLUMN,
        label_column=LABELS_COLUMN,
        max_length=MAX_LENGTH,
    )


train_ds = train_data.map(align, batched=True, remove_columns=[TOKENS_COLUMN, LABELS_COLUMN])
valid_ds = valid_data.map(align, batched=True, remove_columns=[TOKENS_COLUMN, LABELS_COLUMN])

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
    metric_for_best_model="entity_f1",
    greater_is_better=True,
    seed=config.seed,
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


LocalTrainer = local_hf_trainer_class(generation=False)
trainer = LocalTrainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[hf_time_budget_callback(budget)],
)
trainer.train()
validation = trainer.evaluate()
trainer.save_model(paths.output_dir / "best_ner")
tokenizer.save_pretrained(paths.output_dir / "best_ner")

# %% [markdown]
# ## 3. Word-level test inference, exact ID alignment, and persisted recipe
# This adapter serializes a JSON list per row. Replace only this serialization for another contract.

# %%
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
test[TOKENS_COLUMN] = test[TOKENS_COLUMN].map(ast.literal_eval)
tags = predict_word_tags(
    model,
    tokenizer,
    test[TOKENS_COLUMN].tolist(),
    device=config.device,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    label_names=LABEL_NAMES if OUTPUT_LABEL_NAMES else None,
)
submission = build_submission(
    sample,
    {LABELS_COLUMN: [json.dumps(row, ensure_ascii=False) for row in tags]},
    prediction_ids=test[[ID_COLUMN]],
    id_columns=ID_COLUMN,
)
write_submission(
    submission, paths.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
record_run(
    paths.output_dir,
    task="token_classification",
    model=str(MODEL_PATH),
    configuration={
        "competition": config.to_dict(),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
    },
    validation_metrics={"entity_f1": validation["eval_entity_f1"]},
    split=split,
    inference_options={
        "label_names": LABEL_NAMES,
        "max_length": MAX_LENGTH,
        "output_label_names": OUTPUT_LABEL_NAMES,
    },
    checkpoint="best_ner",
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
