# %% [markdown]
# # Translation / seq2seq baseline
# The local model directory must be explicitly allowed by the problem statement.

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
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from olp_ai_26.core.colab import (
    ColabPaths,
    hf_precision_flags,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.nlp.seq2seq import build_seq2seq_model, decode_generated, tokenize_seq2seq_batch

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/seq2seq")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
MODEL_DIR = PROJECT_ROOT / "models" / "allowed_seq2seq_model"
SOURCE_COLUMN = "source"
TARGET_COLUMN = "target"
SOURCE_PREFIX = "translate: "
config = CompetitionConfig(
    data_dir=DATA_DIR,
    output_dir=paths.output_dir,
    pretrained_allowed=False,
)
config.prepare()

# %% [markdown]
# ## 1. Data and split

# %%
frame = pd.read_csv(DATA_DIR / "train.csv")
split = make_split(frame, valid_size=0.1, seed=config.seed)
tokenizer, model = build_seq2seq_model(
    MODEL_DIR,
    pretrained_allowed=config.pretrained_allowed,
    local_files_only=True,
)


def preprocess(batch):
    return tokenize_seq2seq_batch(
        batch,
        tokenizer,
        source_column=SOURCE_COLUMN,
        target_column=TARGET_COLUMN,
        source_prefix=SOURCE_PREFIX,
    )


train_ds = Dataset.from_pandas(split.train).map(
    preprocess, batched=True, remove_columns=list(frame.columns)
)
valid_ds = Dataset.from_pandas(split.valid).map(
    preprocess, batched=True, remove_columns=list(frame.columns)
)

# %% [markdown]
# ## 2. Training; switch to Adafactor only when memory requires it

# %%
arguments = Seq2SeqTrainingArguments(
    output_dir=str(config.output_dir / "seq2seq"),
    learning_rate=3e-4,
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    report_to="none",
    **hf_precision_flags(config.device),
)


def compute_metrics(output):
    predictions, labels = output
    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in row] for row in labels
    ]
    predicted_text = decode_generated(tokenizer, predictions)
    target_text = decode_generated(tokenizer, labels)
    return {"bleu": evaluate_metric("bleu", target_text, predicted_text)}


processor_parameter = (
    "processing_class"
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
    else "tokenizer"
)
trainer = Seq2SeqTrainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    **{processor_parameter: tokenizer},
)
trainer.train()
sync_artifacts(config.output_dir, paths.persistent_dir)
