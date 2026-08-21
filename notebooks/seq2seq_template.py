# %% [markdown]
# # Translation / seq2seq baseline
# The local model directory must be explicitly allowed by the problem statement.

# %%
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.nlp.seq2seq import build_seq2seq_model, decode_generated, tokenize_seq2seq_batch

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DATA_DIR = Path("data")
MODEL_DIR = Path("models/allowed_seq2seq_model")
SOURCE_COLUMN = "source"
TARGET_COLUMN = "target"
SOURCE_PREFIX = "translate: "
config = CompetitionConfig(data_dir=DATA_DIR, pretrained_allowed=False)
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
    fp16=config.device == "cuda",
    report_to="none",
)


def compute_metrics(output):
    predictions, labels = output
    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in row] for row in labels
    ]
    predicted_text = decode_generated(tokenizer, predictions)
    target_text = decode_generated(tokenizer, labels)
    return {"bleu": evaluate_metric("bleu", target_text, predicted_text)}


trainer = Seq2SeqTrainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)
trainer.train()
