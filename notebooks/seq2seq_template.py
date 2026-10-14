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
REQUIRED_MODULES = ("transformers", "datasets", "accelerate", "sacrebleu", "rouge_score")
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
from olp_ai_26.core.config import CompetitionConfig, TimeBudget
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.core.trainer import hf_time_budget_callback, local_hf_trainer_class
from olp_ai_26.nlp.seq2seq import (
    build_seq2seq_model,
    decode_generated,
    generate_texts,
    tokenize_seq2seq_batch,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/seq2seq")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "seq2seq"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
MODEL_DIR = PROJECT_ROOT / "models" / "allowed_seq2seq_model"
SOURCE_COLUMN = "source"
TARGET_COLUMN = "target"
SOURCE_PREFIX = "translate: "
ID_COLUMN = "id"
GROUP_COLUMN, TIME_COLUMN = None, None
MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, NUM_BEAMS = 256, 128, 1
METRIC = "bleu"
BATCH_SIZE, EPOCHS, LEARNING_RATE, ACCUMULATION = 8, 8, 3e-4, 2
config = CompetitionConfig(
    data_dir=DATA_DIR,
    output_dir=paths.output_dir,
    pretrained_allowed=False,
)
config.prepare()
budget = TimeBudget(config.time_budget_minutes)

# %% [markdown]
# ## 1. Data and split

# %%
frame = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
print(frame.head())
split = make_split(
    frame, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, valid_size=0.1, seed=config.seed
)
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
        max_source_length=MAX_SOURCE_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
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
model.generation_config.max_new_tokens = None
model.generation_config.max_length = MAX_TARGET_LENGTH
model.generation_config.num_beams = NUM_BEAMS
arguments = Seq2SeqTrainingArguments(
    output_dir=str(config.output_dir / "seq2seq"),
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=ACCUMULATION,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=METRIC,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=NUM_BEAMS,
    seed=config.seed,
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
    return {METRIC: evaluate_metric(METRIC, target_text, predicted_text)}


processor_parameter = (
    "processing_class"
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
    else "tokenizer"
)
LocalTrainer = local_hf_trainer_class(generation=True)
trainer = LocalTrainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[hf_time_budget_callback(budget)],
    **{processor_parameter: tokenizer},
)
trainer.train()
validation = trainer.evaluate()
trainer.save_model(config.output_dir / "best_model")
tokenizer.save_pretrained(config.output_dir / "best_model")

# %% [markdown]
# ## 3. Full test generation and validated submission

# %%
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})


texts = generate_texts(
    model,
    tokenizer,
    test[SOURCE_COLUMN].fillna("").tolist(),
    device=config.device,
    source_prefix=SOURCE_PREFIX,
    max_source_length=MAX_SOURCE_LENGTH,
    max_target_length=MAX_TARGET_LENGTH,
    num_beams=NUM_BEAMS,
    batch_size=BATCH_SIZE,
)
submission = build_submission(
    sample, {TARGET_COLUMN: texts}, prediction_ids=test[[ID_COLUMN]], id_columns=ID_COLUMN
)
write_submission(
    submission, config.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
record_run(
    config.output_dir,
    task="seq2seq",
    model=str(MODEL_DIR),
    configuration={
        "competition": config.to_dict(),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "gradient_accumulation_steps": ACCUMULATION,
    },
    validation_metrics={METRIC: validation[f"eval_{METRIC}"]},
    split=split,
    inference_options={
        "source_prefix": SOURCE_PREFIX,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "num_beams": NUM_BEAMS,
    },
    checkpoint="best_model",
)
sync_artifacts(config.output_dir, paths.persistent_dir)
submission.head()
