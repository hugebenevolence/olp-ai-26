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
REQUIRED_MODULES = ("transformers", "accelerate", "sacrebleu", "rouge_score")
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

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
from olp_ai_26.multimodal.image_to_text import ImageCaptionDataset, build_image_captioner

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/captioning")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "image_captioning"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
MODEL_DIR = PROJECT_ROOT / "models" / "allowed_caption_model"
IMAGE_COLUMN = "image"
CAPTION_COLUMN = "caption"
BATCH_SIZE, EPOCHS, LEARNING_RATE, ACCUMULATION = 8, 6, 3e-4, 2
ID_COLUMN = "id"
GROUP_COLUMN, TIME_COLUMN = None, None
METRIC = "bleu"
MAX_LENGTH, NUM_BEAMS = 64, 1
config = CompetitionConfig(
    data_dir=DATA_DIR,
    output_dir=paths.output_dir,
    pretrained_allowed=False,
)
config.prepare()
budget = TimeBudget(config.time_budget_minutes)

# %% [markdown]
# ## 1. Data and model policy

# %%
frame = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
print(frame.head())
split = make_split(
    frame, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, valid_size=0.1, seed=config.seed
)
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
    max_length=MAX_LENGTH,
    root=DATA_DIR,
)
valid_ds = ImageCaptionDataset(
    split.valid,
    processor=processor,
    image_column=IMAGE_COLUMN,
    caption_column=CAPTION_COLUMN,
    max_length=MAX_LENGTH,
    root=DATA_DIR,
)

# %% [markdown]
# ## 2. Fine-tune the smallest valid baseline first

# %%
model.generation_config.max_new_tokens = None
model.generation_config.max_length = MAX_LENGTH
model.generation_config.num_beams = NUM_BEAMS
arguments = Seq2SeqTrainingArguments(
    output_dir=str(config.output_dir / "captioning"),
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=ACCUMULATION,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=METRIC,
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=NUM_BEAMS,
    seed=config.seed,
    remove_unused_columns=False,
    report_to="none",
    **hf_precision_flags(config.device),
)
processor_parameter = (
    "processing_class"
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
    else "tokenizer"
)


def compute_metrics(result):
    """Evaluate generated captions against the original untruncated validation references."""
    token_ids = result.predictions.copy()
    token_ids[token_ids == -100] = processor.tokenizer.pad_token_id
    captions = processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return {
        METRIC: evaluate_metric(METRIC, split.valid[CAPTION_COLUMN].fillna("").tolist(), captions)
    }


LocalTrainer = local_hf_trainer_class(generation=True)
trainer = LocalTrainer(
    model=model,
    args=arguments,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
    callbacks=[hf_time_budget_callback(budget)],
    **{processor_parameter: processor},
)
trainer.train()
validation = trainer.evaluate()
trainer.save_model(config.output_dir / "best_model")
processor.save_pretrained(config.output_dir / "best_model")

# %% [markdown]
# ## 3. Full image-to-text inference and validated submission
# Multiple reference captions or CIDEr require the organizer's evaluator rather than BLEU here.

# %%
from PIL import Image

from olp_ai_26.multimodal.image_to_text import generate_captions

test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
captions = []
for start in range(0, len(test), BATCH_SIZE):
    images = []
    for relative_path in test.iloc[start : start + BATCH_SIZE][IMAGE_COLUMN]:
        with Image.open(DATA_DIR / str(relative_path)) as source:
            images.append(source.convert("RGB"))
    captions.extend(
        generate_captions(
            model,
            processor,
            images,
            device=config.device,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
        )
    )
submission = build_submission(
    sample, {CAPTION_COLUMN: captions}, prediction_ids=test[[ID_COLUMN]], id_columns=ID_COLUMN
)
write_submission(
    submission, config.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
record_run(
    config.output_dir,
    task="image_captioning",
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
        "max_length": MAX_LENGTH,
        "num_beams": NUM_BEAMS,
        "image_column": IMAGE_COLUMN,
    },
    checkpoint="best_model",
)
sync_artifacts(config.output_dir, paths.persistent_dir)
submission.head()
