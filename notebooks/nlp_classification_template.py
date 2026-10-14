# %% [markdown]
# # NLP classification baseline
# This CPU baseline should be the first NLP submission before any Transformer fine-tuning.

# %% [markdown]
# ## Colab bootstrap

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd

from olp_ai_26.core.colab import ColabPaths, mount_google_drive, stage_data, sync_artifacts
from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.nlp.classification import TfidfTextClassifier

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/nlp")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "nlp_classification"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
GROUP_COLUMN, TIME_COLUMN = None, None
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
ID_COLUMN = "id"
METRIC = "macro_f1"
REFIT_ALL_TRAIN = False
C, CLASS_WEIGHT = 4.0, "balanced"
config = CompetitionConfig(data_dir=DATA_DIR, output_dir=paths.output_dir)
config.prepare()

# %% [markdown]
# ## 1. Load, inspect, split

# %%
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
print(train.info())
print(train[TARGET_COLUMN].value_counts(dropna=False))
split = make_split(
    train,
    target_columns=TARGET_COLUMN,
    group_column=GROUP_COLUMN,
    time_column=TIME_COLUMN,
    valid_size=0.2,
    seed=config.seed,
)

# %% [markdown]
# ## 2. CPU-first baseline and validation

# %%
model = TfidfTextClassifier(random_state=config.seed, C=C, class_weight=CLASS_WEIGHT)
model.fit(split.train[TEXT_COLUMN], split.train[TARGET_COLUMN])
valid_predictions = model.predict(split.valid[TEXT_COLUMN])
score = evaluate_metric(METRIC, split.valid[TARGET_COLUMN], valid_predictions)
print(f"macro_f1={score:.6f}")

# %% [markdown]
# ## 3. Refit, infer, validate submission

# %%
if REFIT_ALL_TRAIN:
    model.fit(train[TEXT_COLUMN], train[TARGET_COLUMN])
checkpoint = model.save(config.output_dir / "pipeline.pkl")
model = TfidfTextClassifier.load(checkpoint)
test_predictions = model.predict(test[TEXT_COLUMN])
submission = build_submission(
    sample,
    {TARGET_COLUMN: test_predictions},
    prediction_ids=test[[ID_COLUMN]],
    id_columns=ID_COLUMN,
)
write_submission(
    submission, config.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
record_run(
    config.output_dir,
    task="text_classification",
    model="word_char_tfidf_logistic",
    configuration={"competition": config.to_dict(), "C": C, "class_weight": CLASS_WEIGHT},
    validation_metrics={METRIC: score},
    split=split,
    data_summary={"train_rows": len(train), "test_rows": len(test)},
    inference_options={
        "text_column": TEXT_COLUMN,
        "id_column": ID_COLUMN,
        "refit_all_train": REFIT_ALL_TRAIN,
    },
    checkpoint=str(checkpoint),
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
