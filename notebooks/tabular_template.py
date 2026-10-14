# %% [markdown]
# # Tabular classification / regression
# Start with a dummy, then linear or histogram boosting. Fit preprocessing on training only.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import numpy as np
import pandas as pd

from olp_ai_26.core.colab import ColabPaths, mount_google_drive, stage_data, sync_artifacts
from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.ml.tabular import TabularBaseline, TabularConfig

# %% [markdown]
# ## 0. Configuration
# Select the official prediction type: labels, a positive-class probability, or regression value.

# %%
DRIVE_DATA_ARCHIVE = None
PERSISTENT_DIR = None
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, paths.data_dir)
DATA_DIR = paths.data_dir
ID_COLUMN, TARGET_COLUMN = "id", "label"
GROUP_COLUMN, TIME_COLUMN = None, None
DROP_COLUMNS = []  # exclude identifiers or features unavailable at prediction time
FEATURE_COLUMNS = None  # explicit list recommended after inspection
TASK, MODEL = "classification", "linear"  # regression; dummy; hist_gradient_boosting
METRIC = "macro_f1"  # accuracy, balanced_accuracy, roc_auc, log_loss, rmse, mae
PREDICTION_TYPE = "label"  # label | probability | regression
POSITIVE_CLASS = 1
REFIT_ALL_TRAIN = False  # freeze choices first; never include validation score from this refit
config = CompetitionConfig(data_dir=DATA_DIR, output_dir=paths.output_dir / "tabular", device="cpu")
training = TabularConfig(task=TASK, model=MODEL, seed=config.seed)
config.prepare()

# %% [markdown]
# ## 1. Inspect and choose a leakage-aware split

# %%
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
print(train.info())
print(train[TARGET_COLUMN].describe())
excluded = {ID_COLUMN, TARGET_COLUMN, GROUP_COLUMN, TIME_COLUMN, *DROP_COLUMNS}
features = FEATURE_COLUMNS or [column for column in train if column not in excluded]
if excluded.intersection(features):
    raise ValueError("ID, target, group/time, and dropped columns must not be features")
split = make_split(
    train,
    target_columns=TARGET_COLUMN if TASK == "classification" else None,
    group_column=GROUP_COLUMN,
    time_column=TIME_COLUMN,
    seed=config.seed,
)

# %% [markdown]
# ## 2. Fit and validate before any full-training refit

# %%
model = TabularBaseline(training).fit(split.train[features], split.train[TARGET_COLUMN])


def metric_predictions(frame):
    """Use probabilities only for metrics that require them; keep class order explicit."""
    if METRIC == "log_loss":
        return model.predict_proba(frame[features])
    if METRIC in {"roc_auc", "average_precision"}:
        if len(model.classes_) != 2:
            raise ValueError("This probability adapter is binary; wire official multiclass scoring")
        column = model.classes_.tolist().index(POSITIVE_CLASS)
        return model.predict_proba(frame[features])[:, column]
    return model.predict(frame[features])


metric_targets = split.valid[TARGET_COLUMN]
if METRIC in {"roc_auc", "average_precision"}:
    metric_targets = (metric_targets == POSITIVE_CLASS).astype(int)
metric_options = {"labels": model.classes_} if METRIC == "log_loss" else {}
score = evaluate_metric(METRIC, metric_targets, metric_predictions(split.valid), **metric_options)
print({METRIC: score})
if REFIT_ALL_TRAIN:
    model.fit(train[features], train[TARGET_COLUMN])
checkpoint = model.save(config.output_dir / "pipeline.pkl")
reloaded = TabularBaseline.load(checkpoint)
np.testing.assert_array_equal(model.predict(test[features]), reloaded.predict(test[features]))

# %% [markdown]
# ## 3. Infer, align IDs, validate, and synchronize the complete bundle

# %%
if PREDICTION_TYPE == "probability":
    positive_index = model.classes_.tolist().index(POSITIVE_CLASS)
    predictions = model.predict_proba(test[features])[:, positive_index]
elif PREDICTION_TYPE in {"label", "regression"}:
    predictions = model.predict(test[features])
else:
    raise ValueError("Unknown prediction type")
submission = build_submission(
    sample, {TARGET_COLUMN: predictions}, prediction_ids=test[[ID_COLUMN]], id_columns=ID_COLUMN
)
write_submission(
    submission,
    config.output_dir / "submission.csv",
    sample=sample,
    id_columns=ID_COLUMN,
    allowed_labels={TARGET_COLUMN: model.classes_}
    if TASK == "classification" and PREDICTION_TYPE == "label"
    else None,
)
record_run(
    config.output_dir,
    task=TASK,
    model=MODEL,
    configuration={"competition": config.to_dict(), "training": training},
    validation_metrics={METRIC: score},
    split=split,
    data_summary={"features": features, "train_rows": len(train), "test_rows": len(test)},
    inference_options={
        "prediction_type": PREDICTION_TYPE,
        "positive_class": POSITIVE_CLASS,
        "refit_all_train": REFIT_ALL_TRAIN,
    },
    checkpoint=str(checkpoint),
)
sync_artifacts(config.output_dir, paths.persistent_dir)
submission.head()
