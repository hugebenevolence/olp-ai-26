# %% [markdown]
# # NLP classification baseline
# This CPU baseline should be the first NLP submission before any Transformer fine-tuning.

# %% [markdown]
# ## Colab bootstrap

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd

from olp_ai_26.core.colab import ColabPaths, mount_google_drive, stage_data, sync_artifacts
from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.metrics import evaluate_metric
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
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
ID_COLUMN = "id"
config = CompetitionConfig(data_dir=DATA_DIR, output_dir=paths.output_dir)
config.prepare()

# %% [markdown]
# ## 1. Load, inspect, split

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(train.info())
print(train[TARGET_COLUMN].value_counts(dropna=False))
split = make_split(train, target_columns=TARGET_COLUMN, valid_size=0.2, seed=config.seed)

# %% [markdown]
# ## 2. CPU-first baseline and validation

# %%
model = TfidfTextClassifier(random_state=config.seed)
model.fit(split.train[TEXT_COLUMN], split.train[TARGET_COLUMN])
valid_predictions = model.predict(split.valid[TEXT_COLUMN])
score = evaluate_metric("macro_f1", split.valid[TARGET_COLUMN], valid_predictions)
print(f"macro_f1={score:.6f}")

# %% [markdown]
# ## 3. Refit, infer, validate submission

# %%
model.fit(train[TEXT_COLUMN], train[TARGET_COLUMN])
test_predictions = model.predict(test[TEXT_COLUMN])
submission = build_submission(sample, {TARGET_COLUMN: test_predictions})
write_submission(
    submission, config.output_dir / "submission.csv", sample=sample, id_columns=ID_COLUMN
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
