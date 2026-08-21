# %% [markdown]
# # NLP classification baseline
# This CPU baseline should be the first NLP submission before any Transformer fine-tuning.

# %%
from pathlib import Path

import pandas as pd

from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.nlp.classification import TfidfTextClassifier

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DATA_DIR = Path("data")
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
ID_COLUMN = "id"
config = CompetitionConfig(data_dir=DATA_DIR)
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
submission.head()
