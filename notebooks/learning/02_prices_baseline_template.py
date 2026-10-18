# %% [markdown]
# # House Prices — fill-in baseline
#
# Problem family: **Skewed tabular regression**. Official metric: **RMSE of natural log
# prices**.
#
# **Learning objective:** Train-fold imputation/one-hot scaling + Ridge on log prices.
#
# [Dataset and original
# task](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) ·
# [Primary solution
# evidence](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard)
#
# The author's notebook is titled Top 4% and studies stacked regressions. This is historical
# author-reported evidence, not a current winning rank. This lesson implements a smaller
# train-fold stack; it is an advanced study route, not a verified global SOTA.
#
# **Watch for:** Use log(price), not a similar log1p metric; never remove validation outliers to
# improve the score.
#
# **Standalone:** this file contains every project-specific helper it uses. Only installed
# third-party libraries and your local data/assets are required; no repository checkout, source
# imports, project installation, network access or external bootstrap is used. Edit DATA_DIR and
# output/storage settings in stage 1. Train media on local disk, then persist your output bundle.
#
# This notebook is a learning exercise, not a claimed leaderboard reproduction. No trained
# weights or real Kaggle data are bundled. Keep grading answers closed. Use only explicitly
# permitted resources. Internet/prepared-code access is not assumed on contest day.
#
# Fill only the TODO blocks. Every code cell is followed by a closed Reveal expected code
# section. Attempt, run, then reveal; equivalent correct implementations are welcome. Record
# hints needed. Setup/configuration is supplied.
#
# After the lesson, close both notebooks and rebuild the essential route from memory. See the
# repository's blank-notebook playbook for the final transfer test.
#

# %% [markdown]
# ## 1. Bootstrap and configuration
#
# Run this supplied setup cell; edit only configuration values. Later algorithm cells contain
# TODO blocks. No weights or datasets are downloaded. Stage data locally before repeated reads.
# Each candidate gets a new output directory.
#
#
#

# %% [exercise]
import json
import pickle
import random
import shutil
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _write(value: Any, path: Path | str, kind: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        if kind == "json":
            temporary.write_text(
                json.dumps(
                    value, default=_json_default, indent=2, ensure_ascii=False, allow_nan=False
                ),
                encoding="utf-8",
            )
        elif kind == "pickle":
            with temporary.open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            import torch

            torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_json(value: Any, path: Path | str) -> Path:
    """Atomically save UTF-8 metadata, including paths, dataclasses, and NumPy values."""
    return _write(value, path, "json")


def save_pickle(value: Any, path: Path | str) -> Path:
    """Atomically save a fitted local pipeline with preprocessing and original label values."""
    return _write(value, path, "pickle")


def load_pickle(path: Path | str) -> Any:
    """Reload a trusted, locally produced pickle; never load untrusted downloaded pickle files."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_torch_state(value: Any, path: Path | str) -> Path:
    """Atomically save torch state without importing torch until this function is used."""
    return _write(value, path, "torch")


@dataclass
class CompetitionConfig:
    """Explicit notebook state; no installed project is required."""

    data_dir: Path
    output_dir: Path
    seed: int = 42
    num_workers: int = 0
    pretrained_allowed: bool = False
    time_budget_minutes: float = 60
    device: str = "auto"

    def prepare(self) -> None:
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def to_dict(self) -> dict:
        return {**asdict(self), "data_dir": str(self.data_dir), "output_dir": str(self.output_dir)}


class TimeBudget:
    """Leave delivery time available; a running batch/fit cannot be interrupted safely."""

    def __init__(self, minutes: float, reserve_minutes: float = 10) -> None:
        if not 0 <= reserve_minutes < minutes:
            raise ValueError("Require 0 <= reserve < total minutes")
        self.deadline = time.monotonic() + 60 * (minutes - reserve_minutes)

    @property
    def should_stop(self) -> bool:
        return time.monotonic() >= self.deadline

    def checkpoint(self, stage: str) -> None:
        if self.should_stop:
            raise TimeoutError(f"Delivery reserve reached before {stage}")


def runtime_paths(
    *, runtime_dir: Path | None = None, persistent_dir: Path | None = None
) -> SimpleNamespace:
    """Media stays on local disk; optional durable storage is an explicit destination."""
    root = (
        Path(runtime_dir)
        if runtime_dir is not None
        else (Path("/content/olp_runtime") if "google.colab" in sys.modules else Path.cwd())
    )
    data, output = root / "data", root / "outputs"
    data.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        data_dir=data,
        output_dir=output,
        persistent_dir=Path(persistent_dir) if persistent_dir is not None else None,
    )


def mount_google_drive() -> None:
    """Call only when an explicitly configured Colab path uses Drive."""
    from google.colab import drive

    drive.mount("/content/drive")


def stage_data(source: Path | str, destination: Path | str) -> Path:
    """Copy a folder or extract a ZIP once, then train from the local destination."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError("Staging destination must be separate from its source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Reuse staged DATA_DIR and set DATA_ARCHIVE=None")
    destination.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise ValueError("Archive contains a path outside the destination")
            archive.extractall(destination)
    return destination


def sync_artifacts(source: Path | str, destination: Path | str | None) -> list[Path]:
    """Copy the entire candidate, including encoders, preserving separate run folders."""
    if destination is None:
        return []
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Output and backup folders must be separate")
    copied = []
    for path in source.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            try:
                shutil.copy2(path, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            copied.append(target)
    return copied


def gpu_report() -> dict:
    """Inspect the actual runtime before choosing image size, batch size or precision."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "precision": "float32"}
    free, total = torch.cuda.mem_get_info()
    return {
        "device": torch.cuda.get_device_name(),
        "free_GiB": free / 2**30,
        "total_GiB": total / 2**30,
        "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
    }


def evaluate_metric(name: str, truth: Any, prediction: Any, **kwargs: Any) -> float:
    """Only the metrics used in this lesson; task-specific graders appear beside the model."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    if name == "roc_auc":
        return float(roc_auc_score(truth, prediction, **kwargs))
    if name == "accuracy":
        return float(accuracy_score(truth, prediction))
    if name == "dice_mean_image":
        y, p = np.asarray(truth, bool), np.asarray(prediction, bool)
        if y.shape != p.shape or not y.size:
            raise ValueError("Need aligned nonempty masks")
        axes = tuple(range(1, y.ndim))
        denominator = y.sum(axis=axes) + p.sum(axis=axes)
        return float(
            np.divide(
                2 * (y & p).sum(axis=axes), denominator, out=np.ones(len(y)), where=denominator != 0
            ).mean()
        )
    raise ValueError(f"Unsupported lesson metric: {name}")


FAMILY, VARIANT = "02_prices", "baseline"
DATA_ARCHIVE, PERSISTENT_DIR = None, None
if "google.colab" in sys.modules and any(
    "/content/drive/" in str(value) for value in (DATA_ARCHIVE, PERSISTENT_DIR)
):
    mount_google_drive()
paths = runtime_paths(persistent_dir=PERSISTENT_DIR)
DATA_DIR = (
    paths.data_dir / "learning" / FAMILY
    if "google.colab" in sys.modules
    else Path.cwd() / "data" / "learning" / FAMILY
)
if DATA_ARCHIVE is not None:
    stage_data(DATA_ARCHIVE, DATA_DIR)
RUN_ID = f"{FAMILY}-{VARIANT}-{uuid4().hex[:8]}"
OUTPUT = paths.output_dir / "learning" / RUN_ID
DURABLE_OUTPUT = paths.persistent_dir / RUN_ID if paths.persistent_dir is not None else None
PRETRAINED_ALLOWED, WEIGHT_PATH = False, None
SEED, MAX_ROWS, IMAGE_SIZE, BATCH_SIZE, EPOCHS = 42, 6000, 128, 16, 3
TIME_BUDGET_MINUTES, DELIVERY_RESERVE_MINUTES = 60, 10
GROUP_COLUMN, TIME_COLUMN = None, None
USE_IMBALANCE_WEIGHTS = False
USE_HISTORY_FEATURES = False
MODEL_NAME = "resnet18"
IMAGE_SUFFIX = ".png"
TEXT_MODEL_DIR, IMAGE_MODEL_CHECKPOINT, EMBEDDING_ARCHIVE = None, None, None
ASSET_PROVENANCE = (
    ""  # exact asset/version/license; required when enabling local pretrained features
)
EMBEDDING_PATH, EMBEDDING_DIM, MAX_LENGTH = None, 64, 128
THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
competition = CompetitionConfig(
    data_dir=DATA_DIR,
    output_dir=OUTPUT,
    seed=SEED,
    num_workers=0,
    pretrained_allowed=PRETRAINED_ALLOWED,
    time_budget_minutes=TIME_BUDGET_MINUTES,
)
competition.prepare()
budget = TimeBudget(TIME_BUDGET_MINUTES, reserve_minutes=DELIVERY_RESERVE_MINUTES)
print(gpu_report())
print("Local data:", DATA_DIR, "New candidate:", OUTPUT)

# %% [markdown]
# ## 2. Load data and enforce the contract
#
# Check schema and identity before fitting. A row is not always a prediction unit: detection has
# several annotation rows per image, and matching has several records per product.
#
#
#

# %% [exercise]
ID_COLUMN = "Id"
TARGET_COLUMNS = ["SalePrice"]
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
assert (train[TARGET_COLUMNS[0]] > 0).all()
if MAX_ROWS is not None and len(train) > MAX_ROWS:
    train = train.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
for name, frame in (("train", train), ("test", test), ("sample", sample)):
    if (
        frame[ID_COLUMN].isna().any()
        or frame[ID_COLUMN].eq("").any()
        or frame[ID_COLUMN].duplicated().any()
    ):
        raise ValueError(f"{name} must have unique, nonempty prediction-unit IDs")
assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN])
assert set(test[ID_COLUMN]) == set(sample[ID_COLUMN])
assert sample.columns.tolist() == [ID_COLUMN, *TARGET_COLUMNS]
print(train.head(3).to_string(index=False))
print("Shapes:", train.shape, test.shape, sample.shape)

# %% [markdown]
# ## 3. Inspect and visualize the data
#
# Write down three observations from these plots before revealing any solution: missingness,
# examples that look wrong, and a possible leakage unit. Never select experiments using hidden
# test labels.
#
#
#

# %% [exercise]
print(train.dtypes.to_string())
print(train.describe(include="all").to_string())
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
train.isna().mean().sort_values().tail(12).plot.barh(ax=axes[0], title="Missing fraction")
train["SalePrice"].hist(ax=axes[1], bins=30)
axes[1].set_title("Raw prices")
np.log(train["SalePrice"]).hist(ax=axes[2], bins=30)
axes[2].set_title("Natural log prices")
plt.tight_layout()
fig.savefig(OUTPUT / "inspection.png")
plt.show()
plt.close(fig)

# %% [markdown]
# ## 4. Freeze a leakage-aware split
#
# Keep these exact folds for baseline/advanced comparisons. Random stratification is only
# appropriate for independent samples. Set a group or time column when the acquisition process
# demands it; assert the chosen boundary.
#
#
#

# %% [exercise]
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split


@dataclass(slots=True)
class SplitResult:
    """Training/validation tables plus their original positional indexes."""

    train: pd.DataFrame
    valid: pd.DataFrame
    train_indices: np.ndarray
    valid_indices: np.ndarray


def _stratification_key(frame: pd.DataFrame, targets: Sequence[str]) -> pd.Series:
    if not targets:
        raise ValueError("At least one target is required for stratification")
    if len(targets) == 1:
        return frame[targets[0]].astype(str)
    return frame[list(targets)].astype(str).agg("|".join, axis=1)


def make_split(
    frame: pd.DataFrame,
    *,
    target_columns: str | Sequence[str] | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    valid_size: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    """Create a deterministic holdout with leakage-aware group and time options."""
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("At least two rows are required")
    targets = [target_columns] if isinstance(target_columns, str) else list(target_columns or [])
    for column in [*targets, group_column, time_column]:
        if column is not None and column not in frame.columns:
            raise KeyError(f"Missing split column: {column}")

    indices = np.arange(len(frame))
    if time_column:
        timestamps = pd.to_datetime(frame[time_column], errors="raise")
        if timestamps.isna().any():
            raise ValueError("Time split requires complete timestamps")
        order = np.argsort(timestamps.to_numpy(), kind="stable")
        ordered_times = timestamps.to_numpy()[order]
        boundaries = np.flatnonzero(ordered_times[1:] != ordered_times[:-1]) + 1
        if not len(boundaries):
            raise ValueError("Need at least two distinct times for a strict chronological split")
        desired = len(frame) * (1 - valid_size)
        cut = int(boundaries[np.argmin(abs(boundaries - desired))])
        train_idx, valid_idx = order[:cut], order[cut:]
    elif group_column and targets:
        folds = max(2, round(1 / valid_size))
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        try:
            train_idx, valid_idx = next(
                splitter.split(indices, _stratification_key(frame, targets), frame[group_column])
            )
        except ValueError:
            fallback = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
            train_idx, valid_idx = next(fallback.split(indices, groups=frame[group_column]))
    elif group_column:
        splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
        train_idx, valid_idx = next(splitter.split(indices, groups=frame[group_column]))
    else:
        stratify = _stratification_key(frame, targets) if targets else None
        try:
            train_idx, valid_idx = train_test_split(
                indices, test_size=valid_size, random_state=seed, stratify=stratify
            )
        except ValueError:
            train_idx, valid_idx = train_test_split(
                indices, test_size=valid_size, random_state=seed, stratify=None
            )

    if group_column:
        overlap = set(frame.iloc[train_idx][group_column]) & set(
            frame.iloc[valid_idx][group_column]
        )
        if overlap:
            raise RuntimeError(f"Group leakage detected for {len(overlap)} groups")
    return SplitResult(
        train=frame.iloc[train_idx].reset_index(drop=True),
        valid=frame.iloc[valid_idx].reset_index(drop=True),
        train_indices=np.asarray(train_idx),
        valid_indices=np.asarray(valid_idx),
    )


# >>> exercise: choose the legal split and retain exactly these folds
split = make_split(
    train,
    target_columns=None,
    group_column=GROUP_COLUMN,
    time_column=TIME_COLUMN,
    valid_size=0.2,
    seed=SEED,
)
tr, va = split.train.copy(), split.valid.copy()
# <<< exercise
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
if GROUP_COLUMN:
    assert not set(tr[GROUP_COLUMN]) & set(va[GROUP_COLUMN])
if TIME_COLUMN:
    assert pd.to_datetime(tr[TIME_COLUMN]).max() < pd.to_datetime(va[TIME_COLUMN]).min()
print("Frozen fold sizes:", len(tr), len(va))

# %% [markdown]
# ## 5. Audit imbalance and choose a response
#
# Measure prevalence on the training fold. Choose the response according to the official score:
# class weights, positive weights, coverage sampling or a log target are hypotheses. Compare
# with the unweighted candidate on the same split. Never rebalance validation.
#
# Long-tailed regression is not a class-balancing problem. The log transform aligns the
# objective with the task's score. Investigate tail residuals instead of assigning arbitrary
# class weights.
#

# %% [exercise]
quantiles = tr["SalePrice"].quantile([0, 0.1, 0.5, 0.9, 1])
print("Training target quantiles:", quantiles.to_dict())
# >>> exercise: transform the positive training target to match the official log-space loss
y_train = np.log(tr["SalePrice"].to_numpy())
# <<< exercise
print("Response: log target; do not oversample or trim validation tails")
np.log(tr["SalePrice"]).hist(bins=25)
plt.title("Training log-price distribution")
plt.tight_layout()
plt.savefig(OUTPUT / "imbalance.png")
plt.show()
plt.close()

# %% [markdown]
# ## 6. Build features or datasets
#
# Inspect one transformed example or batch. Explain its shape, range, label encoding and
# train/validation/test consistency. Fitted preprocessing sees the training fold only.
#
#
#

# %% [exercise]
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

feature_columns = [
    column
    for column in train
    if column not in {ID_COLUMN, *TARGET_COLUMNS, GROUP_COLUMN, TIME_COLUMN}
]
X_train, X_valid, X_test = tr[feature_columns], va[feature_columns], test[feature_columns]
numeric = X_train.select_dtypes("number").columns.tolist()
categorical = [column for column in feature_columns if column not in numeric]
# >>> exercise: keep fitted imputation and encoding inside the model pipeline
preprocessing = ColumnTransformer(
    [
        (
            "numeric",
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
            ),
            numeric,
        ),
        (
            "category",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical,
        ),
    ]
)
# <<< exercise
print("Numeric / categorical:", len(numeric), len(categorical))
print("Raw feature preview:", X_train.head(2).to_string(index=False))

# %% [markdown]
# ## 7. Define the model and loss
#
# State why the output dimension and loss fit this prediction contract. First verify the
# simplest model. Pretrained initialization is disabled unless the exact local asset is
# permitted.
#
#
#

# %% [exercise]
from sklearn.linear_model import Ridge

# >>> exercise: regress the transformed target with regularization
model = make_pipeline(preprocessing, Ridge(alpha=10, solver="lsqr"))
# <<< exercise
MODEL_DESCRIPTION = "one-hot Ridge on natural log prices"


def predict_scores(candidate, frame):
    return np.exp(candidate.predict(frame))


# %% [markdown]
# ## 8. Train and preserve the candidate
#
# First fit a tiny subset as a debugging drill; rebuild before the real-fold experiment. Main
# torch lessons define their small training loop directly below, including precision, time
# reserves and atomic best checkpoints. For sklearn fit the full preprocessing pipeline together.
#
#
#

# %% [exercise]
started = perf_counter()
budget.checkpoint("fit")
# >>> exercise: fit once on the training fold and save the entire fitted pipeline
model.fit(X_train, y_train)
checkpoint_path = save_pickle(model, OUTPUT / "pipeline.pkl")
# <<< exercise
fit_seconds = perf_counter() - started
validation_predictions = predict_scores(model, X_valid)
selected_threshold = 0.5
print("Fit seconds:", fit_seconds)

# %% [markdown]
# ## 9. Validate, inspect errors and compare cost
#
# Use all validation predictions for the exact score. Inspect individual mistakes and
# minority/group performance; the diagnostic metric does not replace the official metric. Keep a
# score/runtime ledger before choosing more complexity.
#
#
#

# %% [exercise]
from numpy.typing import ArrayLike


def log_price_rmse(truth: ArrayLike, prediction: ArrayLike) -> float:
    """House Prices RMSE of natural logarithms; unlike RMSLE, there is no added one."""
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not y.size or not np.isfinite([y, p]).all():
        raise ValueError("Expected aligned nonempty finite prices")
    if np.any(y <= 0) or np.any(p <= 0):
        raise ValueError("Log-price evaluation requires strictly positive prices")
    return float(np.sqrt(np.mean((np.log(y) - np.log(p)) ** 2)))


# >>> exercise: match RMSE(log true, log predicted) exactly
validation_score = log_price_rmse(va["SalePrice"], validation_predictions)
# <<< exercise
errors = va[[ID_COLUMN, "SalePrice"]].copy()
errors["prediction"] = validation_predictions
errors["log_residual"] = np.log(errors["prediction"]) - np.log(errors["SalePrice"])
print("Official log RMSE:", validation_score)
print(
    "Constant log-price reference:",
    log_price_rmse(va["SalePrice"], np.full(len(va), np.exp(np.log(tr["SalePrice"]).mean()))),
)
print(errors.loc[errors["log_residual"].abs().nlargest(15).index].to_string(index=False))
errors.to_csv(OUTPUT / "validation_errors.csv", index=False)
plt.scatter(np.log(va["SalePrice"]), errors["log_residual"], s=10)
plt.axhline(0, color="black")
plt.xlabel("True log price")
plt.ylabel("Log residual")
plt.savefig(OUTPUT / "validation.png")
plt.show()
plt.close()

# %% [markdown]
# ## 10. Reload and run complete inference
#
# Rebuild or reload the saved candidate before test inference. Preserve ID order through the
# loader, then align to the sample by ID. Check that reloaded validation predictions agree. This
# is inference recovery, not exact training continuation.
#
#
#

# %% [exercise]
started = perf_counter()
# >>> exercise: prove inference comes from the saved candidate, then predict every test row
restored = load_pickle(checkpoint_path)
np.testing.assert_allclose(predict_scores(restored, X_valid), validation_predictions)
test_predictions = predict_scores(restored, X_test)
# <<< exercise
assert np.isfinite(test_predictions).all()
assert (test_predictions > 0).all()
output_values = {TARGET_COLUMNS[0]: test_predictions}
save_json({"feature_columns": feature_columns, "model": str(model)}, OUTPUT / "recipe.json")
inference_seconds = perf_counter() - started

# %% [markdown]
# ## 11. Validate the submission and persist the recipe
#
# Validate the on-disk UTF-8 file, expected columns, ID order and task-specific value
# constraints. Keep source, preprocessing, fold IDs, labels, thresholds, score and runtime with
# the selected artifact. A checked CSV is not proof of upload acceptance.
#
#
#


# %% [exercise]
@dataclass(slots=True)
class SubmissionValidation:
    """Validation errors and non-fatal warnings for a candidate submission."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        """Raise one ``ValueError`` containing every validation error."""
        if self.errors:
            raise ValueError("Invalid submission: " + "; ".join(self.errors))


def build_submission(
    sample: pd.DataFrame,
    predictions: Mapping[str, Sequence[object]],
    *,
    prediction_ids: pd.DataFrame | Sequence[object] | None = None,
    id_columns: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Copy the sample and optionally align predictions using unique (possibly composite) IDs.

    Without IDs, predictions must already be in sample order. Prefer explicit IDs for inference.
    Alignment rejects missing, duplicate, extra, or absent keys rather than guessing row order.
    """
    submission = sample.copy()
    ids = [id_columns] if isinstance(id_columns, str) else list(id_columns or [])
    order = None
    if prediction_ids is not None:
        if not ids:
            raise ValueError("id_columns is required with prediction_ids")
        if isinstance(prediction_ids, pd.DataFrame):
            keys = prediction_ids[ids].reset_index(drop=True)
        elif len(ids) == 1:
            keys = pd.DataFrame({ids[0]: list(prediction_ids)})
        else:
            raise ValueError("Composite prediction IDs must be a DataFrame")
        for name, frame in (("sample", sample), ("predictions", keys)):
            if (
                frame[ids].isna().any().any()
                or frame[ids].eq("").any().any()
                or frame.duplicated(ids).any()
            ):
                raise ValueError(f"{name} IDs must be nonmissing and unique across {ids}")
        expected = pd.MultiIndex.from_frame(sample[ids])
        actual = pd.MultiIndex.from_frame(keys)
        order = actual.get_indexer(expected)
        if len(actual) != len(expected) or (order < 0).any():
            raise ValueError("Prediction IDs must match the sample IDs exactly")
    for column, values in predictions.items():
        if column not in submission.columns:
            raise KeyError(f"Prediction column not present in sample submission: {column}")
        if len(values) != len(submission):
            raise ValueError(
                f"Column {column} has {len(values)} values; expected {len(submission)}"
            )
        if column in ids:
            raise ValueError(f"Cannot overwrite ID column: {column}")
        values = list(values)
        submission[column] = values if order is None else [values[index] for index in order]
    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    id_columns: str | Iterable[str] | None = None,
    allowed_labels: Mapping[str, Iterable[object]] | None = None,
) -> SubmissionValidation:
    """Check schema, row count, ID order, missing values, and optional allowed labels."""
    errors: list[str] = []
    warnings: list[str] = []
    if submission.columns.duplicated().any() or sample.columns.duplicated().any():
        errors.append("Submission and sample column names must be unique")
        return SubmissionValidation(valid=False, errors=errors)
    if list(submission.columns) != list(sample.columns):
        errors.append(
            f"Columns/order differ: got {list(submission.columns)}, expected {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        errors.append(f"Row count differs: got {len(submission)}, expected {len(sample)}")
    if submission.isna().any().any():
        columns = submission.columns[submission.isna().any()].tolist()
        errors.append(f"Missing values in columns: {columns}")
    numeric = submission.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all():
        errors.append("Non-finite numeric values are not allowed")
    ids = [id_columns] if isinstance(id_columns, str) else list(id_columns or [])
    for column in ids:
        if column not in submission or column not in sample:
            errors.append(f"Missing ID column: {column}")
        elif submission[column].isna().any() or submission[column].eq("").any():
            errors.append(f"Missing ID values in column: {column}")
        elif len(submission) == len(sample) and not submission[column].equals(sample[column]):
            errors.append(f"ID values/order changed in column: {column}")
    if ids and all(column in submission for column in ids) and submission.duplicated(ids).any():
        warnings.append(f"Duplicate ID keys across columns: {ids}")
    for column, labels in (allowed_labels or {}).items():
        if column not in submission:
            errors.append(f"Missing allowed-label column: {column}")
        else:
            invalid = set(submission[column].dropna().unique()) - set(labels)
            if invalid:
                errors.append(f"Unexpected labels in {column}: {sorted(map(str, invalid))[:10]}")
    return SubmissionValidation(valid=not errors, errors=errors, warnings=warnings)


def write_submission(
    submission: pd.DataFrame,
    path: Path | str,
    *,
    sample: pd.DataFrame | None = None,
    id_columns: str | Iterable[str] | None = None,
    allowed_labels: Mapping[str, Iterable[object]] | None = None,
) -> Path:
    """Validate then atomically write a UTF-8 CSV submission."""
    if sample is not None:
        validation = validate_submission(
            submission, sample, id_columns=id_columns, allowed_labels=allowed_labels
        )
        validation.raise_for_errors()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        submission.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return value
    raise TypeError("Expected a dataclass or mapping")


def technical_report(
    *,
    task: str,
    model: str,
    data_summary: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    training_config: Any,
    validation_metrics: Mapping[str, float],
    augmentations: list[str] | None = None,
    checkpoint: str | None = None,
    notes: list[str] | None = None,
) -> str:
    """Render the standard competition handoff report as Markdown text."""
    config = _mapping(training_config)
    lines = [
        "# Technical Report",
        "",
        f"- Task: {task}",
        f"- Model: {model}",
        f"- Checkpoint: {checkpoint or 'not recorded'}",
        "",
        "## Data and validation",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in data_summary.items())
    lines.extend(f"- Split {key}: {value}" for key, value in split_summary.items())
    lines.extend(["", "## Training", ""])
    lines.extend(f"- {key}: {value}" for key, value in config.items())
    lines.extend(["", "## Validation metrics", ""])
    lines.extend(f"- {key}: {value:.6f}" for key, value in validation_metrics.items())
    lines.extend(["", "## Augmentations", ""])
    lines.extend(f"- {item}" for item in (augmentations or ["None"]))
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {item}" for item in notes)
    return "\n".join(lines) + "\n"


def write_technical_report(content: str, path: Path | str) -> Path:
    """Write a UTF-8 technical report, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def record_run(
    output_dir: Path | str,
    *,
    task: str,
    model: str,
    configuration: Any,
    validation_metrics: Mapping[str, float],
    split: SplitResult | None = None,
    data_summary: Mapping[str, Any] | None = None,
    inference_options: Mapping[str, Any] | None = None,
    checkpoint: str | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Save configuration/fold positions and a matching human-readable technical report."""
    root = Path(output_dir)
    split_summary = (
        {"train_rows": len(split.train), "valid_rows": len(split.valid)} if split else {}
    )
    payload = {
        "task": task,
        "model": model,
        "configuration": _mapping(configuration),
        "validation_metrics": dict(validation_metrics),
        "data_summary": dict(data_summary or {}),
        "inference_options": dict(inference_options or {}),
        "checkpoint": checkpoint,
        "notes": notes or [],
        "split": {
            **split_summary,
            "train_indices": split.train_indices if split else [],
            "valid_indices": split.valid_indices if split else [],
        },
    }
    save_json(payload, root / "run.json")
    report = technical_report(
        task=task,
        model=model,
        data_summary=data_summary or {},
        split_summary=split_summary,
        training_config=configuration,
        validation_metrics=validation_metrics,
        checkpoint=checkpoint,
        notes=[*(notes or []), f"Inference options: {dict(inference_options or {})}"],
    )
    return write_technical_report(report, root / "technical_report.md")


# >>> exercise: align predictions to sample IDs and validate the complete file
submission = build_submission(
    sample, output_values, prediction_ids=test[ID_COLUMN], id_columns=ID_COLUMN
)
validate_submission(submission, sample, id_columns=ID_COLUMN).raise_for_errors()
destination = write_submission(submission, OUTPUT / "submission.csv")
# <<< exercise
roundtrip = pd.read_csv(
    destination, converters={ID_COLUMN: str}, keep_default_na=False, float_precision="round_trip"
)
assert roundtrip[ID_COLUMN].tolist() == sample[ID_COLUMN].tolist()
assert roundtrip.columns.tolist() == sample.columns.tolist()
assert len(roundtrip) == len(sample)
pd.testing.assert_frame_equal(roundtrip.astype(str), submission.astype(str), check_dtype=False)
save_json(
    {"train_ids": tr[ID_COLUMN].tolist(), "validation_ids": va[ID_COLUMN].tolist()},
    OUTPUT / "fold_ids.json",
)
save_json(
    {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "max_rows": MAX_ROWS,
        "image_size": IMAGE_SIZE,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "imbalance_weights": USE_IMBALANCE_WEIGHTS,
        "history_features": USE_HISTORY_FEATURES,
        "pretrained_allowed": PRETRAINED_ALLOWED,
        "initial_weight_path": WEIGHT_PATH,
        "asset_provenance": ASSET_PROVENANCE,
        "trainer": globals().get("training"),
    },
    OUTPUT / "lesson_config.json",
)
record_run(
    OUTPUT,
    task=FAMILY,
    model=MODEL_DESCRIPTION,
    configuration=competition.to_dict(),
    validation_metrics={
        "official_score": float(validation_score),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    },
    split=split,
    inference_options={
        "threshold": float(selected_threshold),
        "label_order": TARGET_COLUMNS,
        "image_size": IMAGE_SIZE,
        "max_length": MAX_LENGTH,
        "model_name": MODEL_NAME,
        "asset_provenance": ASSET_PROVENANCE,
    },
    checkpoint=str(checkpoint_path),
    notes=[
        "Educational candidate, not a reproduced Kaggle leaderboard score.",
        "No test labels, test pseudo-labeling or unauthorized external data used.",
        f"Imbalance weights enabled: {USE_IMBALANCE_WEIGHTS}",
    ],
)
pd.DataFrame(
    [
        {
            "run": RUN_ID,
            "family": FAMILY,
            "variant": VARIANT,
            "score": validation_score,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
        }
    ]
).to_csv(OUTPUT / "comparison_row.csv", index=False)
# Save your edited notebook/source alongside this folder before transferring the bundle.
copied = sync_artifacts(OUTPUT, DURABLE_OUTPUT)
print("Checked local CSV:", destination, "Durable files copied:", len(copied))

# ruff: noqa: E402, F401
