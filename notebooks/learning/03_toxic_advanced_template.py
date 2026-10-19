# %% [markdown]
# # Jigsaw Toxic Comment Classification — advanced / SOTA study
#
# Problem family: **Multilabel text classification**. Official metric: **Mean of six label ROC
# AUC values**.
#
# **Learning objective:** Training-vocabulary two-layer BiGRU with optional permitted local
# embeddings.
#
# [Dataset and original
# task](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) ·
# [Primary solution
# evidence](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/toxic-crusaders-1st-place-solution-overview)
#
# The first-place write-up used pretrained embeddings, BiGRUs and additional
# augmentation/stacking. This lesson isolates a small BiGRU. It omits translations and test
# pseudo-labeling; randomly initialized embeddings do not reproduce its pretrained performance.
#
# **Watch for:** Six independent probabilities need sigmoid/BCE, not one softmax; a rare label
# absent in validation makes its AUC undefined.
#
# **Standalone:** this file contains every project-specific helper it uses. Only installed
# third-party libraries and your local data/assets are required; no repository checkout, source
# imports, project installation or external bootstrap is used. Only the optional dataset cell
# needs internet when enabled; all training works offline. Edit DATA_DIR and
# output/storage settings in stage 1. Train media on local disk, then persist your output bundle.
#
# This notebook is a learning exercise, not a claimed leaderboard reproduction. No trained
# weights or real Kaggle data are bundled. Keep grading answers closed. Use only explicitly
# permitted resources. Internet/prepared-code access is not assumed on contest day.
#
# Run the baseline first on the same data and folds. This worked notebook isolates a stronger
# method, explains its cost and limitations, and saves a separate candidate. Keep the baseline
# unless the measured gain justifies the extra runtime.
#
# After the lesson, close both notebooks and rebuild the essential route from memory. See the
# repository's blank-notebook playbook for the final transfer test.
#

# %% [markdown]
# ## 1. Bootstrap and configuration
#
# Run this supplied setup cell; edit only configuration values. Later algorithm cells contain
# TODO blocks. No weights are fetched. The next optional cell can acquire practice data;
# otherwise stage local data before repeated reads.
# Each candidate gets a new output directory.
#
#
#

# %%
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


@torch.inference_mode()
def predict_logits(model: nn.Module, loader: Iterable, *, device: str = "cpu") -> np.ndarray:
    """Ordered, deterministic inference; BF16/FP16 on capable CUDA, FP32 on CPU."""
    model.to(device).eval()
    enabled = torch.device(device).type == "cuda"
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    outputs = []
    for batch in loader:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        with torch.autocast(torch.device(device).type, dtype=dtype, enabled=enabled):
            logits = model(inputs.to(device))
        outputs.append(logits.float().cpu().numpy())
    if not outputs:
        raise ValueError("Inference loader is empty")
    return np.concatenate(outputs)


FAMILY, VARIANT = "03_toxic", "advanced"
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
GROUP_COLUMN, TIME_COLUMN = "_text_group", None
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
# ## 1a. Download the practice dataset (optional, before the timed exercise)
#
# The cell is skipped by default. For online preparation, first accept this competition's
# rules on its Kaggle page and configure credentials using the
# [official authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
# Set DOWNLOAD_DATA=True below. Local uv sync includes the official Kaggle client. In a copied
# notebook/Colab runtime without it, install only `kaggle>=2.2,<3` before running this cell;
# do not install this project or replace Colab's torch/NumPy/CUDA stack.
#
# Use KAGGLE_API_TOKEN in your environment, the client's configured token/legacy credential
# file, or enable PROMPT_FOR_KAGGLE_TOKEN for hidden input. Never paste a token into saved code.
# No rules are accepted automatically, no submission is uploaded, and no weights are fetched.
# Downloads can be several GB: allow space for the cached ZIP plus extracted data. DATA_DIR
# must be a dedicated empty folder (or a completed download from this cell). Existing manual
# data/fixtures are left untouched; keep DOWNLOAD_DATA=False to use them.
#
# This cell reuses a completed dataset/cache, expands nested train/test/CSV archives, and
# checks the expected layout. On Kaggle itself, attach competition data in the notebook UI,
# point DATA_DIR at its mounted folder, and leave this switch off. For restricted-room drills,
# prepare data beforehand and keep this cell disabled. The training timer restarts afterward.
#

# %%
import os
import subprocess
from getpass import getpass
from importlib.util import find_spec

DOWNLOAD_DATA = False
KAGGLE_COMPETITION = "jigsaw-toxic-comment-classification-challenge"
EXPECTED_DATA_PATHS = ("train.csv", "test.csv", "sample_submission.csv")
DOWNLOAD_CACHE = DATA_DIR.parent / "_downloads" / KAGGLE_COMPETITION
DOWNLOAD_TIMEOUT_SECONDS = 7200
PROMPT_FOR_KAGGLE_TOKEN = "google.colab" in sys.modules


def extract_dataset_zip(source: Path, destination: Path) -> None:
    """Extract data without allowing absolute paths, traversal or silent overwrites."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if sum(member.file_size for member in members) > shutil.disk_usage(destination).free:
            raise OSError("Insufficient disk space for the next archive")
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive contains a path outside the dataset folder")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Duplicated paths indicate a wrong archive/layout; preserve the first file.
            with archive.open(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def download_practice_data(
    competition: str,
    destination: Path,
    cache: Path,
    expected_paths: Sequence[str],
    *,
    prompt_token: bool = False,
    timeout_seconds: int = 7200,
) -> Path:
    """Use the official CLI, then publish the complete extracted folder by renaming it."""
    if not competition or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in competition
    ):
        raise ValueError("Expected a Kaggle competition slug, not a URL or shell command")
    destination, cache = Path(destination).resolve(), Path(cache).resolve()
    if (
        destination == cache
        or cache.is_relative_to(destination)
        or destination.is_relative_to(cache)
    ):
        raise ValueError("Dataset and download cache must be separate folders")
    marker = destination / ".kaggle-source.json"
    if marker.is_file():
        source = json.loads(marker.read_text(encoding="utf-8"))
        if source.get("competition") == competition and all(
            (destination / path).exists() for path in expected_paths
        ):
            print("Reusing completed dataset:", destination)
            return destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "DATA_DIR is not empty/completed; choose a new folder or keep DOWNLOAD_DATA=False"
        )
    cache.mkdir(parents=True, exist_ok=True)
    bundle = cache / f"{competition}.zip"
    if not bundle.is_file() or not zipfile.is_zipfile(bundle):
        if find_spec("kaggle") is None:
            raise ModuleNotFoundError(
                "Install kaggle>=2.2,<3 for online preparation; "
                "local users can run uv sync --locked"
            )
        if prompt_token and not os.environ.get("KAGGLE_API_TOKEN"):
            token = getpass("Kaggle API token (blank uses configured credentials): ").strip()
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
        # No shell, embedded credentials, auto-acceptance, uploads or package installation.
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            competition,
            "--path",
            str(cache),
            "--quiet",
        ]
        if bundle.exists():
            command.append("--force")  # A partial/invalid archive must be fetched again.
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode:
            # Do not put client output, possible signed URLs or credentials in saved notebooks.
            raise RuntimeError(
                f"Kaggle download failed (exit {result.returncode}). "
                "Check credentials, competition access/rules, internet and disk space. "
                "No dataset was published."
            )
        if not bundle.is_file() or not zipfile.is_zipfile(bundle):
            raise ValueError("Kaggle did not produce a complete competition ZIP; check the cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}-incoming-{uuid4().hex[:8]}")
    staging.mkdir()
    try:
        extract_dataset_zip(bundle, staging)
        # TGS has train.zip with images/ and masks/; some tasks also wrap CSVs in ZIPs.
        for nested in sorted(staging.rglob("*.zip")):
            folder = nested.parent
            if nested.stem in {"train", "test", "train_images", "test_images"}:
                with zipfile.ZipFile(nested) as archive:
                    names = [item.filename for item in archive.infolist() if not item.is_dir()]
                if names and not all(name.startswith(nested.stem + "/") for name in names):
                    folder = nested.parent / nested.stem
            extract_dataset_zip(nested, folder)
            nested.unlink()  # The outer cached bundle remains available for recovery.
        missing = [path for path in expected_paths if not (staging / path).exists()]
        if missing:
            raise FileNotFoundError(f"Unexpected dataset layout; missing: {missing}")
        save_json(
            {
                "competition": competition,
                "source": f"https://www.kaggle.com/competitions/{competition}/data",
                "expected_paths": list(expected_paths),
            },
            staging / marker.name,
        )
        if destination.exists():
            destination.rmdir()  # Only the previously checked empty directory.
        staging.replace(destination)
    except Exception as error:
        raise RuntimeError(
            "Dataset preparation failed; DATA_DIR was not replaced. "
            f"Inspect partial extraction at {staging}"
        ) from error
    print("Prepared dataset:", destination)
    return destination


if DOWNLOAD_DATA:
    if DATA_ARCHIVE is not None:
        raise ValueError("Choose either DATA_ARCHIVE staging or DOWNLOAD_DATA, then rerun setup")
    DATA_DIR = download_practice_data(
        KAGGLE_COMPETITION,
        DATA_DIR,
        DOWNLOAD_CACHE,
        EXPECTED_DATA_PATHS,
        prompt_token=PROMPT_FOR_KAGGLE_TOKEN,
        timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
    )
source_marker = DATA_DIR / ".kaggle-source.json"
if source_marker.is_file():
    save_json(json.loads(source_marker.read_text(encoding="utf-8")), OUTPUT / "data_source.json")
print("Download enabled:", DOWNLOAD_DATA, "Dataset folder:", DATA_DIR)
# Dataset acquisition is preparation time, not part of the measured training/delivery budget.
budget = TimeBudget(TIME_BUDGET_MINUTES, reserve_minutes=DELIVERY_RESERVE_MINUTES)

# %% [markdown]
# ## 2. Load data and enforce the contract
#
# Check schema and identity before fitting. A row is not always a prediction unit: detection has
# several annotation rows per image, and matching has several records per product.
#
#
#

# %%
ID_COLUMN = "id"
TARGET_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")
train["_text_group"] = train["comment_text"].str.strip().str.casefold()
assert train[TARGET_COLUMNS].isin([0, 1]).all().all()

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
# Inspect the longest texts and unusual character patterns. Preserve negation and useful
# punctuation. Normalized duplicate text is grouped before splitting.
#

# %%
lengths = train["comment_text"].str.len()
print("Text length quantiles:", lengths.quantile([0, 0.5, 0.9, 0.99, 1]).to_dict())
print("Repeated normalized texts:", train["_text_group"].duplicated().sum())
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
lengths.hist(bins=40, ax=axes[0])
axes[0].set_title("Character length")
train[TARGET_COLUMNS].mean().plot.bar(ax=axes[1], title="Positive prevalence")
axes[2].imshow(train[TARGET_COLUMNS].T.dot(train[TARGET_COLUMNS]), cmap="Blues")
axes[2].set_title("Label co-occurrence")
axes[2].set_xticks(range(6), TARGET_COLUMNS, rotation=90)
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

# %%
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


# >>> exercise: hold duplicate normalized comments on the same side of validation
split = make_split(
    train, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, seed=SEED, valid_size=0.2
)
tr, va = split.train.copy(), split.valid.copy()
# <<< exercise
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
if GROUP_COLUMN:
    assert not set(tr[GROUP_COLUMN]) & set(va[GROUP_COLUMN])
if TIME_COLUMN:
    assert pd.to_datetime(tr[TIME_COLUMN]).max() < pd.to_datetime(va[TIME_COLUMN]).min()
print("Fold sizes:", len(tr), len(va))

# %% [markdown]
# ## 5. Audit imbalance and choose a response
#
# Measure prevalence on the training fold. Choose the response according to the official score:
# class weights, positive weights, coverage sampling or a log target are hypotheses. Compare
# with the unweighted candidate on the same split. Never rebalance validation.
#
# This split is group-aware, not a multilabel stratifier. Each of the six labels must have
# positive and negative cases in both folds. If support fails, use a larger pack or a justified
# group-compatible multilabel splitter; do not silently drop a label from the score.
#

# %%
counts = tr[TARGET_COLUMNS].sum()
print("Training positives:", counts.to_dict())
for name, frame in (("train", tr), ("validation", va)):
    if not frame[TARGET_COLUMNS].nunique().eq(2).all():
        raise ValueError(
            f"{name} lacks both outcomes for a label; AUC is undefined. Inspect split/support."
        )
# >>> exercise: derive positive weights from the training fold only
ratios = (len(tr) - counts) / counts
positive_weights = torch.tensor(ratios.to_numpy(), dtype=torch.float32)
class_weight = "balanced" if USE_IMBALANCE_WEIGHTS else None
# <<< exercise
print(
    "Response:", "training-fold weights" if USE_IMBALANCE_WEIGHTS else "unweighted first candidate"
)
counts.plot.bar(title="Training positive counts: six independent labels")
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

# %%
from collections import Counter

from torch.utils.data import TensorDataset

# >>> exercise: build the vocabulary only from training-fold text
counts_words = Counter(word for text in tr["comment_text"] for word in text.lower().split())
vocabulary = {"<pad>": 0, "<unk>": 1}
vocabulary.update({word: i + 2 for i, (word, _) in enumerate(counts_words.most_common(20000))})


# <<< exercise
def encode(texts):
    rows = []
    for text in texts:
        tokens = [vocabulary.get(word, 1) for word in text.lower().split()[:MAX_LENGTH]] or [1]
        rows.append(tokens + [0] * (MAX_LENGTH - len(tokens)))
    return torch.tensor(rows, dtype=torch.long)


def loader_for(frame, training=False):
    targets = (
        frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        if set(TARGET_COLUMNS) <= set(frame)
        else np.zeros((len(frame), 6), dtype=np.float32)
    )
    dataset = TensorDataset(encode(frame["comment_text"]), torch.from_numpy(targets))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training, num_workers=0)


train_loader, valid_loader = loader_for(tr, True), loader_for(va)
save_json(vocabulary, OUTPUT / "vocabulary.json")
print("Token tensor:", encode(tr["comment_text"].head(2)).shape)

# %% [markdown]
# ## 7. Define the model and loss
#
# State why the output dimension and loss fit this prediction contract. First verify the
# simplest model. Pretrained initialization is disabled unless the exact local asset is
# permitted.
#
# Study the historical winner's embeddings/BiGRU idea. The default has random embeddings and a
# small hidden state to run offline; that is not evidence of leaderboard-level performance. An
# optional local NPZ has tokens [V] and vectors [V,EMBEDDING_DIM], loaded only with explicit
# permission/provenance.
#

# %%
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ToxicGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocabulary), EMBEDDING_DIM, padding_idx=0)
        self.encoder = nn.GRU(
            EMBEDDING_DIM, 32, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1
        )
        self.head = nn.Linear(128, len(TARGET_COLUMNS))

    def forward(self, tokens):
        mask = tokens.ne(0)
        packed = pack_padded_sequence(
            self.embedding(tokens), mask.sum(1).cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        values, _ = pad_packed_sequence(encoded, batch_first=True, total_length=tokens.shape[1])
        mean = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        maximum = values.masked_fill(~mask.unsqueeze(-1), -1e4).amax(1)
        return self.head(torch.cat([mean, maximum], dim=1))


model = ToxicGRU()
if EMBEDDING_PATH is not None:
    if not PRETRAINED_ALLOWED or not ASSET_PROVENANCE:
        raise ValueError("Explicit permission/provenance required for these exact local embeddings")
    asset = np.load(EMBEDDING_PATH, allow_pickle=False)
    if asset["vectors"].shape != (len(asset["tokens"]), EMBEDDING_DIM):
        raise ValueError("Embedding dimensions do not match configuration")
    lookup = dict(zip(asset["tokens"].tolist(), asset["vectors"], strict=True))
    with torch.no_grad():
        for word, index in vocabulary.items():
            if word in lookup and index > 1:
                model.embedding.weight[index].copy_(torch.as_tensor(lookup[word]))
    print(
        "Local embedding vocabulary coverage:",
        sum(word in lookup for word in vocabulary) / len(vocabulary),
    )
# >>> exercise: choose independent BCE logits loss and training-only positive weights
criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights if USE_IMBALANCE_WEIGHTS else None).to(
    competition.device
)
# <<< exercise
MODEL_DESCRIPTION = "two-layer bidirectional GRU; local embeddings only when configured"

# %% [markdown]
# ## 8. Train and preserve the candidate
#
# First fit a tiny subset as a debugging drill; rebuild before the real-fold experiment. Main
# torch lessons define their small training loop directly below, including precision, time
# reserves and atomic best checkpoints. For sklearn fit the full preprocessing pipeline together.
#
#
#

# %%


@dataclass
class TrainerConfig:
    """Small lesson loop: AdamW, one step per batch, no scheduler."""

    epochs: int = 3
    learning_rate: float = 1e-3
    metric_name: str = "accuracy"
    mixed_precision: str = "auto"
    scheduler: str = "none"
    patience: int = 2


class Trainer:
    """Visible training loop: validate globally, keep the best, reload before returning."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: TrainerConfig,
        *,
        output_dir: Path,
        device: str,
        metric_fn: Callable | None = None,
        prediction_decoder: Callable | None = None,
        time_budget: TimeBudget | None = None,
        backup_dir: Path | None = None,
    ) -> None:
        if config.scheduler != "none" or config.mixed_precision not in {"auto", "none"}:
            raise ValueError("This minimal loop supports scheduler=none and AMP auto/none")
        if config.epochs < 1 or config.learning_rate <= 0 or config.patience < 1:
            raise ValueError("Epochs, learning rate and patience must be positive")
        self.model, self.criterion = model.to(device), criterion.to(device)
        self.config, self.device = config, device
        self.metric_fn, self.decoder = metric_fn, prediction_decoder
        self.budget, self.backup_dir = time_budget, backup_dir
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / "best.pt"
        self.amp = torch.device(device).type == "cuda" and config.mixed_precision == "auto"
        self.dtype = (
            torch.bfloat16 if self.amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.dtype == torch.float16)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-2
        )

    @torch.inference_mode()
    def evaluate(self, loader: Iterable) -> dict[str, float]:
        self.model.eval()
        truths, predictions, total, count = [], [], 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.autocast(torch.device(self.device).type, dtype=self.dtype, enabled=self.amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            decoded = self.decoder(logits.float()) if self.decoder else logits.argmax(1)
            truths.append(targets.cpu().numpy())
            predictions.append(decoded.cpu().numpy())
            total += float(loss) * len(targets)
            count += len(targets)
        if not count:
            raise ValueError("Empty validation fold")
        # Salt overrides this method to select by the exact original-resolution grader.
        score = (
            self.metric_fn(np.concatenate(truths), np.concatenate(predictions))
            if self.metric_fn
            else -total / count
        )
        return {"loss": total / count, "metric": float(score)}

    def fit(self, train_loader: Iterable, valid_loader: Iterable) -> list[dict]:
        history, best, stale = [], -float("inf"), 0
        for epoch in range(self.config.epochs):
            if self.budget and self.budget.should_stop:
                break
            self.model.train()
            total, count = 0.0, 0
            for inputs, targets in train_loader:
                if self.budget and self.budget.should_stop:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    torch.device(self.device).type, dtype=self.dtype, enabled=self.amp
                ):
                    loss = self.criterion(self.model(inputs), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total += float(loss.detach()) * len(targets)
                count += len(targets)
            if not count or (self.budget and self.budget.should_stop):
                break
            values = self.evaluate(valid_loader)
            if not np.isfinite(values["metric"]):
                raise ValueError("Nonfinite selection metric")
            history.append({"epoch": epoch, "train_loss": total / count, **values})
            print(history[-1])
            with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(history[-1], allow_nan=False) + "\n")
            if values["metric"] > best:
                best, stale = values["metric"], 0
                save_torch_state(
                    {"model": self.model.state_dict(), "epoch": epoch, "score": best},
                    self.checkpoint_path,
                )
                sync_artifacts(self.output_dir, self.backup_dir)
            else:
                stale += 1
            if stale >= self.config.patience:
                break
        if not self.checkpoint_path.is_file():
            raise TimeoutError("No complete validation/checkpoint; shorten the debug candidate")
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)["model"]
        )
        return history


started = perf_counter()
training = TrainerConfig(
    epochs=EPOCHS,
    learning_rate=1e-3,
    metric_name="roc_auc",
    mixed_precision="auto",
    scheduler="none",
    patience=2,
)
# >>> exercise: decode every logit independently and select using global multilabel AUROC
trainer = Trainer(
    model,
    criterion,
    training,
    output_dir=OUTPUT,
    device=competition.device,
    prediction_decoder=torch.sigmoid,
    metric_fn=lambda y, p: evaluate_metric("roc_auc", y, p, average="macro"),
    time_budget=budget,
    backup_dir=DURABLE_OUTPUT,
)
history = trainer.fit(train_loader, valid_loader)
# <<< exercise
checkpoint_path = trainer.checkpoint_path
fit_seconds = perf_counter() - started
validation_predictions = (
    torch.from_numpy(predict_logits(model, valid_loader, device=competition.device))
    .sigmoid()
    .numpy()
)
selected_threshold = 0.5

# %% [markdown]
# ## 9. Validate, inspect errors and compare cost
#
# Use all validation predictions for the exact score. Inspect individual mistakes and
# minority/group performance; the diagnostic metric does not replace the official metric. Keep a
# score/runtime ledger before choosing more complexity.
#
# AUROC measures ranking within each label. A .5 threshold is only an optional diagnostic
# decision boundary. Label-wise calibration or thresholds cannot change the required probability
# submission into hard labels.
#

# %%
# >>> exercise: average six independently computed AUC values, never threshold for AUROC
per_label_auc = {
    label: evaluate_metric("roc_auc", va[label], validation_predictions[:, index])
    for index, label in enumerate(TARGET_COLUMNS)
}
validation_score = float(np.mean(list(per_label_auc.values())))
# <<< exercise
print("Official mean AUC:", validation_score, "Per label:", per_label_auc)
errors = va[[ID_COLUMN, *TARGET_COLUMNS]].copy()
errors["mean_absolute_error"] = abs(va[TARGET_COLUMNS].to_numpy() - validation_predictions).mean(1)
print(errors.nlargest(15, "mean_absolute_error").to_string(index=False))
errors.to_csv(OUTPUT / "validation_errors.csv", index=False)
pd.Series(per_label_auc).plot.bar(title="Validation AUC by label")
plt.ylim(0, 1)
plt.tight_layout()
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

# %%

started = perf_counter()
# >>> exercise: rebuild from the saved vocabulary and own best checkpoint
vocabulary = json.loads((OUTPUT / "vocabulary.json").read_text(encoding="utf-8"))
restored = ToxicGRU().to(competition.device)
restored.load_state_dict(
    torch.load(checkpoint_path, map_location=competition.device, weights_only=True)["model"],
    strict=True,
)
reloaded = (
    torch.from_numpy(predict_logits(restored, loader_for(va), device=competition.device))
    .sigmoid()
    .numpy()
)
np.testing.assert_allclose(reloaded, validation_predictions, rtol=1e-5, atol=1e-6)
test_predictions = (
    torch.from_numpy(predict_logits(restored, loader_for(test), device=competition.device))
    .sigmoid()
    .numpy()
)
# <<< exercise
assert test_predictions.shape == (len(test), 6)
assert (
    np.isfinite(test_predictions).all()
    and ((0 <= test_predictions) & (test_predictions <= 1)).all()
)
output_values = {label: test_predictions[:, i] for i, label in enumerate(TARGET_COLUMNS)}
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


# %%
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

# ruff: noqa: E402
