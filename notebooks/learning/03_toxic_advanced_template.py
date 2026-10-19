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
# ## How to use this lesson
#
# Read each worked component, predict its output, run its checks, then explain the result. Keep
# the baseline folds and compare one change at a time. This is a worked advanced study, not a
# claimed leaderboard reproduction.
#
# By the end you should be able to state the data/metric contract, choose a legal split, construct
# one correct model input, validate a saved candidate and deliver an ID-aligned file. Mini-checks
# use tiny separate examples or the current training batch; they do not certify real-data quality.
#
#  | Stage | Goal |
# |---|---|
# | 1 | Environment and configuration |
# | 1a | Optional dataset acquisition |
# | 2 | Understand and load the data contract |
# | 3 | Inspect examples and distributions |
# | 4 | Build and test the validation split |
# | 5 | Measure imbalance and choose a response |
# | 6 | Build features or one correct batch |
# | 7 | Connect outputs, loss and official metric |
# | 8 | Train and preserve a recoverable candidate |
# | 9 | Interpret validation and compare experiments |
# | 10 | Reload and infer every test item |
# | 11 | Deliver and verify the submission |

# %% [markdown]
# ## 1. Environment and configuration
#
# Run the supplied imports and runtime helpers in order. Edit paths, time budget and model policy
# in the configuration cells. Inspect the actual device; CPU checks cannot establish GPU
# throughput. Supplied storage/safety definitions are folded where the viewer supports it: run
# them, but reconstruct the short algorithmic functions during closed-book practice.

# %% [provided] [stage-1]
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

# ruff: noqa: E402

# %% [markdown]
# ### Supplied helper — `CompetitionConfig`
#
# Explicit notebook state; no installed project is required.


# %% [provided] [stage-1]
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


# %% [markdown]
# ### Supplied helper — `TimeBudget`
#
# Leave delivery time available; a running batch/fit cannot be interrupted safely.


# %% [provided] [stage-1]
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


# %% [markdown]
# ### Supplied helper — `runtime_paths`
#
# Media stays on local disk; optional durable storage is an explicit destination.


# %% [provided] [stage-1]
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


# %% [markdown]
# ### Supplied helper — `mount_google_drive`
#
# Call only when an explicitly configured Colab path uses Drive.


# %% [provided] [stage-1]
def mount_google_drive() -> None:
    """Call only when an explicitly configured Colab path uses Drive."""
    from google.colab import drive

    drive.mount("/content/drive")


# %% [markdown]
# ### Supplied helper — `stage_data`
#
# Copy a folder or extract a ZIP once, then train from the local destination.


# %% [provided] [stage-1]
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


# %% [markdown]
# ### Supplied helper — `gpu_report`
#
# Inspect the actual runtime before choosing image size, batch size or precision.


# %% [provided] [stage-1]
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


# %% [markdown]
# ### Configure this run
#
# Edit these values before proceeding. DATA_DIR points at your staged pack; choose a separate
# empty folder for an online download. Keep model permission explicit, start with a small row/time
# budget, and let the actual device report guide neural batch size. Each run writes to a new
# candidate directory.

# %% [configuration] [stage-1]
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
PRETRAINED_ALLOWED = False
SEED, MAX_ROWS, BATCH_SIZE, EPOCHS = 42, 6000, 16, 3
TIME_BUDGET_MINUTES, DELIVERY_RESERVE_MINUTES = 60, 10
GROUP_COLUMN, TIME_COLUMN = "_text_group", None
USE_IMBALANCE_WEIGHTS = False
MODEL_NAME = "bigru"

# %% [markdown]
# ### Configure this run
#
# Edit these values before proceeding. DATA_DIR points at your staged pack; choose a separate
# empty folder for an online download. Keep model permission explicit, start with a small row/time
# budget, and let the actual device report guide neural batch size. Each run writes to a new
# candidate directory.

# %% [configuration] [stage-1]
ASSET_PROVENANCE = (
    ""  # exact asset/version/license; required when enabling local pretrained features
)
EMBEDDING_PATH, EMBEDDING_DIM, MAX_LENGTH = None, 64, 128
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
# ## 1a. Optional dataset acquisition
#
# The cell is skipped by default. For online preparation, first accept this competition's
# rules on its Kaggle page and configure credentials using the
# [official authentication
# guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
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

# %% [markdown]
# ### Supplied helper — `_json_default`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [provided] [stage-1a]
def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


# %% [markdown]
# ### Supplied helper — `_write`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [provided] [stage-1a]
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


# %% [markdown]
# ### Supplied helper — `save_json`
#
# Atomically save UTF-8 metadata, including paths, dataclasses, and NumPy values.


# %% [provided] [stage-1a]
def save_json(value: Any, path: Path | str) -> Path:
    """Atomically save UTF-8 metadata, including paths, dataclasses, and NumPy values."""
    return _write(value, path, "json")


# %% [configuration] [stage-1a]
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

# %% [markdown]
# ### Supplied helper — `extract_dataset_zip`
#
# Extract data without allowing absolute paths, traversal or silent overwrites.


# %% [provided] [stage-1a]
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


# %% [markdown]
# ### Supplied helper — `download_practice_data`
#
# Use the official CLI, then publish the complete extracted folder by renaming it.


# %% [provided] [stage-1a]
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


# %% [provided] [stage-1a]
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
budget = TimeBudget(TIME_BUDGET_MINUTES, reserve_minutes=DELIVERY_RESERVE_MINUTES)

# %% [markdown]
# ## 2. Understand and load the data contract
#
# Each comment can have several of six independent labels. Return six probabilities per id in the
# sample's exact label order. A row may have no positive label or several; the labels are not
# mutually exclusive classes.
#
# Read the first rows, column names and actual shapes below. Expected: unique nonempty prediction
# IDs; matching test/sample ID sets; a target schema that agrees with the official output. Dataset
# size depends on your local files and MAX_ROWS.

# %% [worked] [stage-2]
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

# %% [check] [stage-2]
assert len(train) > 0 and len(test) == len(sample)
assert train[ID_COLUMN].is_unique and sample[ID_COLUMN].is_unique
print("Data-contract checks passed: prediction units and sample coverage.")

# %% [markdown]
# ### Stop and explain
#
# What is the prediction unit, and which columns may enter the model?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Each comment can have several of six independent labels. Return six probabilities per id in the
# sample's exact label order. A row may have no positive label or several; the labels are not
# mutually exclusive classes.
#
# </details>

# %% [markdown]
# ## 3. Inspect examples and distributions
#
# Inspect the plotted examples before selecting a model. Write observations before reading the
# next section.
#
# Which labels are rarest? How often do labels co-occur? Are empty or normalized duplicate
# comments present? Do long comments get truncated by the proposed tokenizer?

# %% [worked] [stage-3]
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
# ### Stop and explain
#
# Record three observations: one distribution issue, one suspicious example, and one possible
# leakage source. What experiment does each justify?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Use the actual plots and rows; there is no fixed answer independent of your pack. Convert each
# observation into a falsifiable check before spending training time.
#
# </details>

# %% [markdown]
# ## 4. Build and test the validation split
#
# Keep normalized duplicate comments together. A group split is not automatically a multilabel
# stratifier: explicitly check positive and negative support for every label in both folds.
# Undefined AUROC must not be silently dropped.

# %% [worked] [stage-4]
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

# %% [markdown]
# ### Supplied helper — `SplitResult`
#
# Training/validation tables plus their original positional indexes.


# %% [provided] [stage-4]
@dataclass(slots=True)
class SplitResult:
    """Training/validation tables plus their original positional indexes."""

    train: pd.DataFrame
    valid: pd.DataFrame
    train_indices: np.ndarray
    valid_indices: np.ndarray


# %% [markdown]
# ### Supplied helper — `_stratification_key`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [provided] [stage-4]
def _stratification_key(frame: pd.DataFrame, targets: Sequence[str]) -> pd.Series:
    if not targets:
        raise ValueError("At least one target is required for stratification")
    if len(targets) == 1:
        return frame[targets[0]].astype(str)
    return frame[list(targets)].astype(str).agg("|".join, axis=1)


# %% [markdown]
# ### Supplied helper — `make_split`
#
# Create a deterministic holdout with leakage-aware group and time options.


# %% [provided] [stage-4]
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


# %% [markdown]
# ### Worked component — `create_holdout`
#
# Input: a row-per-prediction table and its actual leakage columns. Return: the split record and
# independent training/validation copies. The union must retain all rows; identities must not
# cross folds.


# %% [worked] [stage-4]
def create_holdout(
    frame: pd.DataFrame,
    target_columns: Any = None,
    group_column: str | None = None,
    time_column: str | None = None,
    seed: int = 42,
) -> Any:
    """Input: a row-per-prediction table and its actual leakage columns. Return: the split
    record and independent training/validation copies. The union must retain all rows;
    identities must not cross folds.
    """
    split = make_split(
        frame,
        target_columns=target_columns,
        group_column=group_column,
        time_column=time_column,
        valid_size=0.2,
        seed=seed,
    )
    tr, va = split.train.copy(), split.valid.copy()
    return split, tr, va


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-4]
toy_frame = pd.DataFrame({"id": range(12), "group": np.repeat(range(6), 2)})
toy_split, toy_train, toy_valid = create_holdout(toy_frame, group_column="group", seed=7)
assert len(toy_train) + len(toy_valid) == 12, "The split must retain every row"
assert not set(toy_train["group"]) & set(toy_valid["group"]), "Repeated identities leaked"
assert not set(toy_split.train_indices) & set(toy_split.valid_indices)
toy_again = create_holdout(toy_frame, group_column="group", seed=7)[0]
np.testing.assert_array_equal(toy_split.valid_indices, toy_again.valid_indices)
print("Split checks passed: complete, group-disjoint and reproducible.")

# %% [worked] [stage-4]
split, tr, va = create_holdout(
    train, target_columns=None, group_column=GROUP_COLUMN, time_column=TIME_COLUMN, seed=SEED
)
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
if GROUP_COLUMN:
    assert not set(tr[GROUP_COLUMN]) & set(va[GROUP_COLUMN])
if TIME_COLUMN:
    assert pd.to_datetime(tr[TIME_COLUMN]).max() < pd.to_datetime(va[TIME_COLUMN]).min()
print("Fold sizes:", len(tr), len(va))

# %% [check] [stage-4]
assert len(tr) > 0 and len(va) > 0
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
print("Actual-fold checks passed: nonempty disjoint prediction IDs.")

# %% [markdown]
# ### Stop and explain
#
# Would a random row split measure the generalization you need here? Identify the boundary that
# would leak information.
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Keep normalized duplicate comments together. A group split is not automatically a multilabel
# stratifier: explicitly check positive and negative support for every label in both folds.
# Undefined AUROC must not be silently dropped.
#
# </details>

# %% [markdown]
# ## 5. Measure imbalance and choose a response
#
# For label c, compute (N-positive_c)/positive_c from training comments. These independent ratios
# can weight BCE or motivate balanced binary classifiers. Start unweighted and never oversample or
# rebalance validation.
#
# $w_c=(N-n_c)/n_c$; $\mathrm{score}=\frac{1}{6}\sum_c\mathrm{AUROC}(y_c,p_c)$. Independent labels
# require sigmoid/BCE, not a six-way softmax.

# %% [worked] [stage-5]
counts = tr[TARGET_COLUMNS].sum()
print("Training positives:", counts.to_dict())
for name, frame in (("train", tr), ("validation", va)):
    if not frame[TARGET_COLUMNS].nunique().eq(2).all():
        raise ValueError(
            f"{name} lacks both outcomes for a label; AUC is undefined. Inspect split/support."
        )

# %% [markdown]
# ### Worked component — `multilabel_imbalance`
#
# Input: positive training counts in target-column order, row count and weighting flag. Return:
# per-label ratios, float32 BCE positive weights [C] and sklearn class_weight.


# %% [worked] [stage-5]
def multilabel_imbalance(counts: pd.Series, training_rows: int, use_weights: bool) -> Any:
    """Input: positive training counts in target-column order, row count and weighting flag.
    Return: per-label ratios, float32 BCE positive weights [C] and sklearn class_weight.
    """
    ratios = (training_rows - counts) / counts
    positive_weights = torch.tensor(ratios.to_numpy(), dtype=torch.float32)
    class_weight = "balanced" if use_weights else None
    return ratios, positive_weights, class_weight


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-5]
toy_counts = pd.Series([2, 4], index=["rare", "common"])
toy_ratios, toy_weights, toy_class_weight = multilabel_imbalance(toy_counts, 8, False)
np.testing.assert_allclose(toy_ratios, [3, 1])
np.testing.assert_allclose(toy_weights.numpy(), [3, 1])
assert toy_class_weight is None and toy_weights.shape == (2,)
print("Multilabel checks passed: independent positive weights remain in target-column order.")

# %% [worked] [stage-5]
ratios, positive_weights, class_weight = multilabel_imbalance(
    counts, len(tr), USE_IMBALANCE_WEIGHTS
)
print(
    "Response:", "training-fold weights" if USE_IMBALANCE_WEIGHTS else "unweighted first candidate"
)
counts.plot.bar(title="Training positive counts: six independent labels")
plt.tight_layout()
plt.savefig(OUTPUT / "imbalance.png")
plt.show()
plt.close()

# %% [markdown]
# ### Stop and explain
#
# Does changing training weights also change the official score or validation distribution? What
# would justify enabling the response?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# For label c, compute (N-positive_c)/positive_c from training comments. These independent ratios
# can weight BCE or motivate balanced binary classifiers. Start unweighted and never oversample or
# rebalance validation.
#
# </details>

# %% [markdown]
# ## 6. Build features or one correct batch
#
# The BiGRU route builds a training-only vocabulary, pads/truncates token sequences and optionally
# loads explicitly permitted local embeddings. Inspect integer IDs, padding, sequence lengths and
# six-target batch alignment.

# %% [markdown]
# ### Worked component — `predict_logits`
#
# Ordered, deterministic inference; BF16/FP16 on capable CUDA, FP32 on CPU.


# %% [worked] [stage-6]
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


# %% [worked] [stage-6]
from collections import Counter

from torch.utils.data import TensorDataset

counts_words = Counter(word for text in tr["comment_text"] for word in text.lower().split())
vocabulary = {"<pad>": 0, "<unk>": 1}
vocabulary.update({word: i + 2 for i, (word, _) in enumerate(counts_words.most_common(20000))})

# %% [markdown]
# ### Worked component — `encode`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-6]
def encode(texts):
    rows = []
    for text in texts:
        tokens = [vocabulary.get(word, 1) for word in text.lower().split()[:MAX_LENGTH]] or [1]
        rows.append(tokens + [0] * (MAX_LENGTH - len(tokens)))
    return torch.tensor(rows, dtype=torch.long)


# %% [markdown]
# ### Worked component — `loader_for`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-6]
def loader_for(frame, training=False):
    targets = (
        frame[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        if set(TARGET_COLUMNS) <= set(frame)
        else np.zeros((len(frame), 6), dtype=np.float32)
    )
    dataset = TensorDataset(encode(frame["comment_text"]), torch.from_numpy(targets))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training, num_workers=0)


# %% [worked] [stage-6]
train_loader, valid_loader = loader_for(tr, True), loader_for(va)
save_json(vocabulary, OUTPUT / "vocabulary.json")
print("Token tensor:", encode(tr["comment_text"].head(2)).shape)

# %% [markdown]
# ### Stop and explain
#
# State the feature/batch shapes and the exact point at which preprocessing is fitted. What
# happens to a previously unseen validation input?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# The BiGRU route builds a training-only vocabulary, pads/truncates token sequences and optionally
# loads explicitly permitted local embeddings. Inspect integer IDs, padding, sequence lengths and
# six-target batch alignment.
#
# </details>

# %% [markdown]
# ## 7. Connect outputs, loss and official metric
#
# The two-layer bidirectional GRU encodes word order and emits six raw logits. BCEWithLogitsLoss
# applies independent sigmoid objectives. Pretrained embeddings are optional legal local assets,
# not an automatic download.

# %% [markdown]
# ### Supplied helper — `save_torch_state`
#
# Atomically save torch state without importing torch until this function is used.


# %% [provided] [stage-7]
def save_torch_state(value: Any, path: Path | str) -> Path:
    """Atomically save torch state without importing torch until this function is used."""
    return _write(value, path, "torch")


# %% [markdown]
# ### Supplied helper — `sync_artifacts`
#
# Copy the entire candidate, including encoders, preserving separate run folders.


# %% [provided] [stage-7]
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


# %% [markdown]
# ### Worked component — `evaluate_metric`
#
# Only the metrics used in this lesson; task-specific graders appear beside the model.


# %% [worked] [stage-7]
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


# %% [worked] [stage-7]
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# %% [markdown]
# ### Worked component — `ToxicGRU`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-7]
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


# %% [worked] [stage-7]
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
criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights if USE_IMBALANCE_WEIGHTS else None).to(
    competition.device
)
MODEL_DESCRIPTION = "two-layer bidirectional GRU; local embeddings only when configured"

# %% [markdown]
# ### Worked component — `score_multilabel`
#
# Input: label table, [N,C] probabilities and ordered target names. Return all label AUCs and
# their unweighted mean. Never silently omit a label or threshold its scores.


# %% [worked] [stage-7]
def score_multilabel(truth: pd.DataFrame, predictions: np.ndarray, targets: Sequence[str]) -> Any:
    """Input: label table, [N,C] probabilities and ordered target names. Return all label AUCs
    and their unweighted mean. Never silently omit a label or threshold its scores.
    """
    per_label_auc = {
        label: evaluate_metric("roc_auc", truth[label], predictions[:, index])
        for index, label in enumerate(targets)
    }
    validation_score = float(np.mean(list(per_label_auc.values())))
    return per_label_auc, validation_score


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-7]
toy_truth = pd.DataFrame({"first": [0, 0, 1, 1], "second": [0, 0, 1, 1]})
toy_predictions = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
toy_per_label, toy_score = score_multilabel(toy_truth, toy_predictions, ["first", "second"])
assert toy_per_label == {"first": 1.0, "second": 0.0} and toy_score == 0.5
print("Multilabel AUROC checks passed: each label is scored before unweighted averaging.")

# %% [markdown]
# **Actual model/batch compatibility.** Run an evaluation forward pass before full training.
# Expected: finite logits with the stated batch/output dimensions. Restore the model's previous
# mode afterward.

# %% [check] [stage-7]
was_training = model.training
model.eval()
with torch.no_grad():
    example_inputs, example_targets = next(iter(valid_loader))
    example_inputs = example_inputs[:2].to(next(model.parameters()).device)
    example_logits = model(example_inputs)
    assert tuple(example_logits.shape) == (len(example_inputs), len(TARGET_COLUMNS)), (
        "Model outputs do not match the task contract"
    )
    assert torch.isfinite(example_logits).all()
model.train(was_training)
print("Actual batch/model compatibility passed:", tuple(example_logits.shape))

# %% [markdown]
# ### Stop and explain
#
# Explain the output shape, loss and prediction decoding without reading the implementation. Which
# convention would silently produce an invalid score?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# The two-layer bidirectional GRU encodes word order and emits six raw logits. BCEWithLogitsLoss
# applies independent sigmoid objectives. Pretrained embeddings are optional legal local assets,
# not an automatic download. Compute AUROC separately for every target and take their unweighted
# mean. Inspect per-label failures and rare-label support. A strong common label cannot substitute
# for a missing rare-label score.
#
# </details>

# %% [markdown]
# ## 8. Train and preserve a recoverable candidate
#
# Fit vectorization and all binary targets together, or train the GRU with six float targets.
# Check one batch and preserve vocabulary, preprocessing, target order and the best candidate
# before increasing epochs.

# %% [markdown]
# ### Supplied helper — `TrainerConfig`
#
# Small lesson loop: AdamW, one step per batch, no scheduler.


# %% [provided] [stage-8]
@dataclass
class TrainerConfig:
    """Small lesson loop: AdamW, one step per batch, no scheduler."""

    epochs: int = 3
    learning_rate: float = 1e-3
    metric_name: str = "accuracy"
    mixed_precision: str = "auto"
    scheduler: str = "none"
    patience: int = 2


# %% [markdown]
# ### Supplied helper — `Trainer`
#
# Visible training loop: validate globally, keep the best, reload before returning.


# %% [provided] [stage-8]
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


# %% [worked] [stage-8]
started = perf_counter()
training = TrainerConfig(
    epochs=EPOCHS,
    learning_rate=1e-3,
    metric_name="roc_auc",
    mixed_precision="auto",
    scheduler="none",
    patience=2,
)
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
checkpoint_path = trainer.checkpoint_path
fit_seconds = perf_counter() - started

# %% [worked] [stage-8]
validation_predictions = (
    torch.from_numpy(predict_logits(model, valid_loader, device=competition.device))
    .sigmoid()
    .numpy()
)
selected_threshold = 0.5

# %% [check] [stage-8]
assert checkpoint_path.is_file(), "No saved candidate is available for inference"
assert np.isfinite(fit_seconds) and fit_seconds >= 0
print("Candidate checks passed: persisted artifact and measured fit time.")

# %% [markdown]
# ### Stop and explain
#
# What evidence proves the saved artifact can reproduce inference? Which training state is not
# promised by this reload?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Keep architecture/preprocessing, label order and best weights together, then compare reloaded
# validation predictions. This lesson preserves inference recovery, not an exact continuation of
# optimizer/scaler/random state.
#
# </details>

# %% [markdown]
# ## 9. Interpret validation and compare experiments
#
# Compute AUROC separately for every target and take their unweighted mean. Inspect per-label
# failures and rare-label support. A strong common label cannot substitute for a missing
# rare-label score.

# %% [worked] [stage-9]
per_label_auc, validation_score = score_multilabel(va, validation_predictions, TARGET_COLUMNS)
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

# %% [check] [stage-9]
assert np.isfinite(validation_score), "Selection requires a finite official score"
print("Validation check passed. Compare this score and runtime with the baseline ledger.")

# %% [markdown]
# ### Stop and explain
#
# What is the next single experiment, and what result would make you reject it?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Compare TF-IDF against the GRU on identical comment groups. Record per-label AUROC, macro AUROC,
# truncation frequency and total runtime. Larger sequence length must justify its memory and
# latency cost.
#
# </details>

# %% [markdown]
# ## 10. Reload and infer every test item
#
# Reload the fitted text pipeline or saved vocabulary/model and reproduce validation
# probabilities. Infer all test comments without fitting vocabulary or changing truncation. Assert
# finite [N,6] probabilities in [0,1].

# %% [worked] [stage-10]
started = perf_counter()
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
assert test_predictions.shape == (len(test), 6)
assert (
    np.isfinite(test_predictions).all()
    and ((0 <= test_predictions) & (test_predictions <= 1)).all()
)
output_values = {label: test_predictions[:, i] for i, label in enumerate(TARGET_COLUMNS)}

# %% [worked] [stage-10]
inference_seconds = perf_counter() - started

# %% [check] [stage-10]
assert set(output_values) == set(sample.columns) - {ID_COLUMN}
assert all(len(values) == len(test) for values in output_values.values())
print("Inference checks passed: every test item has the required output columns.")

# %% [markdown]
# ### Stop and explain
#
# Can this saved candidate handle every private input without refitting or downloading anything?
# Which assets must travel with it?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Reload the fitted text pipeline or saved vocabulary/model and reproduce validation
# probabilities. Infer all test comments without fitting vocabulary or changing truncation. Assert
# finite [N,6] probabilities in [0,1].
#
# </details>

# %% [markdown]
# ## 11. Deliver and verify the submission
#
# Each comment can have several of six independent labels. Return six probabilities per id in the
# sample's exact label order. A row may have no positive label or several; the labels are not
# mutually exclusive classes.
#
# Write the CSV, read it back, and verify IDs, columns, row count, missing values and allowed
# prediction values before saving the complete recipe. A local check does not confirm organizer
# upload acceptance.

# %% [markdown]
# ### Supplied helper — `SubmissionValidation`
#
# Validation errors and non-fatal warnings for a candidate submission.


# %% [provided] [stage-11]
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


# %% [markdown]
# ### Supplied helper — `build_submission`
#
# Copy the sample and optionally align predictions using unique (possibly composite) IDs.
#
# Without IDs, predictions must already be in sample order. Prefer explicit IDs for inference.
# Alignment rejects missing, duplicate, extra, or absent keys rather than guessing row order.


# %% [provided] [stage-11]
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


# %% [markdown]
# ### Supplied helper — `validate_submission`
#
# Check schema, row count, ID order, missing values, and optional allowed labels.


# %% [provided] [stage-11]
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


# %% [markdown]
# ### Supplied helper — `write_submission`
#
# Validate then atomically write a UTF-8 CSV submission.


# %% [provided] [stage-11]
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


# %% [markdown]
# ### Supplied helper — `_mapping`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [provided] [stage-11]
def _mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return value
    raise TypeError("Expected a dataclass or mapping")


# %% [markdown]
# ### Supplied helper — `technical_report`
#
# Render the standard competition handoff report as Markdown text.


# %% [provided] [stage-11]
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


# %% [markdown]
# ### Supplied helper — `write_technical_report`
#
# Write a UTF-8 technical report, creating its parent directory.


# %% [provided] [stage-11]
def write_technical_report(content: str, path: Path | str) -> Path:
    """Write a UTF-8 technical report, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


# %% [markdown]
# ### Supplied helper — `record_run`
#
# Save configuration/fold positions and a matching human-readable technical report.


# %% [provided] [stage-11]
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


# %% [markdown]
# ### Worked component — `deliver_submission`
#
# Input: sample template, named predictions, prediction IDs, key-column name and output path.
# Return: ID-aligned checked frame and saved CSV path. Reject duplicated, missing or extra
# prediction keys.


# %% [worked] [stage-11]
def deliver_submission(
    sample_frame: pd.DataFrame,
    predictions: Mapping[str, Sequence[object]],
    prediction_ids: Sequence[object],
    id_column: str,
    output_path: Path,
) -> Any:
    """Input: sample template, named predictions, prediction IDs, key-column name and output
    path. Return: ID-aligned checked frame and saved CSV path. Reject duplicated, missing or
    extra prediction keys.
    """
    submission = build_submission(
        sample_frame, predictions, prediction_ids=prediction_ids, id_columns=id_column
    )
    validate_submission(submission, sample_frame, id_columns=id_column).raise_for_errors()
    destination = write_submission(submission, output_path)
    return submission, destination


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-11]
toy_sample = pd.DataFrame({"key": ["001", "NA"], "prediction": [0.0, 0.0]})
toy_submission, toy_path = deliver_submission(
    toy_sample, {"prediction": [0.8, 0.2]}, ["NA", "001"], "key", OUTPUT / "checks" / "delivery.csv"
)
assert toy_submission["key"].tolist() == ["001", "NA"], "Preserve literal IDs and sample order"
np.testing.assert_allclose(toy_submission["prediction"], [0.2, 0.8])
toy_loaded = pd.read_csv(toy_path, converters={"key": str}, keep_default_na=False)
assert toy_loaded["key"].tolist() == ["001", "NA"]
try:
    deliver_submission(
        toy_sample,
        {"prediction": [0.2, 0.8]},
        ["001", "001"],
        "key",
        OUTPUT / "checks" / "invalid.csv",
    )
except ValueError:
    pass
else:
    raise AssertionError("Duplicate prediction IDs must be rejected, not guessed")
print("Delivery checks passed: shuffled/literal IDs and duplicate rejection.")

# %% [worked] [stage-11]
submission, destination = deliver_submission(
    sample, output_values, test[ID_COLUMN], ID_COLUMN, OUTPUT / "submission.csv"
)
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

# %% [worked] [stage-11]
save_json(
    {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "max_rows": MAX_ROWS,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "imbalance_weights": USE_IMBALANCE_WEIGHTS,
        "pretrained_allowed": PRETRAINED_ALLOWED,
        "asset_provenance": ASSET_PROVENANCE,
        "trainer": globals().get("training"),
    },
    OUTPUT / "lesson_config.json",
)

# %% [worked] [stage-11]
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

# %% [worked] [stage-11]
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
copied = sync_artifacts(OUTPUT, DURABLE_OUTPUT)
print("Checked local CSV:", destination, "Durable files copied:", len(copied))

# %% [check] [stage-11]
assert destination.is_file()
assert submission[ID_COLUMN].tolist() == sample[ID_COLUMN].tolist()
print("Delivery complete. Next: restart inference, then attempt the reconstruction task.")

# %% [markdown]
# ### Stop and explain
#
# Which ID-order or encoding mistake would survive a plausible-looking preview? What did the
# on-disk round trip prove?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Shuffled sample IDs and literal strings such as 001/NA require explicit alignment and CSV
# parsing. Readback checks the delivered file's schema/order/values; official task semantics and
# upload acceptance still need separate confirmation.
#
# </details>

# %% [markdown]
# ## Transfer and closed-book reconstruction
#
# A new pack has three independent labels, duplicate comments and very long messages. Update
# output dimension, target order, group split and truncation audit, then prove that metric
# averaging still handles all labels.
#
# 1. Close the reference notebook and create an empty personal notebook.
# 2. Rebuild the data contract, split, features/model, fit, official metric and submission path.
# 3. Deliver a checked small-pack CSV within 45–60 minutes, then reload it in a fresh session.
# 4. Record elapsed time, hints, bugs and the first function you could not reconstruct.
# 5. Rebuild that function the next day with a different schema or seed.
#
# For the 2k assistant, send one function signature, input/output shapes and one failing toy test.
# Ask for an explanation or correction of that function only. Keep the rest of the notebook
# outside the prompt.
#
# ### Controlled comparison
#
# Compare TF-IDF against the GRU on identical comment groups. Record per-label AUROC, macro AUROC,
# truncation frequency and total runtime. Larger sequence length must justify its memory and
# latency cost.
#
# A first correct submission is the readiness gate. Advanced complexity is selected only after an
# honest metric/runtime comparison; no test fitting or model-weight permission is implied.
