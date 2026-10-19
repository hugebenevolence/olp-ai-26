# %% [markdown]
# # Shopee Price Match Guarantee — fill-in baseline
#
# Problem family: **Multimodal product matching**. Official metric: **Mean per-query set F1,
# including self**.
#
# **Learning objective:** Training-title TF-IDF + batched cosine threshold matching.
#
# [Dataset and original task](https://www.kaggle.com/competitions/shopee-product-matching) ·
# [Primary solution
# evidence](https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr)
#
# The linked first-place write-up motivates image/text matching and normalized modality fusion.
# This baseline fits training-title TF-IDF and cosine matching; the advanced companion studies
# permitted frozen local encoders and fusion. Neither implements ArcFace training or iterative
# neighborhood blending.
#
# **Watch for:** Hold out entire product groups. Matching is transductive gallery inference, not
# fitting on test labels. Include each item itself.
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
# Complete the named exercises and their nearby checks in order. Setup and safety helpers are
# supplied.
#
# After the lesson, close both notebooks and rebuild the essential route from memory. See the
# repository's blank-notebook playbook for the final transfer test.
#
# ## How to use this lesson
#
# Complete one named exercise, run its nearby checks, then interpret the result. Supplied cells
# contain setup or integration; edit only configuration or marked exercise bodies. Every baseline
# code cell retains a collapsed full reference immediately afterward. Use Hint 1, then Hint 2,
# before opening a reference.
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
from collections.abc import Iterable, Mapping, Sequence
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
FAMILY, VARIANT = "07_matching", "baseline"
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
SEED, MAX_ROWS = 42, 6000
TIME_BUDGET_MINUTES, DELIVERY_RESERVE_MINUTES = 60, 10
GROUP_COLUMN, TIME_COLUMN = "label_group", None
USE_IMBALANCE_WEIGHTS = False
MODEL_NAME = "title_tfidf"

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
KAGGLE_COMPETITION = "shopee-product-matching"
EXPECTED_DATA_PATHS = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "train_images",
    "test_images",
)
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
# Return a set of matching posting IDs for every query item. The original Shopee contract includes
# the query itself. Product groups define relevance; pairwise accuracy is not the official mean
# per-query set F1.
#
# Read the first rows, column names and actual shapes below. Expected: unique nonempty prediction
# IDs; matching test/sample ID sets; a target schema that agrees with the official output. Dataset
# size depends on your local files and MAX_ROWS.

# %% [worked] [stage-2]
ID_COLUMN, TARGET_COLUMNS = "posting_id", ["matches"]
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str, "label_group": str})
test = pd.read_csv(DATA_DIR / "test.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
train["title"] = train["title"].fillna("")
test["title"] = test["title"].fillna("")
train["_image"] = "train_images/" + train["image"]
test["_image"] = "test_images/" + test["image"]
if MAX_ROWS is not None and len(train) > MAX_ROWS:
    group_sizes = train.groupby("label_group").size().sample(frac=1, random_state=SEED)
    # Keep whole groups, including the group that crosses the approximate row budget.
    keep_groups = group_sizes.index[group_sizes.cumsum().shift(fill_value=0) < MAX_ROWS]
    train = train.loc[train["label_group"].isin(keep_groups)].reset_index(drop=True)
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

# %% [worked] [stage-2]
print("Shapes:", train.shape, test.shape, sample.shape)
if train["label_group"].eq("").any() or train["label_group"].isna().any():
    raise ValueError("Every training item needs a product-group identity")

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
# Return a set of matching posting IDs for every query item. The original Shopee contract includes
# the query itself. Product groups define relevance; pairwise accuracy is not the official mean
# per-query set F1.
#
# </details>

# %% [markdown]
# ## 3. Inspect examples and distributions
#
# Inspect the plotted examples before selecting a model. Write observations before reading the
# next section.
#
# How large are product groups? Do duplicate titles or images cross identities? Inspect examples
# with similar words but different products and different titles for the same product.

# %% [worked] [stage-3]
group_sizes = train.groupby("label_group").size()
print("Product group-size quantiles:", group_sizes.quantile([0, 0.5, 0.9, 1]).to_dict())
print("Singleton product fraction:", (group_sizes == 1).mean())
print("Repeated image hashes:", train["image_phash"].duplicated().sum())
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (_, row) in zip(axes, train.head(3).iterrows(), strict=False):
    with Image.open(DATA_DIR / row["_image"]) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(row["title"][:55])
    ax.axis("off")
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
# Hold out entire label_group values. Random item or pair splits leak product identity.
# Validation/test gallery comparisons are inference in this task; vocabulary or encoder fitting on
# test still requires explicit permission.

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


# %% [worked] [stage-4]
if TIME_COLUMN:
    raise ValueError("Use group holdout here; design temporal gallery evaluation separately")

# %% [markdown]
# ### Exercise 1 — `create_holdout`
#
# Input: a row-per-prediction table and its actual leakage columns. Return: the split record and
# independent training/validation copies. The union must retain all rows; identities must not
# cross folds.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Hold out entire label_group values. Random item or pair splits leak product identity.
# Validation/test gallery comparisons are inference in this task; vocabulary or encoder fitting on
# test still requires explicit permission.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# Use make_split with valid_size=0.2; return copies of its train and valid tables.
#
# </details>


# %% [exercise] [stage-4]
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
    # >>> exercise: implement create_holdout using the stated contract
    split = make_split(
        frame,
        target_columns=target_columns,
        group_column=group_column,
        time_column=time_column,
        valid_size=0.2,
        seed=seed,
    )
    tr, va = split.train.copy(), split.valid.copy()
    # <<< exercise
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
    train, target_columns=None, group_column="label_group", time_column=TIME_COLUMN, seed=SEED
)
assert not set(tr["label_group"]) & set(va["label_group"])
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
print("Held-out product groups:", va["label_group"].nunique(), "Gallery items:", len(va))

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
# Hold out entire label_group values. Random item or pair splits leak product identity.
# Validation/test gallery comparisons are inference in this task; vocabulary or encoder fitting on
# test still requires explicit permission.
#
# </details>

# %% [markdown]
# ## 5. Measure imbalance and choose a response
#
# Large product groups create many more positive pairs: a group of k items has k(k-1)/2 pairs
# excluding self. Measure this on training groups and avoid overrepresenting a few large products
# if training a pair model.
#
# $F1_i=2|P_i\cap T_i|/(|P_i|+|T_i|)$, averaged over queries. A self-only prediction can score
# above zero without retrieving another product item.

# %% [markdown]
# ### Exercise 2 — `positive_pair_counts`
#
# Input: one training product-group label per item. Return: group sizes, positive unordered pair
# count and all unordered non-self pairs.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Large product groups create many more positive pairs: a group of k items has k(k-1)/2 pairs
# excluding self. Measure this on training groups and avoid overrepresenting a few large products
# if training a pair model.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# Each group of size k contributes k*(k-1)/2 positive pairs; calculate the full denominator from
# the number of items.
#
# </details>


# %% [exercise] [stage-5]
def positive_pair_counts(groups: pd.Series) -> Any:
    """Input: one training product-group label per item. Return: group sizes, positive
    unordered pair count and all unordered non-self pairs.
    """
    # >>> exercise: implement positive_pair_counts using the stated contract
    sizes = groups.value_counts()
    positive_pairs = int((sizes * (sizes - 1) / 2).sum())
    all_pairs = len(groups) * (len(groups) - 1) // 2
    # <<< exercise
    return sizes, positive_pairs, all_pairs


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-5]
toy_sizes, toy_positive, toy_all = positive_pair_counts(pd.Series(["a", "a", "b"]))
assert toy_positive == 1 and toy_all == 3
assert int(toy_sizes.sum()) == 3
print("Pair-count checks passed: one positive among three unordered non-self pairs.")

# %% [worked] [stage-5]
sizes, positive_pairs, all_pairs = positive_pair_counts(tr["label_group"])
print("Positive pairs:", positive_pairs, "All pairs:", all_pairs)
print("Response: group holdout and per-query F1; keep validation unchanged")
print("For metric learning later: sample product groups, not all pairs equally.")
sizes.hist(bins=20)
plt.title("Training product-group sizes")
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
# Large product groups create many more positive pairs: a group of k items has k(k-1)/2 pairs
# excluding self. Measure this on training groups and avoid overrepresenting a few large products
# if training a pair model.
#
# </details>

# %% [markdown]
# ## 6. Build features or one correct batch
#
# Fit title TF-IDF on training products only. L2-normalized sparse title features allow cosine
# similarity. Transform validation/test titles with the saved vocabulary, keeping gallery row
# order tied to posting_id.

# %% [worked] [stage-6]
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# ### Exercise 3 — `make_text_features`
#
# Return an unfitted word unigram/bigram TF-IDF vectorizer. Vocabulary and IDF will be learned
# only in the training step; construction must not read validation or test text.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Fit title TF-IDF on training products only. L2-normalized sparse title features allow cosine
# similarity. Transform validation/test titles with the saved vocabulary, keeping gallery row
# order tied to posting_id.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# Use ngram_range=(1,2), min_df=1, max_features=60000 and sublinear_tf=True.
#
# </details>


# %% [exercise] [stage-6]
def make_text_features() -> Any:
    """Return an unfitted word unigram/bigram TF-IDF vectorizer. Vocabulary and IDF will be
    learned only in the training step; construction must not read validation or test text.
    """
    # >>> exercise: implement make_text_features using the stated contract
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=60000, sublinear_tf=True
    )
    # <<< exercise
    return vectorizer


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-6]
toy_vectorizer = make_text_features()
toy_vectorizer.fit(["red leaf", "green leaf", "green stem"])
toy_features = toy_vectorizer.transform(["unseenword leaf", ""])
assert toy_features.shape[0] == 2
assert "unseenword" not in toy_vectorizer.vocabulary_, "Transformation must not refit vocabulary"
assert np.isfinite(toy_features.data).all()
print("Text feature checks passed: new words and empty text are handled without fitting.")

# %% [worked] [stage-6]
vectorizer = make_text_features()
print("Training title preview:", tr["title"].head(3).tolist())

# %% [markdown]
# ### Stop and explain
#
# State the feature/batch shapes and the exact point at which preprocessing is fitted. What
# happens to a previously unseen validation input?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Fit title TF-IDF on training products only. L2-normalized sparse title features allow cosine
# similarity. Transform validation/test titles with the saved vocabulary, keeping gallery row
# order tied to posting_id.
#
# </details>

# %% [markdown]
# ## 7. Connect outputs, loss and official metric
#
# Compare title cosine scores in bounded query batches. Every query must include itself exactly
# once, including zero-feature titles. Batch size must not change the returned match sets.

# %% [markdown]
# ### Supplied helper — `save_pickle`
#
# Atomically save a fitted local pipeline with preprocessing and original label values.


# %% [provided] [stage-7]
def save_pickle(value: Any, path: Path | str) -> Path:
    """Atomically save a fitted local pipeline with preprocessing and original label values."""
    return _write(value, path, "pickle")


# %% [markdown]
# ### Supplied helper — `load_pickle`
#
# Reload a trusted, locally produced pickle; never load untrusted downloaded pickle files.


# %% [provided] [stage-7]
def load_pickle(path: Path | str) -> Any:
    """Reload a trusted, locally produced pickle; never load untrusted downloaded pickle files."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


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


# %% [worked] [stage-7]
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ### Worked component — `matching_f1`
#
# Shopee mean per-query set F1, including the query itself in its group and prediction.


# %% [worked] [stage-7]
def matching_f1(groups: Sequence[object], matches: Sequence[Sequence[int]]) -> float:
    """Shopee mean per-query set F1, including the query itself in its group and prediction."""
    labels = np.asarray(groups)
    if not len(labels) or len(labels) != len(matches):
        raise ValueError("Provide one match set per gallery item")
    values = []
    for index, predictions in enumerate(matches):
        candidate = set(predictions)
        if not candidate or any(not 0 <= item < len(labels) for item in candidate):
            raise ValueError("Match sets must contain valid gallery positions")
        relevant = set(np.flatnonzero(labels == labels[index]).tolist())
        values.append(2 * len(candidate & relevant) / (len(candidate) + len(relevant)))
    return float(np.mean(values))


# %% [markdown]
# ### Exercise 4 — `match_positions`
#
# Input: row-aligned features [N,D], cosine threshold and query batch size. Return N lists of
# gallery positions including self exactly once. Changing batch size must not change the sets.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Compare title cosine scores in bounded query batches. Every query must include itself exactly
# once, including zero-feature titles. Batch size must not change the returned match sets.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# bound the dense query-by-gallery matrix and always include self
#
# </details>


# %% [exercise] [stage-7]
def match_positions(features, threshold, batch_size=32):
    if not 0 < threshold <= 1:
        raise ValueError("Choose a cosine threshold in (0, 1]")
    matches = []
    # >>> exercise: bound the dense query-by-gallery matrix and always include self
    for start in range(0, features.shape[0], batch_size):
        similarities = cosine_similarity(features[start : start + batch_size], features)
        for local, row in enumerate(similarities):
            row[start + local] = 1.0
            matches.append(np.flatnonzero(row >= threshold).tolist())
    # <<< exercise
    return matches


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-7]
toy_vectors = np.eye(3, dtype=np.float32)
assert match_positions(toy_vectors, 0.8, batch_size=1) == [[0], [1], [2]]
assert match_positions(toy_vectors, 0.8, batch_size=3) == [[0], [1], [2]], (
    "Batching changed results"
)
toy_duplicates = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
assert match_positions(toy_duplicates, 0.8, batch_size=1) == [[0, 1], [0, 1], [2]], (
    "Retrieve matching items, not just self"
)
assert match_positions(toy_duplicates, 0.8, batch_size=3) == [[0, 1], [0, 1], [2]]
toy_zero = np.zeros((2, 3), dtype=np.float32)
assert match_positions(toy_zero, 0.8, batch_size=1) == [[0], [1]], (
    "Zero-feature items still need self"
)
print("Matching checks passed: self inclusion and batch-size invariance.")

# %% [markdown]
# ### Worked component — `features_for`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-7]
def features_for(frame, weight=0.0):
    return vectorizer.transform(frame["title"])


# %% [worked] [stage-7]
MODEL_DESCRIPTION = "training-title TF-IDF cosine matching"

# %% [markdown]
# ### Exercise 5 — `score_match_sets`
#
# Input: validation group labels and complete predicted gallery-index sets, including self. Return
# mean per-query set F1, not binary pair accuracy.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Score complete predicted sets including self. Inspect false merges, missed group members and
# match counts. Self-only is an informative reference, not proof of useful retrieval. A second
# group split checks threshold optimism.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# Pass complete sets and labels in the same gallery order to matching_f1.
#
# </details>


# %% [exercise] [stage-7]
def score_match_sets(groups: Sequence[object], positions: Sequence[Sequence[int]]) -> Any:
    """Input: validation group labels and complete predicted gallery-index sets, including
    self. Return mean per-query set F1, not binary pair accuracy.
    """
    # >>> exercise: implement score_match_sets using the stated contract
    value = matching_f1(groups, positions)
    # <<< exercise
    return value


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-7]
assert score_match_sets(["a", "a", "b"], [[0, 1], [0, 1], [2]]) == 1
np.testing.assert_allclose(score_match_sets(["a", "a", "b"], [[0], [1], [2]]), 7 / 9)
print("Set-F1 checks passed: complete sets score one; self-only scores 7/9 on this gallery.")

# %% [markdown]
# ### Stop and explain
#
# Explain the output shape, loss and prediction decoding without reading the implementation. Which
# convention would silently produce an invalid score?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# Compare title cosine scores in bounded query batches. Every query must include itself exactly
# once, including zero-feature titles. Batch size must not change the returned match sets. Score
# complete predicted sets including self. Inspect false merges, missed group members and match
# counts. Self-only is an informative reference, not proof of useful retrieval. A second group
# split checks threshold optimism.
#
# </details>

# %% [markdown]
# ## 8. Train and preserve a recoverable candidate
#
# The sparse route fits vocabulary/IDF on training products. The advanced route persists permitted
# feature/encoder assets and selects its mixture on validation. Count feature extraction and
# selection in total runtime.

# %% [worked] [stage-8]
started = perf_counter()
budget.checkpoint("fit training-title vocabulary")

# %% [markdown]
# ### Exercise 6 — `fit_candidate`
#
# Fit only the supplied training data and atomically persist the entire fitted candidate under
# output_dir. Return its checkpoint path. The caller must not pass validation/test data to this
# function.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# The sparse route fits vocabulary/IDF on training products. The advanced route persists permitted
# feature/encoder assets and selects its mixture on validation. Count feature extraction and
# selection in total runtime.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# Fit the supplied object, then save_pickle the same fitted object. Persist preprocessing with the
# estimator.
#
# </details>


# %% [exercise] [stage-8]
def fit_candidate(candidate: Any, training_text: Sequence[str], output_dir: Path) -> Any:
    """Fit only the supplied training data and atomically persist the entire fitted candidate
    under output_dir. Return its checkpoint path. The caller must not pass validation/test
    data to this function.
    """
    # >>> exercise: implement fit_candidate using the stated contract
    candidate.fit(training_text)
    checkpoint_path = save_pickle(candidate, output_dir / "vectorizer.pkl")
    # <<< exercise
    return checkpoint_path


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-8]
toy_vectorizer = make_text_features()
toy_path = fit_candidate(
    toy_vectorizer, ["red product", "green product"], OUTPUT / "checks" / "fit"
)
toy_restored = load_pickle(toy_path)
assert "red" in toy_restored.vocabulary_ and "testonlyword" not in toy_restored.vocabulary_
assert toy_restored.transform(["testonlyword"]).shape[0] == 1
print("Fit/reload checks passed: saved training vocabulary handles a new title.")

# %% [worked] [stage-8]
checkpoint_path = fit_candidate(vectorizer, tr["title"], OUTPUT)
fit_seconds = perf_counter() - started

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
# Score complete predicted sets including self. Inspect false merges, missed group members and
# match counts. Self-only is an informative reference, not proof of useful retrieval. A second
# group split checks threshold optimism.

# %% [worked] [stage-9]
selection_started = perf_counter()
weights_to_compare = (0.0,)
thresholds = (0.5,)
comparisons = []
for weight in weights_to_compare:
    for threshold in thresholds:
        budget.checkpoint("validation matching comparison")
        features = features_for(va, weight)
        positions = match_positions(features, threshold)
        value = score_match_sets(va["label_group"].tolist(), positions)
        comparisons.append({"weight": weight, "threshold": threshold, "score": value})
print(
    "Self-only reference F1:",
    matching_f1(va["label_group"].tolist(), [[index] for index in range(len(va))]),
)
comparison = pd.DataFrame(comparisons)
chosen = comparison.sort_values("score", ascending=False, kind="stable").iloc[0]
selected_weight, selected_threshold = float(chosen["weight"]), float(chosen["threshold"])
validation_score = float(chosen["score"])
validation_positions = match_positions(features_for(va, selected_weight), selected_threshold)
fit_seconds += perf_counter() - selection_started
print("Validation-only comparison:", comparison.to_string(index=False))
comparison.to_csv(OUTPUT / "threshold_comparison.csv", index=False)

# %% [worked] [stage-9]
errors = pd.DataFrame(
    {
        ID_COLUMN: va[ID_COLUMN],
        "true_group": va["label_group"],
        "predicted_match_count": [len(value) for value in validation_positions],
    }
)
print(errors.head(20).to_string(index=False))
errors.to_csv(OUTPUT / "validation_errors.csv", index=False)
comparison.plot.scatter(
    x="threshold",
    y="score",
    c="weight",
    cmap="viridis",
    title="Validation F1 versus threshold; color = image weight",
)
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
# Compare title-only, image-only and fused features on identical held-out product groups. Include
# extraction/storage costs and self-only F1. Validate cache provenance and prove the route can
# process new IDs before choosing it.
#
# </details>

# %% [markdown]
# ## 10. Reload and infer every test item
#
# Reload the trained vocabulary or preserved encoders/cache, recover chosen fusion settings and
# infer all test match sets. Translate gallery positions back to posting IDs; include self and
# avoid duplicate IDs.

# %% [worked] [stage-10]
started = perf_counter()

# %% [markdown]
# ### Exercise 7 — Reload and predict
#
# Reload the trained vocabulary or preserved encoders/cache, recover chosen fusion settings and
# infer all test match sets. Translate gallery positions back to posting IDs; include self and
# avoid duplicate IDs.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Reload the trained vocabulary or preserved encoders/cache, recover chosen fusion settings and
# infer all test match sets. Translate gallery positions back to posting IDs; include self and
# avoid duplicate IDs.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# reload the training vocabulary; never refit it on the test gallery
#
# </details>

# %% [exercise] [stage-10]
# >>> exercise: reload the training vocabulary; never refit it on the test gallery
vectorizer = load_pickle(checkpoint_path)
assert match_positions(features_for(va), selected_threshold) == validation_positions
test_positions = match_positions(features_for(test), selected_threshold)
# <<< exercise

# %% [worked] [stage-10]
output_values = {
    "matches": [" ".join(test.iloc[positions][ID_COLUMN].tolist()) for positions in test_positions]
}
for index, positions in enumerate(test_positions):
    assert index in positions and len(positions) == len(set(positions))
save_json(
    {
        "threshold": selected_threshold,
        "vocabulary_fit": "training products only",
        "matching": "test-to-test gallery inference; no training on test",
    },
    OUTPUT / "recipe.json",
)
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
# Reload the trained vocabulary or preserved encoders/cache, recover chosen fusion settings and
# infer all test match sets. Translate gallery positions back to posting IDs; include self and
# avoid duplicate IDs.
#
# </details>

# %% [markdown]
# ## 11. Deliver and verify the submission
#
# Return a set of matching posting IDs for every query item. The original Shopee contract includes
# the query itself. Product groups define relevance; pairwise accuracy is not the official mean
# per-query set F1.
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
# ### Exercise 8 — `deliver_submission`
#
# Input: sample template, named predictions, prediction IDs, key-column name and output path.
# Return: ID-aligned checked frame and saved CSV path. Reject duplicated, missing or extra
# prediction keys.
#
# <details>
# <summary>Hint 1 — choose the operation</summary>
#
# Return a set of matching posting IDs for every query item. The original Shopee contract includes
# the query itself. Product groups define relevance; pairwise accuracy is not the official mean
# per-query set F1.
#
# </details>
#
# <details>
# <summary>Hint 2 — implementation direction</summary>
#
# Build by ID, raise validation errors, then write the validated frame. Never assume test order
# equals sample order.
#
# </details>


# %% [exercise] [stage-11]
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
    # >>> exercise: implement deliver_submission using the stated contract
    submission = build_submission(
        sample_frame, predictions, prediction_ids=prediction_ids, id_columns=id_column
    )
    validate_submission(submission, sample_frame, id_columns=id_column).raise_for_errors()
    destination = write_submission(submission, output_path)
    # <<< exercise
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
save_json(
    {
        "seed": SEED,
        "max_rows": MAX_ROWS,
        "model_name": MODEL_NAME,
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
# A private gallery introduces unseen posting IDs and zero-text items. Explain why an ID-bound
# feature cache cannot handle them, then rebuild the sparse baseline and verify self inclusion
# across query batch sizes.
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
# Compare title-only, image-only and fused features on identical held-out product groups. Include
# extraction/storage costs and self-only F1. Validate cache provenance and prove the route can
# process new IDs before choosing it.
#
# A first correct submission is the readiness gate. Advanced complexity is selected only after an
# honest metric/runtime comparison; no test fitting or model-weight permission is implied.
