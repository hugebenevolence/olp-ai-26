# %% [markdown]
# # Global Wheat Detection — advanced / SOTA study
#
# Problem family: **Object detection**. Official metric: **Mean per-image TP/(TP+FP+FN), IoU .50
# to .75**.
#
# **Learning objective:** Single Faster R-CNN ResNet FPN, permitted local initialization and
# confidence tuning.
#
# [Dataset and original task](https://www.kaggle.com/competitions/global-wheat-detection) ·
# [Primary solution evidence](https://github.com/dungnb1333/Global-Wheat-Dection-1st-Place)
#
# The first-place repository includes Faster R-CNN FPN alongside EfficientDet, fusion and extra
# data. This lesson uses a smaller single torchvision FPN detector in that family; it omits
# extra data, test pseudo-labeling and the multi-model ensemble.
#
# **Watch for:** Split images, not box rows. Torchvision consumes xyxy; Kaggle submits
# confidence,x,y,width,height. This is not COCO mAP.
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
from numpy.typing import ArrayLike
from PIL import Image
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
FAMILY, VARIANT = "06_wheat", "advanced"
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
SEED, MAX_ROWS, IMAGE_SIZE, BATCH_SIZE, EPOCHS = 42, 300, 384, 2, 2
TIME_BUDGET_MINUTES, DELIVERY_RESERVE_MINUTES = 60, 10
GROUP_COLUMN, TIME_COLUMN = "source", None
USE_IMBALANCE_WEIGHTS = False
MODEL_NAME = "fasterrcnn_resnet50_fpn"
IMAGE_SUFFIX = ".jpg"

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
KAGGLE_COMPETITION = "global-wheat-detection"
EXPECTED_DATA_PATHS = ("train.csv", "sample_submission.csv", "train", "test")
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
# One image may contain many boxes, so annotation rows must first be grouped by image_id. The
# output is a prediction string of confidence and xywh boxes per image, including a valid empty
# string when no box survives.
#
# Read the first rows, column names and actual shapes below. Expected: unique nonempty prediction
# IDs; matching test/sample ID sets; a target schema that agrees with the official output. Dataset
# size depends on your local files and MAX_ROWS.

# %% [worked] [stage-2]
import ast

ID_COLUMN, TARGET_COLUMNS = "image_id", ["PredictionString"]
annotations = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
parsed = np.asarray([ast.literal_eval(text) for text in annotations["bbox"]], dtype=np.float32)
assert parsed.shape == (len(annotations), 4)
assert np.isfinite(parsed).all() and (parsed[:, 2:] > 0).all()
annotations[["xmin", "ymin"]] = parsed[:, :2]
annotations[["xmax", "ymax"]] = parsed[:, :2] + parsed[:, 2:]
annotations["label"] = 1  # background is zero in torchvision
annotations["image"] = "train/" + annotations[ID_COLUMN] + IMAGE_SUFFIX
train = (
    annotations.groupby(ID_COLUMN, sort=False)
    .agg(
        source=("source", "first"),
        box_count=("bbox", "size"),
        image=("image", "first"),
        width=("width", "first"),
        height=("height", "first"),
    )
    .reset_index()
)
assert annotations.groupby(ID_COLUMN)["source"].nunique().max() == 1
test = sample[[ID_COLUMN]].copy()

# %% [worked] [stage-2]
test["image"] = "test/" + test[ID_COLUMN] + IMAGE_SUFFIX
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
# One image may contain many boxes, so annotation rows must first be grouped by image_id. The
# output is a prediction string of confidence and xywh boxes per image, including a valid empty
# string when no box survives.
#
# </details>

# %% [markdown]
# ## 3. Inspect examples and distributions
#
# Inspect the plotted examples before selecting a model. Write observations before reading the
# next section.
#
# How do box counts and sizes vary by acquisition source? Do drawn boxes fit the actual image? Are
# there negative images absent from the annotation table, and are bbox values xywh or xyxy?

# %% [worked] [stage-3]
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (_, row) in zip(axes, train.head(3).iterrows(), strict=False):
    with Image.open(DATA_DIR / row["image"]) as image:
        ax.imshow(image.convert("RGB"))
    boxes = annotations.loc[annotations[ID_COLUMN] == row[ID_COLUMN]]
    for _, box in boxes.iterrows():
        ax.add_patch(
            Rectangle(
                (box["xmin"], box["ymin"]),
                box["xmax"] - box["xmin"],
                box["ymax"] - box["ymin"],
                fill=False,
                edgecolor="red",
            )
        )
    ax.set_title(f"{row['box_count']} boxes; {row['source']}")
    ax.axis("off")
fig.savefig(OUTPUT / "inspection.png")
plt.show()
plt.close(fig)
print("Per-source images:", train["source"].value_counts().to_dict())

# %% [worked] [stage-3]
print(
    "Box-size quantiles:",
    pd.DataFrame(parsed[:, 2:], columns=["width", "height"]).quantile([0, 0.5, 0.9, 1]),
)

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
# Split image records, never annotation rows. Keep acquisition sources together when available. If
# one image's boxes appear on both sides, the validation score no longer measures generalization
# to unseen images.

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
print("Image counts:", len(tr), len(va), "Validation sources:", va["source"].unique())

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
# Split image records, never annotation rows. Keep acquisition sources together when available. If
# one image's boxes appear on both sides, the validation score no longer measures generalization
# to unseen images.
#
# </details>

# %% [markdown]
# ## 5. Measure imbalance and choose a response
#
# There is one foreground class, but image density and source frequencies can be uneven. Derive
# optional density sampling weights from training images only. Do not treat every box as an
# independent training example.
#
# Per-image score uses $TP/(TP+FP+FN)$ averaged over IoU thresholds. This notebook exposes strict
# IoU/confidence boundaries and inclusive-pixel areas; it does not substitute COCO AP.

# %% [worked] [stage-5]
from torch.utils.data import WeightedRandomSampler

# %% [markdown]
# ### Worked component — `density_sampling`
#
# Input: one box-count value per training image. Return: density bins, frequencies and per-image
# sampling weights [N]. This is not a box-level split.


# %% [worked] [stage-5]
def density_sampling(box_counts: pd.Series) -> Any:
    """Input: one box-count value per training image. Return: density bins, frequencies and
    per-image sampling weights [N]. This is not a box-level split.
    """
    density_bins = pd.cut(box_counts, bins=[-1, 2, 10, 50, 1e9])
    frequency = density_bins.value_counts()
    sampling_weights = (1 / density_bins.map(frequency).astype(float)).to_numpy()
    return density_bins, frequency, sampling_weights


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-5]
toy_bins, toy_frequency, toy_sampling = density_sampling(pd.Series([1, 1, 8]))
np.testing.assert_allclose(toy_sampling, [0.5, 0.5, 1.0])
assert np.isfinite(toy_sampling).all()
print("Density checks passed: rare density bins receive larger training sampling weights.")

# %% [worked] [stage-5]
density_bins, frequency, sampling_weights = density_sampling(tr["box_count"])
print("One foreground class; no invented foreground class-balancing weights")
print("Density bins:", frequency.to_dict(), "Sampling enabled:", USE_IMBALANCE_WEIGHTS)
tr["box_count"].hist(bins=20)
plt.title("Training boxes per image")
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
# There is one foreground class, but image density and source frequencies can be uneven. Derive
# optional density sampling weights from training images only. Do not treat every box as an
# independent training example.
#
# </details>

# %% [markdown]
# ## 6. Build features or one correct batch
#
# The ResNet-FPN detector uses the same variable-length box contract and image-level split. Verify
# xywh-to-xyxy conversion, tensor dtypes, image sizes and train/eval modes before increasing
# resolution.

# %% [worked] [stage-6]
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

# %% [markdown]
# ### Worked component — `BoxDetectionDataset`
#
# Read detection annotations with one row per bounding box.
#
# Expected coordinates are absolute ``xmin, ymin, xmax, ymax`` values. Rows sharing an image
# path are grouped into one target dictionary in the format expected by torchvision detectors.
# Class index 0 is reserved for background, so foreground labels must start at 1.


# %% [worked] [stage-6]
class BoxDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Read detection annotations with one row per bounding box.

    Expected coordinates are absolute ``xmin, ymin, xmax, ymax`` values. Rows sharing an image
    path are grouped into one target dictionary in the format expected by torchvision detectors.
    Class index 0 is reserved for background, so foreground labels must start at 1.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str = "image",
        label_column: str = "label",
        box_columns: tuple[str, str, str, str] = ("xmin", "ymin", "xmax", "ymax"),
        root: Path | str = ".",
    ) -> None:
        """Validate annotation columns and group rows by image path."""
        required = {image_column, label_column, *box_columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Detection table is missing columns: {sorted(missing)}")
        self.groups = list(frame.groupby(image_column, sort=False))
        self.label_column = label_column
        self.box_columns = box_columns
        self.root = Path(root)

    def __len__(self) -> int:
        """Return the number of distinct images."""
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load one RGB image and all of its boxes and labels."""
        relative_path, rows = self.groups[index]
        with Image.open(self.root / str(relative_path)) as source:
            image = TF.pil_to_tensor(source.convert("RGB")).float() / 255
        boxes = torch.as_tensor(rows.loc[:, self.box_columns].to_numpy(), dtype=torch.float32)
        labels = torch.as_tensor(rows[self.label_column].to_numpy(), dtype=torch.int64)
        if (labels < 1).any():
            raise ValueError("Torchvision foreground detection labels must start at 1")
        if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("Boxes must be finite exclusive xyxy rectangles with positive area")
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


# %% [markdown]
# ### Worked component — `DetectionInferenceDataset`
#
# Read an ordered image manifest without resizing or normalizing detector inputs.


# %% [worked] [stage-6]
class DetectionInferenceDataset(Dataset[Any]):
    """Read an ordered image manifest without resizing or normalizing detector inputs."""

    def __init__(
        self, frame: pd.DataFrame, *, image_column: str = "image", root: Path | str = "."
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.root = Path(root)

    def __len__(self) -> int:
        """Return the number of inference images."""
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return an RGB tensor and empty target for the variable-size detection collator."""
        with Image.open(self.root / str(self.frame.iloc[index][self.image_column])) as image:
            return TF.pil_to_tensor(image.convert("RGB")).float() / 255, {}


# %% [markdown]
# ### Worked component — `detection_collate`
#
# Collate variable-size images and variable-length detection targets.


# %% [worked] [stage-6]
def detection_collate(batch: list[tuple[Any, Any]]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Collate variable-size images and variable-length detection targets."""
    return tuple(zip(*batch, strict=True))


# %% [markdown]
# ### Worked component — `predict_detections`
#
# Return CPU detection dictionaries in the manifest/dataset order, fitting nothing.


# %% [worked] [stage-6]
@torch.inference_mode()
def predict_detections(
    model: nn.Module, loader: Any, *, device: str = "cpu"
) -> list[dict[str, torch.Tensor]]:
    """Return CPU detection dictionaries in the manifest/dataset order, fitting nothing."""
    model.to(device).eval()
    outputs = []
    for images, _ in loader:
        outputs.extend(
            {key: value.cpu() for key, value in prediction.items()}
            for prediction in model([image.to(device) for image in images])
        )
    return outputs


# %% [markdown]
# ### Worked component — `loader_for`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-6]
def loader_for(frame, training=False, labeled=True):
    if labeled:
        # Merge in manifest order; a box table's original order is not a prediction ID contract.
        ordered = frame[[ID_COLUMN]].merge(
            annotations, on=ID_COLUMN, sort=False, validate="one_to_many"
        )
        dataset = BoxDetectionDataset(ordered, root=DATA_DIR)
    else:
        dataset = DetectionInferenceDataset(frame, root=DATA_DIR)
    sampler = (
        WeightedRandomSampler(sampling_weights, len(dataset), replacement=True)
        if training and USE_IMBALANCE_WEIGHTS
        else None
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training and sampler is None,
        sampler=sampler,
        collate_fn=detection_collate,
        num_workers=0,
    )


# %% [worked] [stage-6]
train_loader, valid_loader = loader_for(tr, True), loader_for(va)
images, targets = next(iter(train_loader))
assert images[0].shape[0] == 3 and targets[0]["boxes"].shape[1] == 4
print("Images in batch:", len(images), "First target boxes:", targets[0]["boxes"].shape)
truth_boxes = [
    annotations.loc[annotations[ID_COLUMN] == key, ["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    for key in va[ID_COLUMN]
]

# %% [markdown]
# ### Stop and explain
#
# State the feature/batch shapes and the exact point at which preprocessing is fitted. What
# happens to a previously unseen validation input?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# The ResNet-FPN detector uses the same variable-length box contract and image-level split. Verify
# xywh-to-xyxy conversion, tensor dtypes, image sizes and train/eval modes before increasing
# resolution.
#
# </details>

# %% [markdown]
# ## 7. Connect outputs, loss and official metric
#
# ResNet50-FPN adds a richer multiscale representation at a higher resource cost. It is a studied
# detector route, not a reproduction of the winning ensemble. Training returns losses; evaluation
# returns box dictionaries.

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
# ### Worked component — `local_checkpoint`
#
# Resolve a staged checkpoint; pretrained opt-in requires an explicit permitted local file.
#
# A checkpoint supplied with pretrained disabled can be the team's own trained state. The
# caller must record provenance; a local path alone does not establish competition permission.


# %% [worked] [stage-7]
def local_checkpoint(pretrained_allowed: bool, checkpoint_path: Path | str | None) -> str:
    """Resolve a staged checkpoint; pretrained opt-in requires an explicit permitted local file.

    A checkpoint supplied with pretrained disabled can be the team's own trained state. The
    caller must record provenance; a local path alone does not establish competition permission.
    """
    if pretrained_allowed and checkpoint_path is None:
        raise ValueError("Stage the exact permitted weights locally and set checkpoint_path")
    if checkpoint_path is None:
        return ""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


# %% [markdown]
# ### Worked component — `build_detection_model`
#
# Build a torchvision box detector and replace its classification head.
#
# Args:
#     num_classes: Total number of labels including background class 0.
#     architecture: ``fasterrcnn_resnet50_fpn_v2``, ``fasterrcnn_resnet50_fpn``, or
#         ``fasterrcnn_mobilenet_v3_large_fpn``.
#     pretrained_allowed: Require explicitly staged permitted weights when enabled.
#     checkpoint_path: Original permitted COCO state for initialization; matching task state
#         for reload.


# %% [worked] [stage-7]
def build_detection_model(
    num_classes: int,
    *,
    architecture: str = "fasterrcnn_resnet50_fpn_v2",
    pretrained_allowed: bool = False,
    checkpoint_path: Path | str | None = None,
) -> nn.Module:
    """Build a torchvision box detector and replace its classification head.

    Args:
        num_classes: Total number of labels including background class 0.
        architecture: ``fasterrcnn_resnet50_fpn_v2``, ``fasterrcnn_resnet50_fpn``, or
            ``fasterrcnn_mobilenet_v3_large_fpn``.
        pretrained_allowed: Require explicitly staged permitted weights when enabled.
        checkpoint_path: Original permitted COCO state for initialization; matching task state
            for reload.
    """
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    factories = {
        "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
        "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    }
    try:
        factory = factories[architecture]
    except KeyError as error:
        raise ValueError(
            f"Unknown detector {architecture!r}; choose {sorted(factories)}"
        ) from error
    checkpoint = local_checkpoint(pretrained_allowed, checkpoint_path)
    model = factory(weights=None, weights_backbone=None)
    if checkpoint and pretrained_allowed:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes)
    if checkpoint and not pretrained_allowed:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


# %% [markdown]
# ### Worked component — `train_detection_epoch`
#
# Train one epoch using the loss dictionary returned by torchvision detectors.


# %% [worked] [stage-7]
def train_detection_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "auto",
    time_budget: TimeBudget | None = None,
) -> float:
    """Train one epoch using the loss dictionary returned by torchvision detectors."""
    resolved = torch.device(
        device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(resolved).train()
    total = 0.0
    batches = 0
    for images, targets in loader:
        if time_budget and time_budget.should_stop:
            break
        images = [image.to(resolved) for image in images]
        targets = [{key: value.to(resolved) for key, value in target.items()} for target in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


# %% [worked] [stage-7]

# %% [markdown]
# ### Worked component — `wheat_score`
#
# Wheat reference convention: confidence-ordered greedy matching, mean .50:.05:.75.
#
# Absolute xyxy boxes; the linked winning implementation uses inclusive (+1) pixel areas.
# Expose that choice because other graders use continuous exclusive coordinates. Compare
# fixtures with the actual official grader before transferring this metric to a new contest.
# The task statement uses strict IoU comparisons and scores every no-target image zero.
# Set strict_iou=False only to compare with the linked winner's non-strict implementation.
# This averages TP/(TP+FP+FN) over thresholds then images, not confidence-integrated AP.


# %% [worked] [stage-7]
def wheat_score(
    targets: Sequence[ArrayLike],
    boxes: Sequence[ArrayLike],
    scores: Sequence[ArrayLike],
    *,
    score_threshold: float = 0.5,
    inclusive_pixels: bool = True,
    strict_iou: bool = True,
) -> float:
    """Wheat reference convention: confidence-ordered greedy matching, mean .50:.05:.75.

    Absolute xyxy boxes; the linked winning implementation uses inclusive (+1) pixel areas.
    Expose that choice because other graders use continuous exclusive coordinates. Compare
    fixtures with the actual official grader before transferring this metric to a new contest.
    The task statement uses strict IoU comparisons and scores every no-target image zero.
    Set strict_iou=False only to compare with the linked winner's non-strict implementation.
    This averages TP/(TP+FP+FN) over thresholds then images, not confidence-integrated AP.
    """
    if not len(targets) or not (len(targets) == len(boxes) == len(scores)):
        raise ValueError("Need one target, prediction and confidence array per image")
    image_scores = []
    offset = 1.0 if inclusive_pixels else 0.0
    for references, predictions, confidences in zip(targets, boxes, scores, strict=True):
        gt = np.asarray(references, dtype=float).reshape(-1, 4)
        predicted = np.asarray(predictions, dtype=float).reshape(-1, 4)
        confidence = np.asarray(confidences, dtype=float)
        if confidence.shape != (len(predicted),) or not np.isfinite(confidence).all():
            raise ValueError("One finite confidence per predicted box is required")
        for rectangles in (gt, predicted):
            if not np.isfinite(rectangles).all() or np.any(rectangles[:, 2:] <= rectangles[:, :2]):
                raise ValueError("Boxes must be finite positive-area xyxy rectangles")
        if not len(gt):
            image_scores.append(0.0)
            continue
        keep = np.argsort(-confidence, kind="stable")
        predicted = predicted[keep[confidence[keep] > score_threshold]]
        low = np.maximum(predicted[:, None, :2], gt[None, :, :2])
        high = np.minimum(predicted[:, None, 2:], gt[None, :, 2:])
        intersection = np.maximum(high - low + offset, 0).prod(axis=2)
        area_p = (predicted[:, 2:] - predicted[:, :2] + offset).prod(axis=1)
        area_y = (gt[:, 2:] - gt[:, :2] + offset).prod(axis=1)
        iou = intersection / np.maximum(area_p[:, None] + area_y[None, :] - intersection, 1e-12)
        threshold_scores = []
        for threshold in np.arange(50, 80, 5) / 100:
            available = np.ones(len(gt), dtype=bool)
            matched = 0
            for row in iou:
                matched_iou = row > threshold if strict_iou else row >= threshold
                candidates = np.where(available & matched_iou)[0]
                if len(candidates):
                    best = candidates[np.argmax(row[candidates])]
                    available[best] = False
                    matched += 1
            denominator = len(gt) + len(predicted) - matched
            threshold_scores.append(matched / denominator if denominator else 1.0)
        image_scores.append(np.mean(threshold_scores))
    return float(np.mean(image_scores))


# %% [worked] [stage-7]
if PRETRAINED_ALLOWED and not ASSET_PROVENANCE:
    raise ValueError("Record the exact permitted local detector weights")

# %% [markdown]
# ### Worked component — `make_model`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-7]
def make_model(initialize=False):
    network = build_detection_model(
        num_classes=2,
        architecture=MODEL_NAME,
        pretrained_allowed=PRETRAINED_ALLOWED if initialize else False,
        checkpoint_path=WEIGHT_PATH if initialize else None,
    )
    network.transform.min_size = (IMAGE_SIZE,)
    network.transform.max_size = IMAGE_SIZE
    return network


# %% [worked] [stage-7]
model = make_model(initialize=True).to(competition.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
MODEL_DESCRIPTION = MODEL_NAME

# %% [markdown]
# ### Worked component — `evaluate_boxes`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-7]
def evaluate_boxes(predictions, threshold):
    return wheat_score(
        truth_boxes,
        [item["boxes"].numpy() for item in predictions],
        [item["scores"].numpy() for item in predictions],
        score_threshold=threshold,
        inclusive_pixels=True,
    )


# %% [markdown]
# ### Worked component — `select_confidence`
#
# Input: validation box dictionaries, confidence grid and official scoring callback. Return
# scores, selected confidence and score. The callback owns the image truth and IoU convention.


# %% [worked] [stage-7]
def select_confidence(
    predictions: Sequence[dict], candidates: Sequence[float], score_fn: Any
) -> Any:
    """Input: validation box dictionaries, confidence grid and official scoring callback.
    Return scores, selected confidence and score. The callback owns the image truth and IoU
    convention.
    """
    scores_by_threshold = {threshold: score_fn(predictions, threshold) for threshold in candidates}
    selected_threshold = max(
        candidates, key=lambda value: (scores_by_threshold[value], -abs(value - 0.5))
    )
    validation_score = scores_by_threshold[selected_threshold]
    return scores_by_threshold, selected_threshold, validation_score


# %% [markdown]
# **Check before continuing.** Predict the result, run the small test below, and use its assertion
# message to locate an error. Expected: all assertions pass and the printed success message
# appears. These checks do not fit the real candidate.

# %% [check] [stage-7]
toy_predictions = [{"boxes": np.empty((0, 4)), "scores": np.array([])}]
toy_scores, toy_threshold, toy_score = select_confidence(
    toy_predictions, (0.2, 0.5, 0.8), lambda predictions, threshold: 1 - abs(threshold - 0.5)
)
assert toy_threshold == 0.5 and toy_score == 1
assert len(toy_scores) == 3
print("Confidence checks passed: selection uses the supplied validation scorer.")

# %% [markdown]
# ### Stop and explain
#
# Explain the output shape, loss and prediction decoding without reading the implementation. Which
# convention would silently produce an invalid score?
#
# <details>
# <summary>Reasoning checkpoint — open after writing your answer</summary>
#
# ResNet50-FPN adds a richer multiscale representation at a higher resource cost. It is a studied
# detector route, not a reproduction of the winning ensemble. Training returns losses; evaluation
# returns box dictionaries. Sweep a small confidence grid on held-out images only with the actual
# Wheat scoring convention. Duplicate boxes count as false positives. Confidence filtering is
# distinct from the IoU criterion for matching boxes.
#
# </details>

# %% [markdown]
# ## 8. Train and preserve a recoverable candidate
#
# Use the detector's FP32 loss dictionary and preserve the best validation-selected state
# atomically. An inferior epoch must not overwrite it. Reserve time for full-image inference and
# prediction-string validation.

# %% [worked] [stage-8]
started = perf_counter()
best = -float("inf")
checkpoint_path = OUTPUT / "best.pt"
for epoch in range(EPOCHS):
    if budget.should_stop:
        break
    loss = train_detection_epoch(
        model, train_loader, optimizer, device=competition.device, time_budget=budget
    )
    if budget.should_stop:
        break
    predictions = predict_detections(model, valid_loader, device=competition.device)
    value = evaluate_boxes(predictions, 0.5)
    if value > best:
        best = value
        save_torch_state(
            {"model": model.state_dict(), "epoch": epoch, "score": value}, checkpoint_path
        )
    print(epoch, "loss", loss, "wheat reference score", value)
if not checkpoint_path.exists():
    raise TimeoutError("No complete validation before the reserve; shorten the debug candidate")
model.load_state_dict(
    torch.load(checkpoint_path, map_location=competition.device, weights_only=True)["model"]
)

# %% [worked] [stage-8]
fit_seconds = perf_counter() - started
validation_predictions = predict_detections(model, valid_loader, device=competition.device)

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
# Sweep a small confidence grid on held-out images only with the actual Wheat scoring convention.
# Duplicate boxes count as false positives. Confidence filtering is distinct from the IoU
# criterion for matching boxes.

# %% [worked] [stage-9]
candidates = THRESHOLD_GRID
scores_by_threshold, selected_threshold, validation_score = select_confidence(
    validation_predictions, candidates, evaluate_boxes
)
print("Wheat reference score:", validation_score, scores_by_threshold)
print("No-box reference under the statement: 0.0")
errors = pd.DataFrame(
    {
        ID_COLUMN: va[ID_COLUMN],
        "true_boxes": [len(value) for value in truth_boxes],
        "predicted_boxes": [
            int((value["scores"] > selected_threshold).sum()) for value in validation_predictions
        ],
    }
)
print(errors.to_string(index=False))
errors.to_csv(OUTPUT / "validation_errors.csv", index=False)
plt.scatter(errors["true_boxes"], errors["predicted_boxes"])
plt.xlabel("True box count")
plt.ylabel("Predicted box count")
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
# Compare MobileNet and ResNet-FPN using the same image/source split and grader. Include full test
# inference time. Test resolution or confidence independently; do not introduce an ensemble before
# measuring its delivery cost.
#
# </details>

# %% [markdown]
# ## 10. Reload and infer every test item
#
# Rebuild the detector without redownloading initialization, load its best state and reproduce
# validation boxes. Predict all test image records, retain their identities and convert final xyxy
# boxes to output xywh.

# %% [worked] [stage-10]
started = perf_counter()
restored = make_model().to(competition.device)
restored.load_state_dict(
    torch.load(checkpoint_path, map_location=competition.device, weights_only=True)["model"],
    strict=True,
)
reloaded = predict_detections(restored, valid_loader, device=competition.device)
for expected, actual in zip(validation_predictions, reloaded, strict=True):
    np.testing.assert_allclose(expected["boxes"], actual["boxes"], rtol=1e-5, atol=1e-5)
test_predictions = predict_detections(
    restored, loader_for(test, labeled=False), device=competition.device
)

# %% [markdown]
# ### Worked component — `prediction_string`
#
# Run this supplied definition before its callers. Its inputs and invariants are explained in this
# section.


# %% [worked] [stage-10]
def prediction_string(prediction):
    values = []
    for score, box in zip(prediction["scores"], prediction["boxes"], strict=True):
        if float(score) <= selected_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        if not np.isfinite([float(score), x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid predicted box")
        values.append(f"{float(score):.6f} {x1:.4f} {y1:.4f} {x2 - x1:.4f} {y2 - y1:.4f}")
    return " ".join(values)


# %% [worked] [stage-10]
output_values = {"PredictionString": [prediction_string(value) for value in test_predictions]}
save_json(
    {
        "score_threshold": selected_threshold,
        "resize": IMAGE_SIZE,
        "architecture": MODEL_NAME,
        "box_input": "xyxy",
        "submission": "score x y width height",
        "metric": "confidence-ordered inclusive-pixel wheat reference; not COCO AP",
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
# Rebuild the detector without redownloading initialization, load its best state and reproduce
# validation boxes. Predict all test image records, retain their identities and convert final xyxy
# boxes to output xywh.
#
# </details>

# %% [markdown]
# ## 11. Deliver and verify the submission
#
# One image may contain many boxes, so annotation rows must first be grouped by image_id. The
# output is a prediction string of confidence and xywh boxes per image, including a valid empty
# string when no box survives.
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
        "image_size": IMAGE_SIZE,
        "model_name": MODEL_NAME,
        "imbalance_weights": USE_IMBALANCE_WEIGHTS,
        "pretrained_allowed": PRETRAINED_ALLOWED,
        "initial_weight_path": WEIGHT_PATH,
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
        "image_size": IMAGE_SIZE,
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
# A new table contains ten annotations per image and a source column. Demonstrate source-disjoint
# image splits, a batch with different box counts, and a valid empty prediction string before
# fitting.
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
# Compare MobileNet and ResNet-FPN using the same image/source split and grader. Include full test
# inference time. Test resolution or confidence independently; do not introduce an ensemble before
# measuring its delivery cost.
#
# A first correct submission is the readiness gate. Advanced complexity is selected only after an
# honest metric/runtime comparison; no test fitting or model-weight permission is implied.
