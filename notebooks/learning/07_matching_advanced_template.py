# %% [markdown]
# # Shopee Price Match Guarantee — advanced / SOTA study
#
# Problem family: **Multimodal product matching**. Official metric: **Mean per-query set F1,
# including self**.
#
# **Learning objective:** Permitted local image/text encoders, normalized modality fusion and
# validation threshold selection.
#
# [Dataset and original task](https://www.kaggle.com/competitions/shopee-product-matching) ·
# [Primary solution
# evidence](https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr)
#
# The first-place write-up emphasizes matching rules over marginal encoder gains and normalizing
# each modality before fusion. This lesson implements frozen local encoder extraction,
# normalize-then-concatenate matching and a small threshold comparison. It omits ArcFace
# training and iterative neighborhood blending.
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


FAMILY, VARIANT = "07_matching", "advanced"
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
GROUP_COLUMN, TIME_COLUMN = "label_group", None
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
# Stage the original train/test CSVs and train_images/test_images directories. This matching
# competition asks for a set of gallery posting IDs per query. The test gallery is used for
# pairwise inference; no test labels or pseudo-label training are used.
#

# %%
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
print("Shapes:", train.shape, test.shape, sample.shape)

if train["label_group"].eq("").any() or train["label_group"].isna().any():
    raise ValueError("Every training item needs a product-group identity")

# %% [markdown]
# ## 3. Inspect and visualize the data
#
# Write down three observations from these plots before revealing any solution: missingness,
# examples that look wrong, and a possible leakage unit. Never select experiments using hidden
# test labels.
#
#
#

# %%
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
# ## 4. Freeze a leakage-aware split
#
# Keep these exact folds for baseline/advanced comparisons. Random stratification is only
# appropriate for independent samples. Set a group or time column when the acquisition process
# demands it; assert the chosen boundary.
#
# A product-group holdout estimates generalization to unseen products. A random item split can
# expose the same product on both sides. This exercise always holds out label_group,
# independently of the generic GROUP_COLUMN field.
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


if TIME_COLUMN:
    raise ValueError("Use group holdout here; design temporal gallery evaluation separately")
# >>> exercise: hold out complete product groups, not random pairs or random items
split = make_split(train, group_column="label_group", valid_size=0.2, seed=SEED)
tr, va = split.train.copy(), split.valid.copy()
# <<< exercise
assert not set(tr["label_group"]) & set(va["label_group"])
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
print("Held-out product groups:", va["label_group"].nunique(), "Gallery items:", len(va))

# %% [markdown]
# ## 5. Audit imbalance and choose a response
#
# Measure prevalence on the training fold. Choose the response according to the official score:
# class weights, positive weights, coverage sampling or a log target are hypotheses. Compare
# with the unweighted candidate on the same split. Never rebalance validation.
#
# Most arbitrary item pairs are negatives. Pairwise accuracy would reward predicting no matches.
# Mean per-query set F1 and group-aware validation match the actual decision problem; a later
# supervised metric-learning sampler should reflect product identities.
#

# %%
# >>> exercise: measure group-size imbalance, which determines positive/negative pair imbalance
sizes = tr.groupby("label_group").size()
positive_pairs = int((sizes * (sizes - 1) / 2).sum())
all_pairs = len(tr) * (len(tr) - 1) // 2
# <<< exercise
print("Positive pairs:", positive_pairs, "All pairs:", all_pairs)
print("Response: group holdout and per-query F1; keep validation unchanged")
print("For metric learning later: sample product groups, not all pairs equally.")
sizes.hist(bins=20)
plt.title("Training product-group sizes")
plt.savefig(OUTPUT / "imbalance.png")
plt.show()
plt.close()

# %% [markdown]
# ## 6. Build features or datasets
#
# Inspect one transformed example or batch. Explain its shape, range, label encoding and
# train/validation/test consistency. Fitted preprocessing sees the training fold only.
#
# Two supported asset paths: local text model/tokenizer plus a local timm image checkpoint, or a
# precomputed NPZ with ids [N], image [N,Di], text [N,Dt], and fit_ids. Frozen encoders must not
# have been trained on this lesson's validation/test labels. Cache provenance is a declaration
# to audit, not automatic proof. An empty fit_ids means no task-specific encoder fitting. No
# asset is fetched.
#

# %%
from sklearn.preprocessing import normalize

feature_started = perf_counter()
if not PRETRAINED_ALLOWED or not ASSET_PROVENANCE:
    raise ValueError(
        "Advanced learned embeddings require explicit permission and exact local asset provenance"
    )
gallery = pd.concat([tr, va, test], ignore_index=True)
if EMBEDDING_ARCHIVE is not None:
    asset = np.load(EMBEDDING_ARCHIVE, allow_pickle=False)
    ids = asset["ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Embedding IDs must be unique")
    # Declare task-specific encoder fitting IDs; frozen external encoders use an empty array.
    if not set(asset["fit_ids"].astype(str)) <= set(tr[ID_COLUMN]):
        raise ValueError("Encoder was fitted outside this training fold: validation/test leakage")
    order = pd.Index(ids).get_indexer(gallery[ID_COLUMN])
    if (order < 0).any():
        raise ValueError("Embedding cache does not cover every gallery ID")
    image_features, text_features = asset["image"][order], asset["text"][order]
else:
    if TEXT_MODEL_DIR is None or IMAGE_MODEL_CHECKPOINT is None:
        raise ValueError(
            "Stage permitted local encoders/tokenizer, or a provenance-checked NPZ cache"
        )
    from timm.data import resolve_model_data_config
    from torch.utils.data import Dataset
    from torchvision.transforms import v2
    from transformers import AutoModel, AutoTokenizer

    class ImageTableDataset(Dataset[Any]):
        """Load RGB images and optional class indexes from a pandas annotation table."""

        def __init__(
            self,
            frame: pd.DataFrame,
            *,
            image_column: str,
            target_column: str | None = None,
            root: Path | str = ".",
            transform: Callable[[Image.Image], torch.Tensor] | None = None,
            label_to_index: Mapping[object, int] | None = None,
        ) -> None:
            self.frame = frame.reset_index(drop=True)
            self.image_column = image_column
            self.target_column = target_column
            self.root = Path(root)
            self.transform = transform or build_image_transforms(training=False)
            self.label_to_index = dict(label_to_index or {})

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int) -> Any:
            row = self.frame.iloc[index]
            path = self.root / str(row[self.image_column])
            with Image.open(path) as source:
                image = source.convert("RGB")
            tensor = self.transform(image)
            if self.target_column:
                label = row[self.target_column]
                target = torch.tensor(self.label_to_index.get(label, label), dtype=torch.long)
                return tensor, target
            return tensor

    def build_image_transforms(
        *,
        size: int = 224,
        training: bool,
        normalize: bool = True,
    ) -> Callable[[Image.Image], torch.Tensor]:
        """Build conservative training or deterministic validation transforms."""
        operations: list[Any]
        if training:
            operations = [
                v2.RandomResizedCrop((size, size), scale=(0.7, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ]
        else:
            operations = [v2.Resize((size, size))]
        operations.append(v2.ToImage())
        operations.append(v2.ToDtype(torch.float32, scale=True))
        if normalize:
            operations.append(v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return v2.Compose(operations)

    import timm

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

    def timm_weight_options(
        pretrained_allowed: bool, checkpoint_path: Path | str | None
    ) -> dict[str, Any]:
        """Load allowed backbone weights from a local file with timm's classifier-head adaptation.

        Empty remote/source fields force the local-file loading path. With pretrained disabled, a
        local file is treated as an exact matching checkpoint produced by the team.
        """
        checkpoint = local_checkpoint(pretrained_allowed, checkpoint_path)
        if pretrained_allowed:
            return {
                "pretrained": True,
                "pretrained_cfg_overlay": {
                    "file": checkpoint,
                    "source": "",
                    "url": None,
                    "hf_hub_id": None,
                },
            }
        return {"pretrained": False, "checkpoint_path": checkpoint}

    def build_image_classifier(
        model_name: str,
        num_classes: int,
        *,
        pretrained_allowed: bool = False,
        checkpoint_path: Path | str | None = None,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build a timm classifier with random or explicitly permitted local weights."""
        model = timm.create_model(
            model_name,
            **timm_weight_options(pretrained_allowed, checkpoint_path),
            num_classes=num_classes,
            drop_rate=dropout,
        )
        return model

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR, local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(TEXT_MODEL_DIR, local_files_only=True)
        .to(competition.device)
        .eval()
    )
    text_chunks = []
    with torch.inference_mode():
        for start in range(0, len(gallery), BATCH_SIZE):
            batch = tokenizer(
                gallery["title"].iloc[start : start + BATCH_SIZE].tolist(),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {key: value.to(competition.device) for key, value in batch.items()}
            hidden = text_encoder(**batch).last_hidden_state
            attention = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention).sum(1) / attention.sum(1).clamp_min(1)
            text_chunks.append(pooled.float().cpu().numpy())
    text_features = np.concatenate(text_chunks)
    image_encoder = build_image_classifier(
        MODEL_NAME, 0, pretrained_allowed=True, checkpoint_path=IMAGE_MODEL_CHECKPOINT
    )
    model_data = resolve_model_data_config(image_encoder)
    transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=model_data["mean"], std=model_data["std"]),
        ]
    )
    dataset = ImageTableDataset(gallery, image_column="_image", root=DATA_DIR, transform=transform)
    image_features = predict_logits(
        image_encoder,
        DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        device=competition.device,
    )
    # Save the complete encoders for inference on future private inputs, not just today's cache.
    text_encoder.save_pretrained(OUTPUT / "text_encoder")
    tokenizer.save_pretrained(OUTPUT / "text_encoder")
    save_torch_state(image_encoder.state_dict(), OUTPUT / "image_encoder.pt")
    save_json(
        {
            "image_model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "mean": model_data["mean"],
            "std": model_data["std"],
            "max_length": MAX_LENGTH,
        },
        OUTPUT / "encoder_recipe.json",
    )
    ids = gallery[ID_COLUMN].to_numpy(dtype=str)
    asset = {
        "ids": ids,
        "image": image_features,
        "text": text_features,
        "fit_ids": np.asarray([], dtype=str),
    }
for matrix in (image_features, text_features):
    if matrix.ndim != 2 or len(matrix) != len(gallery) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite [items, embedding dimensions] features")
if np.any(np.linalg.norm(image_features, axis=1) == 0) or np.any(
    np.linalg.norm(text_features, axis=1) == 0
):
    raise ValueError("Every gallery item needs nonzero image and text features")
feature_ids = pd.Index(gallery[ID_COLUMN])
image_features, text_features = normalize(image_features), normalize(text_features)
feature_seconds = perf_counter() - feature_started
print(
    "Embedding shapes and extraction seconds:",
    image_features.shape,
    text_features.shape,
    feature_seconds,
)

# %% [markdown]
# ## 7. Define the model and loss
#
# State why the output dimension and loss fit this prediction contract. First verify the
# simplest model. Pretrained initialization is disabled unless the exact local asset is
# permitted.
#
# The winner's normalize-each-modality-then-fuse idea is the studied component. These frozen
# encoders are not its ArcFace-trained model ensemble. Test image/text-only weights alongside
# fusion; more modalities need not improve F1.
#

# %%
from sklearn.metrics.pairwise import cosine_similarity


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


def features_for(frame, weight):
    order = feature_ids.get_indexer(frame[ID_COLUMN])
    if (order < 0).any():
        raise ValueError("No cached features for a requested ID; encode the new gallery first")
    # Normalize each modality before concatenating; squared scale gives mixture weight.
    return normalize(
        np.concatenate(
            [
                np.sqrt(weight) * image_features[order],
                np.sqrt(1 - weight) * text_features[order],
            ],
            axis=1,
        )
    )


MODEL_DESCRIPTION = "frozen local image/text normalized embedding fusion"

# %% [markdown]
# ## 8. Train and preserve the candidate
#
# First fit a tiny subset as a debugging drill; rebuild before the real-fold experiment. Main
# torch lessons define their small training loop directly below, including precision, time
# reserves and atomic best checkpoints. For sklearn fit the full preprocessing pipeline together.
#
# Extraction time is included in fit_seconds so a cached run cannot masquerade as zero-cost
# improvement. The source-encoder path saves weights/tokenizer/config for future inputs. A
# cache-only path covers only its recorded IDs and cannot independently embed new private
# images/titles.
#

# %%
started = perf_counter()
# No supervised fitting on validation/test: encoders were frozen before this lesson.
temporary = OUTPUT / "features.npz.tmp"
with temporary.open("wb") as handle:
    np.savez_compressed(
        handle,
        ids=gallery[ID_COLUMN].to_numpy(dtype=str),
        image=image_features,
        text=text_features,
        fit_ids=asset["fit_ids"],
    )
checkpoint_path = temporary.replace(OUTPUT / "features.npz")
fit_seconds = feature_seconds + perf_counter() - started

# %% [markdown]
# ## 9. Validate, inspect errors and compare cost
#
# Use all validation predictions for the exact score. Inspect individual mistakes and
# minority/group performance; the diagnostic metric does not replace the official metric. Keep a
# score/runtime ledger before choosing more complexity.
#
# The bounded grid chooses thresholds/mixture weights on validation only. This reuse makes tiny
# gains optimistic; check a second product-group split when worthwhile. Preserve the baseline
# comparison and include full embedding extraction/inference cost.
#

# %%
selection_started = perf_counter()
weights_to_compare = (0.0, 0.5, 1.0)
thresholds = THRESHOLD_GRID
comparisons = []
for weight in weights_to_compare:
    for threshold in thresholds:
        budget.checkpoint("validation matching comparison")
        features = features_for(va, weight)
        positions = match_positions(features, threshold)
        # >>> exercise: score complete match sets including self, not pairwise accuracy
        value = matching_f1(va["label_group"].tolist(), positions)
        # <<< exercise
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
# >>> exercise: restore normalized features and preserve their explicit ID order
restored_asset = np.load(checkpoint_path, allow_pickle=False)
feature_ids = pd.Index(restored_asset["ids"].astype(str))
image_features, text_features = restored_asset["image"], restored_asset["text"]
assert (
    match_positions(features_for(va, selected_weight), selected_threshold) == validation_positions
)
test_positions = match_positions(features_for(test, selected_weight), selected_threshold)
# <<< exercise
output_values = {
    "matches": [" ".join(test.iloc[positions][ID_COLUMN].tolist()) for positions in test_positions]
}
for index, positions in enumerate(test_positions):
    assert index in positions and len(positions) == len(set(positions))
save_json(
    {
        "threshold": selected_threshold,
        "image_weight": selected_weight,
        "asset_provenance": ASSET_PROVENANCE,
        "frozen_encoders": True,
        "cache_scope": "recorded IDs only; new inputs need encoders or a new permitted cache",
    },
    OUTPUT / "recipe.json",
)
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
