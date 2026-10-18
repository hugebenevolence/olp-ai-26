# %% [markdown]
# # TGS Salt Identification — fill-in baseline
#
# Problem family: **Binary semantic segmentation**. Official metric: **Mean per-image hits at
# strict IoU thresholds .50 to .95**.
#
# **Learning objective:** Small dense convolutional mask predictor + BCE.
#
# [Dataset and original
# task](https://www.kaggle.com/competitions/tgs-salt-identification-challenge) · [Primary
# solution evidence](https://github.com/ybabakhin/kaggle_salt_bes_phalanx)
#
# The winning repository and authors' paper describe encoder-decoder ensembles and
# semi-supervised training. This lesson isolates skip-connected segmentation and threshold
# selection. It omits pseudo-labeling and large pretrained ensembles; it is not an exact winner
# reproduction.
#
# **Watch for:** TGS is neither Dice nor continuous IoU. Restore the original 101x101 dimensions
# before scoring and column-major RLE.
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


FAMILY, VARIANT = "05_salt", "baseline"
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
ID_COLUMN, TARGET_COLUMNS = "id", ["rle_mask"]
train = pd.read_csv(DATA_DIR / "train.csv", converters={ID_COLUMN: str})
sample = pd.read_csv(DATA_DIR / "sample_submission.csv", converters={ID_COLUMN: str})
test = sample[[ID_COLUMN]].copy()
train["image"] = "train/images/" + train[ID_COLUMN] + ".png"
train["mask"] = "train/masks/" + train[ID_COLUMN] + ".png"
test["image"] = "test/images/" + test[ID_COLUMN] + ".png"


def read_mask(path):
    with Image.open(DATA_DIR / path) as image:
        return np.asarray(image.convert("L")) > 0


train["_coverage"] = [read_mask(path).mean() for path in train["mask"]]
train["_has_salt"] = (train["_coverage"] > 0).astype(int)

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
# Mask coverage exposes two imbalances: empty images versus nonempty images, and background
# pixels versus foreground pixels. Inspect both; a high background pixel accuracy would be
# misleading.
#

# %% [exercise]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for column, (_, row) in enumerate(train.head(4).iterrows()):
    with Image.open(DATA_DIR / row["image"]) as image:
        axes[0, column].imshow(image.convert("RGB"))
    axes[1, column].imshow(read_mask(row["mask"]), cmap="gray", vmin=0, vmax=1)
    axes[0, column].set_title(f"coverage={row['_coverage']:.3f}")
    axes[0, column].axis("off")
    axes[1, column].axis("off")
plt.tight_layout()
fig.savefig(OUTPUT / "inspection.png")
plt.show()
plt.close(fig)
print("Empty fraction:", (train["_coverage"] == 0).mean())
print("Coverage quantiles:", train["_coverage"].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict())

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


# >>> exercise: preserve empty/nonempty support while honoring any acquisition groups
split = make_split(
    train,
    target_columns="_has_salt",
    group_column=GROUP_COLUMN,
    time_column=TIME_COLUMN,
    seed=SEED,
    valid_size=0.2,
)
tr, va = split.train.copy(), split.valid.copy()
# <<< exercise
assert not set(tr[ID_COLUMN]) & set(va[ID_COLUMN])
if GROUP_COLUMN:
    assert not set(tr[GROUP_COLUMN]) & set(va[GROUP_COLUMN])
if TIME_COLUMN:
    assert pd.to_datetime(tr[TIME_COLUMN]).max() < pd.to_datetime(va[TIME_COLUMN]).min()
print("Validation empty/nonempty:", va["_has_salt"].value_counts().to_dict())

# %% [markdown]
# ## 5. Audit imbalance and choose a response
#
# Measure prevalence on the training fold. Choose the response according to the official score:
# class weights, positive weights, coverage sampling or a log target are hypotheses. Compare
# with the unweighted candidate on the same split. Never rebalance validation.
#
# The optional response weights positive pixels in BCE. Compare weighted versus unweighted runs
# on these same validation masks. Dice in the advanced model is another training surrogate;
# neither training loss replaces the TGS evaluator.
#

# %% [exercise]
# >>> exercise: compute pixel imbalance only from the training masks
positive = sum(read_mask(path).sum() for path in tr["mask"])
total = sum(read_mask(path).size for path in tr["mask"])
if positive == 0:
    raise ValueError("No foreground training pixels: inspect the split and supervision")
pixel_weight = torch.tensor((total - positive) / positive, dtype=torch.float32)
# <<< exercise
print("Training positive pixel fraction:", positive / total)
print(
    "Empty mask fraction:", (tr["_coverage"] == 0).mean(), "Weight enabled:", USE_IMBALANCE_WEIGHTS
)
tr["_coverage"].hist(bins=20)
plt.title("Training foreground coverage")
plt.savefig(OUTPUT / "imbalance.png")
plt.show()
plt.close()

# %% [markdown]
# ## 6. Build features or datasets
#
# Inspect one transformed example or batch. Explain its shape, range, label encoding and
# train/validation/test consistency. Fitted preprocessing sees the training fold only.
#
# Training and mask transforms share geometry; masks use nearest-neighbor interpolation in the
# inline dataset. There is no ImageNet normalization in this scratch route. Inspect image/mask
# alignment after transforms, not only the raw pair.
#

# %% [exercise]
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class SegmentationTableDataset(Dataset[Any]):
    """Read paired RGB images and indexed/binary masks described by a table.

    The same random transform is applied to the image and mask. Without a mask column the dataset
    returns only an image tensor, which is suitable for test inference.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str,
        mask_column: str | None = None,
        root: Path | str = ".",
        size: int = 256,
        training: bool = False,
        binary_mask: bool = True,
        normalize: bool = True,
        foreground_values: Sequence[int] | None = None,
        ignore_index: int | None = None,
    ) -> None:
        """Store table/path settings and create paired torchvision transforms."""
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.mask_column = mask_column
        self.root = Path(root)
        self.binary_mask = binary_mask
        self.normalize = normalize
        self.foreground_values = foreground_values
        self.ignore_index = ignore_index
        operations: list[Callable] = [v2.Resize((size, size), antialias=True)]
        if training:
            operations += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(p=0.2)]
        self.spatial_transform = v2.Compose(operations)

    def __len__(self) -> int:
        """Return the number of rows/examples."""
        return len(self.frame)

    def __getitem__(self, index: int) -> Any:
        """Load one image and, when configured, its spatially aligned mask."""
        row = self.frame.iloc[index]
        with Image.open(self.root / str(row[self.image_column])) as source:
            image = v2.functional.to_image(source.convert("RGB"))
        if not self.mask_column:
            image = self.spatial_transform(image)
            image = v2.functional.to_dtype(image, torch.float32, scale=True)
            if self.normalize:
                image = v2.functional.normalize(
                    image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
            return image
        with Image.open(self.root / str(row[self.mask_column])) as source:
            mask = tv_tensors.Mask(v2.functional.to_image(source.convert("L")))
        image, mask = self.spatial_transform(image, mask)
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        if self.normalize:
            image = v2.functional.normalize(
                image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        if self.binary_mask:
            ignored = mask == self.ignore_index if self.ignore_index is not None else None
            mask = (
                torch.isin(mask, torch.tensor(self.foreground_values, dtype=mask.dtype))
                if self.foreground_values is not None
                else mask > 0
            ).float()
            if ignored is not None:
                mask[ignored] = self.ignore_index
        else:
            mask = mask.to(torch.long).squeeze(0)
        return image, mask


def loader_for(frame, training=False):
    dataset = SegmentationTableDataset(
        frame,
        image_column="image",
        mask_column="mask" if "mask" in frame else None,
        root=DATA_DIR,
        size=IMAGE_SIZE,
        training=training,
        normalize=False,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=training, num_workers=0)


# >>> exercise: keep validation/test transforms deterministic and mask geometry paired
train_loader, valid_loader = loader_for(tr, False), loader_for(va)
# <<< exercise
pixels, masks = next(iter(train_loader))
assert masks.shape == (len(pixels), 1, IMAGE_SIZE, IMAGE_SIZE)
assert set(masks.unique().tolist()) <= {0.0, 1.0}
print("Batch:", pixels.shape, masks.shape, "Pixels:", pixels.min().item(), pixels.max().item())

# %% [markdown]
# ## 7. Define the model and loss
#
# State why the output dimension and loss fit this prediction contract. First verify the
# simplest model. Pretrained initialization is disabled unless the exact local asset is
# permitted.
#
# A dense predictor produces a foreground logit at every pixel. It can establish a complete
# route quickly but has limited context; U-Net is the next model to study.
#

# %% [exercise]
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    """Weighted sum of binary cross entropy and soft Dice loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        *,
        ignore_index: int | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        """Create the loss; ``dice_weight=1`` selects pure soft Dice."""
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError("dice_weight must be between 0 and 1")
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss from raw logits and binary masks of the same shape."""
        if logits.shape != targets.shape:
            targets = F.interpolate(targets.float(), logits.shape[-2:], mode="nearest")
        keep = (
            torch.ones_like(targets, dtype=torch.bool)
            if self.ignore_index is None
            else targets != self.ignore_index
        )
        targets = targets.float().masked_fill(~keep, 0)
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.bce.pos_weight, reduction="none"
        )
        bce = (pixel_loss * keep).sum() / keep.sum().clamp_min(1)
        probabilities = logits.sigmoid() * keep
        axes = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=axes)
        dice = (2 * intersection + 1e-7) / (
            probabilities.sum(dim=axes) + targets.sum(dim=axes) + 1e-7
        )
        return (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice.mean())


def make_model():
    # >>> exercise: output one raw foreground logit for every input pixel
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 1),
    )
    # <<< exercise


model = make_model()
criterion = DiceBCELoss(dice_weight=0, pos_weight=pixel_weight if USE_IMBALANCE_WEIGHTS else None)
criterion = criterion.to(competition.device)
MODEL_DESCRIPTION = "small dense scratch mask predictor + BCE"

# %% [markdown]
# ## 8. Train and preserve the candidate
#
# First fit a tiny subset as a debugging drill; rebuild before the real-fold experiment. Main
# torch lessons define their small training loop directly below, including precision, time
# reserves and atomic best checkpoints. For sklearn fit the full preprocessing pipeline together.
#
# The base loop's ordinary resized-mask metric would be wrong for this task. The subclass
# selects with original-resolution TGS score at .5. The later bounded threshold comparison is
# performed on the saved candidate, not on test data.
#

# %% [exercise]


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


from numpy.typing import ArrayLike


def salt_score(truth: ArrayLike, prediction: ArrayLike) -> float:
    """TGS single binary mask per image: mean hits at strict IoU thresholds .50:.05:.95.

    Inputs are equally shaped [N,H,W] binary masks. Both empty scores one, exactly one empty
    scores zero. This is not pixel Dice, continuous mean IoU, or instance COCO average precision.
    """
    y, p = np.asarray(truth), np.asarray(prediction)
    if y.ndim != 3 or y.shape != p.shape or not y.size:
        raise ValueError("Expected aligned nonempty [N,H,W] masks at original resolution")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Threshold probability maps before scoring binary masks")
    intersection = np.logical_and(y, p).sum(axis=(1, 2))
    union = np.logical_or(y, p).sum(axis=(1, 2))
    iou = np.divide(intersection, union, out=np.ones(len(y)), where=union != 0)
    thresholds = np.arange(50, 100, 5) / 100
    return float((iou[:, None] > thresholds).mean())


def probability_maps(network, frame):
    logits = predict_logits(
        network, loader_for(frame.drop(columns="mask", errors="ignore")), device=competition.device
    )
    probabilities = []
    for row_index, path in enumerate(frame["image"]):
        with Image.open(DATA_DIR / path) as image:
            width, height = image.size
        restored = F.interpolate(
            torch.from_numpy(logits[row_index : row_index + 1]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        probabilities.append(restored.sigmoid().numpy()[0, 0])
    return np.stack(probabilities)


references = np.stack([read_mask(path) for path in va["mask"]])


class SaltTrainer(Trainer):
    def evaluate(self, loader):
        values = super().evaluate(loader)
        # >>> exercise: choose checkpoints with the original-resolution task score
        probabilities = probability_maps(self.model, va)
        values["metric"] = salt_score(references, probabilities > 0.5)
        # <<< exercise
        return values


started = perf_counter()
training = TrainerConfig(
    epochs=EPOCHS,
    learning_rate=1e-3,
    metric_name="tgs_salt",
    mixed_precision="auto",
    scheduler="none",
    patience=2,
)
trainer = SaltTrainer(
    model,
    criterion,
    training,
    output_dir=OUTPUT,
    device=competition.device,
    prediction_decoder=torch.sigmoid,
    time_budget=budget,
    backup_dir=DURABLE_OUTPUT,
)
history = trainer.fit(train_loader, valid_loader)
fit_seconds = perf_counter() - started
checkpoint_path = trainer.checkpoint_path
validation_probabilities = probability_maps(model, va)

# %% [markdown]
# ## 9. Validate, inspect errors and compare cost
#
# Use all validation predictions for the exact score. Inspect individual mistakes and
# minority/group performance; the diagnostic metric does not replace the official metric. Keep a
# score/runtime ledger before choosing more complexity.
#
# Both empty masks score one, one empty mask scores zero, and an IoU exactly equal to a cutoff
# is not a hit under this TGS convention. Threshold selection reuses validation, so treat small
# gains cautiously and confirm on a separate practice pack.
#

# %% [exercise]
candidates = (0.5,)
# >>> exercise: select any threshold on validation only using the actual TGS score
scores_by_threshold = {
    value: salt_score(references, validation_probabilities > value) for value in candidates
}
selected_threshold = max(
    candidates, key=lambda value: (scores_by_threshold[value], -abs(value - 0.5))
)
validation_score = scores_by_threshold[selected_threshold]
# <<< exercise
print("Original-resolution TGS score:", validation_score, "Thresholds:", scores_by_threshold)
print("Empty-mask reference:", salt_score(references, np.zeros_like(references)))
diagnostic_dice = evaluate_metric(
    "dice_mean_image", references, validation_probabilities > selected_threshold
)
print("Diagnostic Dice (not the official score):", diagnostic_dice)
errors = pd.DataFrame(
    {
        ID_COLUMN: va[ID_COLUMN],
        "salt_score": [
            salt_score(y[None], (p > selected_threshold)[None])
            for y, p in zip(references, validation_probabilities, strict=True)
        ],
        "empty_truth": ~references.any(axis=(1, 2)),
    }
)
print(errors.sort_values("salt_score").head(15).to_string(index=False))
errors.to_csv(OUTPUT / "validation_errors.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(references[0], cmap="gray")
axes[0].set_title("True mask")
axes[1].imshow(validation_probabilities[0], vmin=0, vmax=1)
axes[1].set_title("Foreground probability")
axes[2].imshow(validation_probabilities[0] > selected_threshold, cmap="gray")
axes[2].set_title("Submitted binary mask")
fig.savefig(OUTPUT / "validation.png")
plt.show()
plt.close(fig)

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
def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask as one-indexed, column-major run-length encoding."""
    pixels = np.asarray(mask, dtype=np.uint8).T.reshape(-1)
    padded = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(value) for value in changes)


def rle_to_mask(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode one-indexed, column-major RLE into a ``uint8`` mask of ``(height, width)``."""
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or not str(rle).strip():
        return np.zeros(shape, dtype=np.uint8)
    values = np.asarray([int(value) for value in str(rle).split()], dtype=np.int64)
    if len(values) % 2:
        raise ValueError("RLE must contain start/length pairs")
    starts = values[::2] - 1
    lengths = values[1::2]
    if (starts < 0).any() or (lengths < 1).any() or (starts + lengths > shape[0] * shape[1]).any():
        raise ValueError("RLE runs exceed mask bounds or have invalid start/length")
    if len(starts) > 1 and (starts[1:] < (starts + lengths)[:-1]).any():
        raise ValueError("RLE runs must be ordered and nonoverlapping")
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths, strict=True):
        flat[start : start + length] = 1
    return flat.reshape((shape[1], shape[0])).T


started = perf_counter()
# >>> exercise: restore the best state and encode original-size masks in column-major RLE
restored = make_model().to(competition.device)
restored.load_state_dict(
    torch.load(checkpoint_path, map_location=competition.device, weights_only=True)["model"],
    strict=True,
)
np.testing.assert_allclose(
    probability_maps(restored, va), validation_probabilities, rtol=1e-5, atol=1e-6
)
test_masks = probability_maps(restored, test) > selected_threshold
encoded = [mask_to_rle(mask) for mask in test_masks]
# <<< exercise
for mask, text in zip(test_masks, encoded, strict=True):
    np.testing.assert_array_equal(rle_to_mask(text, mask.shape), mask)
output_values = {"rle_mask": encoded}
save_json(
    {
        "threshold": selected_threshold,
        "image_size": IMAGE_SIZE,
        "normalization": "RGB / 255",
        "original_shape": test_masks.shape[1:],
        "rle": "one-based column-major",
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
