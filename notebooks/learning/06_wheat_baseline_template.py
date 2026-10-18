# %% [markdown]
# # Global Wheat Detection — fill-in baseline
#
# Problem family: **Object detection**. Official metric: **Mean per-image TP/(TP+FP+FN), IoU .50
# to .75**.
#
# **Learning objective:** Scratch Faster R-CNN with MobileNet FPN.
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


FAMILY, VARIANT = "06_wheat", "baseline"
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
USE_HISTORY_FEATURES = False
MODEL_NAME = "fasterrcnn_mobilenet_v3_large_fpn"
IMAGE_SUFFIX = ".jpg"
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
# Global Wheat's CSV contains one row per box. Parse bbox with literal_eval, never eval. The
# prediction unit is one image; create the image manifest before the split. This adapter covers
# the annotated-image training schema; add an explicit image manifest/empty targets for a
# dataset with unannotated negative images.
#

# %% [exercise]
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
print(
    "Box-size quantiles:",
    pd.DataFrame(parsed[:, 2:], columns=["width", "height"]).quantile([0, 0.5, 0.9, 1]),
)

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


# >>> exercise: split unique image records, keeping acquisition sources together
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
print("Image counts:", len(tr), len(va), "Validation sources:", va["source"].unique())

# %% [markdown]
# ## 5. Audit imbalance and choose a response
#
# Measure prevalence on the training fold. Choose the response according to the official score:
# class weights, positive weights, coverage sampling or a log target are hypotheses. Compare
# with the unweighted candidate on the same split. Never rebalance validation.
#
# Detection imbalance includes acquisition source, object count and small-object size. This
# optional sampler balances coarse density bins on training images only. Do not rebalance
# validation or claim a single foreground label needs artificial class weights.
#

# %% [exercise]
from torch.utils.data import WeightedRandomSampler

# >>> exercise: measure image-density imbalance and derive optional training-only sampling weights
density_bins = pd.cut(tr["box_count"], bins=[-1, 2, 10, 50, 1e9])
frequency = density_bins.value_counts()
sampling_weights = (1 / density_bins.map(frequency).astype(float)).to_numpy()
# <<< exercise
print("One foreground class; no invented foreground class-balancing weights")
print("Density bins:", frequency.to_dict(), "Sampling enabled:", USE_IMBALANCE_WEIGHTS)
tr["box_count"].hist(bins=20)
plt.title("Training boxes per image")
plt.savefig(OUTPUT / "imbalance.png")
plt.show()
plt.close()

# %% [markdown]
# ## 6. Build features or datasets
#
# Inspect one transformed example or batch. Explain its shape, range, label encoding and
# train/validation/test consistency. Fitted preprocessing sees the training fold only.
#
# Native detector transforms resize images and boxes together and restore boxes to original
# coordinates at inference. Do not independently resize only images. The manifest-order merge
# prevents attaching valid predictions to the wrong IDs.
#

# %% [exercise]
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


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


def detection_collate(batch: list[tuple[Any, Any]]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Collate variable-size images and variable-length detection targets."""
    return tuple(zip(*batch, strict=True))


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


# >>> exercise: collate variable-length box targets without stacking their box dimension
train_loader, valid_loader = loader_for(tr, True), loader_for(va)
# <<< exercise
images, targets = next(iter(train_loader))
assert images[0].shape[0] == 3 and targets[0]["boxes"].shape[1] == 4
print("Images in batch:", len(images), "First target boxes:", targets[0]["boxes"].shape)
truth_boxes = [
    annotations.loc[annotations[ID_COLUMN] == key, ["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    for key in va[ID_COLUMN]
]

# %% [markdown]
# ## 7. Define the model and loss
#
# State why the output dimension and loss fit this prediction contract. First verify the
# simplest model. Pretrained initialization is disabled unless the exact local asset is
# permitted.
#
# A supported detector library gives a real multiple-box baseline in a few model-construction
# lines. It is still essential to understand its target dictionaries, class-zero background and
# loss dictionary.
#


# %% [exercise]
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


from numpy.typing import ArrayLike


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


if PRETRAINED_ALLOWED and not ASSET_PROVENANCE:
    raise ValueError("Record the exact permitted local detector weights")


def make_model(initialize=False):
    # >>> exercise: include background and disable all implicit weight downloads
    network = build_detection_model(
        num_classes=2,
        architecture=MODEL_NAME,
        pretrained_allowed=PRETRAINED_ALLOWED if initialize else False,
        checkpoint_path=WEIGHT_PATH if initialize else None,
    )
    # <<< exercise
    network.transform.min_size = (IMAGE_SIZE,)
    network.transform.max_size = IMAGE_SIZE
    return network


model = make_model(initialize=True).to(competition.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
MODEL_DESCRIPTION = MODEL_NAME


def evaluate_boxes(predictions, threshold):
    return wheat_score(
        truth_boxes,
        [item["boxes"].numpy() for item in predictions],
        [item["scores"].numpy() for item in predictions],
        score_threshold=threshold,
        inclusive_pixels=True,
    )


# %% [markdown]
# ## 8. Train and preserve the candidate
#
# First fit a tiny subset as a debugging drill; rebuild before the real-fold experiment. Main
# torch lessons define their small training loop directly below, including precision, time
# reserves and atomic best checkpoints. For sklearn fit the full preprocessing pipeline together.
#
# Torchvision detection returns a dictionary of losses during training and a list of box
# predictions in eval mode, so this route defines detector epoch/predict functions above instead
# of the classifier Trainer. It uses FP32. The bounded dataset/resolution is for first-output
# practice; measure real GPU time before expanding.
#

# %% [exercise]
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
    # >>> exercise: retain the best validation-selected model with an atomic write
    if value > best:
        best = value
        save_torch_state(
            {"model": model.state_dict(), "epoch": epoch, "score": value}, checkpoint_path
        )
    # <<< exercise
    print(epoch, "loss", loss, "wheat reference score", value)
if not checkpoint_path.exists():
    raise TimeoutError("No complete validation before the reserve; shorten the debug candidate")
model.load_state_dict(
    torch.load(checkpoint_path, map_location=competition.device, weights_only=True)["model"]
)
fit_seconds = perf_counter() - started
validation_predictions = predict_detections(model, valid_loader, device=competition.device)

# %% [markdown]
# ## 9. Validate, inspect errors and compare cost
#
# Use all validation predictions for the exact score. Inspect individual mistakes and
# minority/group performance; the diagnostic metric does not replace the official metric. Keep a
# score/runtime ledger before choosing more complexity.
#
# The linked primary evaluator uses inclusive (+1) pixel IoU, confidence-ordered one-to-one
# matches, six cutoffs .50–.75, and a strict confidence filter. These conventions are exposed in
# the helper. Check fixtures against any supplied official grader; never substitute COCO AP or a
# nearby metric.
#

# The task statement gives no-target images zero and uses strict IoU cutoffs. The helper
# follows those defaults; strict_iou=False reproduces the linked reference's cutoff rule.
# Compare boundary fixtures with the actual supplied grader.
#

# %% [exercise]
candidates = (0.5,)
# >>> exercise: tune confidence only on this validation image set
scores_by_threshold = {
    threshold: evaluate_boxes(validation_predictions, threshold) for threshold in candidates
}
selected_threshold = max(
    candidates, key=lambda value: (scores_by_threshold[value], -abs(value - 0.5))
)
validation_score = scores_by_threshold[selected_threshold]
# <<< exercise
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
# >>> exercise: rebuild the detector and reload before complete test inference
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
# <<< exercise


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
