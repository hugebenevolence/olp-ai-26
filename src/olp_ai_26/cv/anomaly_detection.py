"""Normal-only image anomaly detection with frozen timm features and patch memory banks.

The implementation is intentionally small enough for a six-hour contest. It learns one normal
feature memory bank and one decision threshold per category. Higher scores mean more anomalous.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

NORMAL_AUGMENTATIONS = (
    "identity",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "brightness_down",
    "brightness_up",
    "contrast_down",
    "contrast_up",
    "saturation_down",
    "saturation_up",
    "cool",
    "warm",
)

SYNTHETIC_ANOMALIES = (
    "cutpaste",
    "mixup",
    "blur",
    "dark_curve",
)


@dataclass(frozen=True)
class OfficialTask2Paths:
    """Resolved roots created from the official nested competition archive."""

    training_root: Path
    test_root: Path
    test_csv: Path


def _safe_extract_zip(archive: Path, destination: Path, *, password: str | None = None) -> None:
    """Extract a ZIP after rejecting traversal and unavailable encrypted members."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as inspection:
        uses_aes = any(member.compress_type == 99 for member in inspection.infolist())
    if uses_aes:
        if not password:
            raise RuntimeError(
                f"{archive.name} uses AES encryption. Set PRIVATE_ZIP_PASSWORD after release."
            )
        import pyzipper

        source_context = pyzipper.AESZipFile(archive)
    else:
        source_context = zipfile.ZipFile(archive)
    with source_context as source:
        members = source.infolist()
        if any(member.flag_bits & 0x1 for member in members) and not password:
            raise RuntimeError(
                f"{archive.name} is encrypted. Set PRIVATE_ZIP_PASSWORD to the organizer password."
            )
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe path inside {archive}: {member.filename}")
        source.extractall(destination, pwd=password.encode() if password else None)


def _copy_nested_archive(outer_archive: Path, member_suffix: str, destination: Path) -> Path:
    """Copy one uniquely named inner ZIP from the official outer ZIP to fast local storage."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer_archive) as source:
        matches = [
            item
            for item in source.infolist()
            if item.filename.replace("\\", "/").endswith(member_suffix)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *{member_suffix} in {outer_archive}; found {len(matches)}"
            )
        member = matches[0]
        if destination.exists() and destination.stat().st_size == member.file_size:
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with source.open(member) as input_stream, temporary.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
        temporary.replace(destination)
    return destination


def stage_official_task2_data(
    source: Path | str,
    destination: Path | str,
    *,
    phase: str = "public",
    private_password: str | None = None,
) -> OfficialTask2Paths:
    """Resolve the official nested ZIP or an already expanded directory into usable task paths.

    The released outer archive contains ``dataset_train.zip``, ``public_test.zip``, and an
    encrypted ``private_test.zip``. Only the training and selected phase archives are copied and
    expanded. Repeated calls reuse existing extracted directories.
    """
    if phase not in {"public", "private"}:
        raise ValueError("phase must be 'public' or 'private'")
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    # Accept a directory that is already expanded by the participant.
    directory_candidates = [source_path, source_path / "expanded"] if source_path.is_dir() else []
    for candidate in directory_candidates:
        training_root = candidate / "dataset_train"
        test_root = candidate / f"{phase}_test"
        test_csv = test_root / "test.csv"
        if (training_root / "train").is_dir() and test_csv.is_file():
            return OfficialTask2Paths(training_root, test_root, test_csv)

    training_root = destination_path / "dataset_train"
    test_root = destination_path / f"{phase}_test"
    test_csv = test_root / "test.csv"
    if (training_root / "train").is_dir() and test_csv.is_file():
        return OfficialTask2Paths(training_root, test_root, test_csv)

    archive_dir = destination_path / "_archives"
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        train_archive = _copy_nested_archive(
            source_path,
            "CV_Data/training_dataset/dataset_train.zip",
            archive_dir / "dataset_train.zip",
        )
        test_archive = _copy_nested_archive(
            source_path,
            f"CV_Data/{phase}_test/{phase}_test.zip",
            archive_dir / f"{phase}_test.zip",
        )
    elif source_path.is_dir():
        train_matches = list(source_path.rglob("dataset_train.zip"))
        test_matches = list(source_path.rglob(f"{phase}_test.zip"))
        if len(train_matches) != 1 or len(test_matches) != 1:
            raise FileNotFoundError(
                "Could not uniquely locate dataset_train.zip and the selected test ZIP"
            )
        train_archive, test_archive = train_matches[0], test_matches[0]
    else:
        raise FileNotFoundError(
            f"Official data source not found: {source_path}. Point OFFICIAL_DATA_SOURCE at "
            "ThiChinhThucData.zip or an expanded CV directory."
        )

    if not (training_root / "train").is_dir():
        _safe_extract_zip(train_archive, destination_path)
    if not test_csv.is_file():
        _safe_extract_zip(
            test_archive,
            destination_path,
            password=private_password if phase == "private" else None,
        )
    if not (training_root / "train").is_dir() or not test_csv.is_file():
        raise RuntimeError("Official task archives extracted without the expected roots")
    return OfficialTask2Paths(training_root, test_root, test_csv)


def load_official_training_table(training_root: Path | str) -> pd.DataFrame:
    """Load and validate the three authoritative task training CSV files."""
    root = Path(training_root)
    csv_paths = sorted(root.glob("train*.csv"))
    if [path.name for path in csv_paths] != ["train1_6.csv", "train2_5.csv", "train3_4.csv"]:
        raise FileNotFoundError(
            f"Expected train1_6.csv, train2_5.csv, and train3_4.csv under {root}"
        )
    frame = pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)
    expected_columns = ["sample_id", "category", "relative_path"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Training CSV columns must be exactly {expected_columns}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Training CSV files contain duplicate sample_id values")
    missing = [path for path in frame["relative_path"] if not (root / str(path)).is_file()]
    if missing:
        raise FileNotFoundError(f"Training CSV references {len(missing)} missing images")
    return frame.sort_values(["category", "sample_id"]).reset_index(drop=True)


def discover_normal_images(train_root: Path | str) -> pd.DataFrame:
    """Build a ``category, relative_path`` table from category subdirectories.

    Args:
        train_root: Directory containing folders such as ``category_01`` through ``category_06``.

    Returns:
        A deterministically sorted table. Paths are relative to ``train_root``.
    """
    root = Path(train_root)
    rows = [
        {
            "category": path.parent.relative_to(root).parts[0],
            "relative_path": path.relative_to(root).as_posix(),
        }
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frame = pd.DataFrame(rows, columns=["category", "relative_path"])
    if frame.empty:
        raise FileNotFoundError(f"No supported images found under {root}")
    return frame.sort_values(["category", "relative_path"]).reset_index(drop=True)


class AnomalyImageDataset(Dataset[torch.Tensor]):
    """Load unlabeled RGB images from a table while preserving values in the 0-1 range."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        root: Path | str,
        path_column: str = "relative_path",
        image_size: int = 256,
    ) -> None:
        """Store paths and create deterministic resize/float transforms."""
        if path_column not in frame:
            raise KeyError(f"Missing image path column: {path_column}")
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.path_column = path_column
        self.transform = v2.Compose(
            [
                v2.Resize((image_size, image_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        """Return the number of image paths."""
        return len(self.frame)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load one image as a ``[3,H,W]`` float tensor in the 0-1 range."""
        path = self.root / str(self.frame.iloc[index][self.path_column])
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image)


class TimmPatchFeatureExtractor(nn.Module):
    """Convert images into normalized multi-scale patch embeddings using a timm backbone."""

    def __init__(
        self,
        model_name: str = "wide_resnet50_2",
        *,
        out_indices: tuple[int, ...] = (2, 3),
        projection_dim: int = 128,
        pretrained_allowed: bool = True,
        checkpoint_path: Path | str | None = None,
        seed: int = 42,
    ) -> None:
        """Build the frozen encoder and deterministic Gaussian random projection.

        ``pretrained_allowed=True`` may download timm weights. Pass a staged local checkpoint to
        avoid network dependence and to make the exact permitted weights reproducible.
        """
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained_allowed and checkpoint_path is None,
            checkpoint_path=str(checkpoint_path or ""),
            features_only=True,
            out_indices=out_indices,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        channels = sum(self.encoder.feature_info.channels())
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(channels, projection_dim, generator=generator)
        projection /= projection_dim**0.5
        self.register_buffer("projection", projection)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized patch embeddings shaped ``[batch, patches, projection_dim]``."""
        features = self.encoder((images - self.mean) / self.std)
        target_size = features[0].shape[-2:]
        aligned = [
            F.interpolate(feature, target_size, mode="bilinear", align_corners=False)
            if feature.shape[-2:] != target_size
            else feature
            for feature in features
        ]
        combined = torch.cat(aligned, dim=1).permute(0, 2, 3, 1)
        patches = combined.flatten(1, 2) @ self.projection
        return F.normalize(patches, dim=-1)


def sample_memory_bank(
    patch_batches: Iterable[torch.Tensor],
    *,
    max_patches: int = 4096,
    seed: int = 42,
) -> torch.Tensor:
    """Uniformly sample a bounded patch memory without retaining all candidates.

    Every patch receives a deterministic random priority and only the smallest priorities are
    retained. This streaming form makes multiple normal augmentations practical on Colab RAM.
    """
    generator = torch.Generator().manual_seed(seed)
    bank: torch.Tensor | None = None
    priorities: torch.Tensor | None = None
    for batch in patch_batches:
        patches = batch.detach().float().cpu().flatten(0, 1)
        batch_priorities = torch.rand(len(patches), generator=generator)
        bank = patches if bank is None else torch.cat((bank, patches))
        priorities = (
            batch_priorities if priorities is None else torch.cat((priorities, batch_priorities))
        )
        if len(bank) > max_patches:
            keep = priorities.topk(max_patches, largest=False).indices
            bank = bank[keep]
            priorities = priorities[keep]
    if bank is None:
        raise ValueError("At least one patch-embedding batch is required")
    return bank.contiguous()


def patch_memory_scores(
    embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    *,
    top_k: int = 3,
    distance_chunk_size: int = 2048,
) -> torch.Tensor:
    """Score each image by its top-k largest nearest-normal-patch distances.

    Chunking bounds the temporary distance matrix. Both tensors must contain the same embedding
    dimension; the memory bank is moved to the embedding device for the calculation.
    """
    if embeddings.ndim != 3 or memory_bank.ndim != 2:
        raise ValueError("Expected embeddings [B,P,D] and memory_bank [M,D]")
    if embeddings.shape[-1] != memory_bank.shape[-1]:
        raise ValueError("Embedding and memory-bank dimensions differ")
    batch, patches, dimensions = embeddings.shape
    flat = embeddings.float().reshape(-1, dimensions)
    bank = memory_bank.to(flat.device, dtype=torch.float32)
    nearest = []
    for start in range(0, len(flat), distance_chunk_size):
        distances = torch.cdist(flat[start : start + distance_chunk_size], bank)
        nearest.append(distances.min(dim=1).values)
    patch_scores = torch.cat(nearest).reshape(batch, patches)
    count = min(max(1, top_k), patches)
    return patch_scores.topk(count, dim=1).values.mean(dim=1)


def cutpaste_batch(
    images: torch.Tensor,
    *,
    area_range: tuple[float, float] = (0.03, 0.15),
    seed: int | None = None,
) -> torch.Tensor:
    """Create allowed synthetic anomalies by moving a random rectangle within each image."""
    if images.ndim != 4:
        raise ValueError("Expected image batch shaped [B,C,H,W]")
    if not 0 < area_range[0] <= area_range[1] < 1:
        raise ValueError("area_range must satisfy 0 < low <= high < 1")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    output = images.clone()
    _, _, height, width = output.shape
    for index in range(len(output)):
        area = float(torch.empty(1).uniform_(*area_range, generator=generator))
        ratio = float(torch.empty(1).uniform_(0.5, 2.0, generator=generator))
        patch_height = max(2, min(height - 1, round((area * height * width / ratio) ** 0.5)))
        patch_width = max(2, min(width - 1, round(patch_height * ratio)))
        source_y = int(torch.randint(height - patch_height + 1, (1,), generator=generator))
        source_x = int(torch.randint(width - patch_width + 1, (1,), generator=generator))
        target_y = int(torch.randint(height - patch_height + 1, (1,), generator=generator))
        target_x = int(torch.randint(width - patch_width + 1, (1,), generator=generator))
        patch = output[
            index, :, source_y : source_y + patch_height, source_x : source_x + patch_width
        ].clone()
        output[index, :, target_y : target_y + patch_height, target_x : target_x + patch_width] = (
            patch
        )
    return output


def apply_synthetic_anomaly(
    images: torch.Tensor,
    name: str,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Create one deterministic synthetic anomaly from official normal images.

    ``mixup`` blends each image with another image from the same category, ``blur`` removes fine
    detail, and ``dark_curve`` overlays a thick quadratic curve. These outputs are synthetic
    positives for calibration only; they must never be inserted into the normal memory bank.

    Args:
        images: Float image batch shaped ``[B,C,H,W]`` with values in the 0-1 range.
        name: One of :data:`SYNTHETIC_ANOMALIES`.
        seed: Reproducibility seed for CutPaste, MixUp pairing, and curve geometry.

    Returns:
        A transformed batch with the same shape, device, and dtype as ``images``.
    """
    if images.ndim != 4:
        raise ValueError("Expected image batch shaped [B,C,H,W]")
    normalized = name.lower()
    if normalized == "cutpaste":
        return cutpaste_batch(images, seed=seed)
    if normalized == "mixup":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if len(images) == 1:
            partners = torch.flip(images, (-1,))
        else:
            # Assign every image a random different partner using one randomized cycle.
            order = torch.randperm(len(images), generator=generator)
            partner_indices = torch.empty_like(order)
            partner_indices[order] = torch.roll(order, shifts=1)
            partners = images[partner_indices.to(images.device)]
        alpha = torch.empty((len(images), 1, 1, 1)).uniform_(
            0.25,
            0.45,
            generator=generator,
        )
        alpha = alpha.to(device=images.device, dtype=images.dtype)
        return ((1 - alpha) * images + alpha * partners).clamp(0, 1)
    if normalized == "blur":
        radius = max(2, min(6, round(min(images.shape[-2:]) / 64)))
        kernel_size = 2 * radius + 1
        sigma = max(1.0, kernel_size / 4)
        return v2.functional.gaussian_blur(
            images,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma],
        )
    if normalized == "dark_curve":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        batch, _, height, width = images.shape
        masks = torch.zeros((batch, 1, height, width), dtype=torch.float32)
        steps = max(height, width) * 2
        time = torch.linspace(0, 1, steps)
        for index in range(batch):
            horizontal = bool(torch.randint(2, (1,), generator=generator).item())
            if horizontal:
                x_points = torch.tensor(
                    [0.0, float(width // 2), float(width - 1)], dtype=torch.float32
                )
                y_points = torch.randint(height, (3,), generator=generator).float()
            else:
                x_points = torch.randint(width, (3,), generator=generator).float()
                y_points = torch.tensor(
                    [0.0, float(height // 2), float(height - 1)], dtype=torch.float32
                )
            one_minus = 1 - time
            curve_x = (
                one_minus.square() * x_points[0]
                + 2 * one_minus * time * x_points[1]
                + time.square() * x_points[2]
            )
            curve_y = (
                one_minus.square() * y_points[0]
                + 2 * one_minus * time * y_points[1]
                + time.square() * y_points[2]
            )
            masks[index, 0, curve_y.round().long(), curve_x.round().long()] = 1
        radius = max(2, round(min(height, width) * 0.0125))
        masks = F.max_pool2d(masks, kernel_size=2 * radius + 1, stride=1, padding=radius)
        masks = masks.to(device=images.device, dtype=images.dtype)
        return images * (1 - 0.85 * masks)
    raise ValueError(f"Unknown synthetic anomaly {name!r}; choose {SYNTHETIC_ANOMALIES}")


def apply_normal_augmentation(images: torch.Tensor, name: str) -> torch.Tensor:
    """Apply one deterministic, mild normal-data augmentation to a tensor batch.

    These transforms expand the normal memory bank; they do not create positive anomaly labels.
    Geometric transforms should only be enabled when the category naturally contains that
    orientation. Photometric factors stay within roughly eight percent of the original.
    """
    if images.ndim != 4:
        raise ValueError("Expected image batch shaped [B,C,H,W]")
    normalized = name.lower()
    if normalized == "identity":
        return images
    if normalized == "hflip":
        return torch.flip(images, (-1,))
    if normalized == "vflip":
        return torch.flip(images, (-2,))
    if normalized == "rot90":
        return torch.rot90(images, 1, (-2, -1))
    if normalized == "rot180":
        return torch.rot90(images, 2, (-2, -1))
    if normalized == "rot270":
        return torch.rot90(images, 3, (-2, -1))
    if normalized in {"brightness_down", "brightness_up"}:
        factor = 0.95 if normalized.endswith("down") else 1.05
        return (images * factor).clamp(0, 1)
    if normalized in {"contrast_down", "contrast_up"}:
        factor = 0.92 if normalized.endswith("down") else 1.08
        mean = images.mean(dim=(-2, -1), keepdim=True)
        return ((images - mean) * factor + mean).clamp(0, 1)
    if normalized in {"saturation_down", "saturation_up"}:
        factor = 0.95 if normalized.endswith("down") else 1.05
        gray = images.mean(dim=1, keepdim=True)
        return ((images - gray) * factor + gray).clamp(0, 1)
    if normalized in {"cool", "warm"}:
        factors = (
            torch.tensor((0.98, 1.00, 1.02), device=images.device, dtype=images.dtype)
            if normalized == "cool"
            else torch.tensor((1.02, 1.00, 0.98), device=images.device, dtype=images.dtype)
        )
        return (images * factors.view(1, 3, 1, 1)).clamp(0, 1)
    raise ValueError(f"Unknown normal augmentation {name!r}; choose {NORMAL_AUGMENTATIONS}")


def calibrate_anomaly_threshold(
    normal_scores: Iterable[float],
    synthetic_scores: Iterable[float] | None = None,
    *,
    normal_quantile: float = 0.99,
) -> dict[str, float]:
    """Calibrate a threshold using synthetic balanced accuracy or a normal-only quantile.

    When synthetic scores are provided, candidate thresholds maximize balanced accuracy between
    held-out normal images and generated anomalies. The returned normal quantile remains useful as
    a conservative alternative and for public-score threshold sweeps.
    """
    normal = np.asarray(list(normal_scores), dtype=np.float64)
    if not len(normal) or not np.isfinite(normal).all():
        raise ValueError("normal_scores must contain finite values")
    if not 0 < normal_quantile < 1:
        raise ValueError("normal_quantile must be between zero and one")
    quantile_threshold = float(np.quantile(normal, normal_quantile))
    result = {
        "threshold": quantile_threshold,
        "normal_quantile_threshold": quantile_threshold,
        "proxy_balanced_accuracy": float("nan"),
    }
    if synthetic_scores is None:
        return result
    synthetic = np.asarray(list(synthetic_scores), dtype=np.float64)
    if not len(synthetic) or not np.isfinite(synthetic).all():
        raise ValueError("synthetic_scores must contain finite values")
    values = np.unique(np.concatenate([normal, synthetic]))
    candidates = np.concatenate(
        ([np.nextafter(values[0], -np.inf)], (values[:-1] + values[1:]) / 2, [values[-1]])
    )
    truth = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(synthetic), dtype=int)])
    scores = np.concatenate([normal, synthetic])
    balanced = np.asarray(
        [balanced_accuracy_score(truth, scores >= threshold) for threshold in candidates]
    )
    best = np.flatnonzero(balanced == balanced.max())
    # Prefer the largest tied threshold to reduce false positives on true normal images.
    index = int(best[-1])
    result["threshold"] = float(candidates[index])
    result["proxy_balanced_accuracy"] = float(balanced[index])
    return result


def select_anomaly_threshold(calibration: dict[str, float], mode: str = "synthetic") -> float:
    """Select a stored synthetic, normal-quantile, minimum, or maximum threshold.

    ``min_synthetic_quantile`` is more sensitive than either constituent threshold and is useful
    when a CutPaste-calibrated baseline under-predicts anomalies. The selected mode must still be
    evaluated through permitted public submissions and frozen before private inference.
    """
    synthetic = float(calibration["threshold"])
    quantile = float(calibration["normal_quantile_threshold"])
    choices = {
        "synthetic": synthetic,
        "normal_quantile": quantile,
        "min_synthetic_quantile": min(synthetic, quantile),
        "max_synthetic_quantile": max(synthetic, quantile),
    }
    try:
        return choices[mode]
    except KeyError as error:
        raise ValueError(f"Unknown threshold mode {mode!r}; choose {sorted(choices)}") from error


def validate_anomaly_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    """Enforce the task's exact sample_id/category/label contract against ``test.csv``."""
    expected = ["sample_id", "category", "label"]
    if list(submission.columns) != expected:
        raise ValueError(f"Submission columns must be exactly {expected}")
    if len(submission) != len(test) or submission["sample_id"].duplicated().any():
        raise ValueError("Submission must contain every test sample exactly once")
    merged = submission.merge(
        test[["sample_id", "category"]],
        on="sample_id",
        how="outer",
        suffixes=("_submission", "_test"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Submission sample IDs do not match test.csv")
    if not (merged["category_submission"] == merged["category_test"]).all():
        raise ValueError("Submission categories do not match test.csv")
    if submission["label"].isna().any() or not set(submission["label"]).issubset({0, 1}):
        raise ValueError("Every label must be integer 0 or 1")
