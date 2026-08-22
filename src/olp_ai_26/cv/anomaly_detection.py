"""Normal-only image anomaly detection with frozen timm features and patch memory banks.

The implementation is intentionally small enough for a six-hour contest. It learns one normal
feature memory bank and one decision threshold per category. Higher scores mean more anomalous.
"""

from __future__ import annotations

import math
import shutil
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
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
    "blur_mild",
)

SYNTHETIC_ANOMALIES = (
    "cutpaste",
    "mixup",
    "cutmix",
    "dark_curve",
    "white_line",
    "gray_curve",
    "white_curve",
)

SYNTHETIC_ANOMALY_DEFAULTS: dict[str, dict[str, object]] = {
    "cutpaste": {"cutpaste_area_range": (0.03, 0.15)},
    "mixup": {"mixup_alpha_range": (0.25, 0.45)},
    "cutmix": {
        "cutmix_area_range": (0.025, 0.10),
        "cutmix_opacity_range": (0.06, 0.14),
    },
    "dark_curve": {
        "curve_length_fraction_range": (0.08, 0.16),
        "curve_width_fraction": 0.012,
        "curve_darkness": 0.75,
    },
    "white_line": {
        "line_length_fraction_range": (0.05, 0.10),
        "line_width_fraction": 0.002,
        "line_opacity": 1.0,
    },
    "gray_curve": {
        "colored_curve_length_fraction_range": (0.06, 0.14),
        "colored_curve_width_fraction": 0.004,
        "curve_color_range": (0.50, 0.75),
        "curve_opacity_range": (0.10, 0.25),
    },
    "white_curve": {
        "colored_curve_length_fraction_range": (0.06, 0.14),
        "colored_curve_width_fraction": 0.0035,
        "curve_color_range": (0.85, 1.00),
        "curve_opacity_range": (0.08, 0.22),
    },
}


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


class DinoV2PatchFeatureExtractor(nn.Module):
    """Extract normalized final-layer DINOv2 patch tokens for AnomalyDINO-style scoring.

    Unlike the CNN PatchCore extractor, this wrapper does not concatenate multiple layers or apply
    a random projection. DINOv2's native 14-by-14 patch tokens are used directly, matching the
    representation choice in AnomalyDINO. The configured square input size must be divisible by 14.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        *,
        image_size: int = 448,
        pretrained_allowed: bool = True,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        """Build a frozen timm DINOv2 encoder with reproducible ImageNet normalization.

        Args:
            model_name: A timm DINOv2 model identifier.
            image_size: Square input resolution. It must be a multiple of DINOv2's patch size 14.
            pretrained_allowed: Whether timm may download the permitted pretrained weights.
            checkpoint_path: Optional staged local checkpoint used instead of a network download.
        """
        super().__init__()
        if image_size < 14 or image_size % 14:
            raise ValueError("DINOv2 image_size must be a positive multiple of 14")
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained_allowed and checkpoint_path is None,
            checkpoint_path=str(checkpoint_path or ""),
            num_classes=0,
            img_size=image_size,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.image_size = image_size
        self.patch_size = 14
        self.num_prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 1))
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized patch tokens shaped ``[batch, patches, embedding_dim]``."""
        if images.ndim != 4 or images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected DINOv2 images shaped [B,3,{self.image_size},{self.image_size}]"
            )
        tokens = self.encoder.forward_features((images - self.mean) / self.std)
        if isinstance(tokens, dict):
            try:
                patches = tokens["x_norm_patchtokens"]
            except KeyError as error:
                raise RuntimeError(
                    "DINOv2 output does not contain normalized patch tokens"
                ) from error
        else:
            patches = tokens[:, self.num_prefix_tokens :]
        expected_patches = (self.image_size // self.patch_size) ** 2
        if patches.shape[1] != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} DINOv2 patches, received {patches.shape[1]}"
            )
        return F.normalize(patches.float(), dim=-1)


def aggregate_patch_neighborhoods(
    embeddings: torch.Tensor,
    *,
    kernel_size: int,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Average ViT patch tokens over a centered local neighborhood.

    This implements the representation step used by multi-degree local-neighborhood anomaly
    detectors. A degree of one leaves the token grid unchanged; degrees three and five add local
    context while preserving one output token per original patch position.

    Args:
        embeddings: Patch tokens shaped ``[batch, patches, dimensions]``.
        kernel_size: Positive odd neighborhood width, normally 1, 3, or 5.
        grid_size: Optional ``(height, width)`` patch grid. When omitted, a square grid is inferred.

    Returns:
        L2-normalized tokens with the same shape as ``embeddings``.
    """
    if embeddings.ndim != 3:
        raise ValueError("Expected patch embeddings shaped [B,P,D]")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    batch, patches, dimensions = embeddings.shape
    if grid_size is None:
        side = math.isqrt(patches)
        grid_size = (side, side)
    height, width = grid_size
    if height < 1 or width < 1 or height * width != patches:
        raise ValueError("grid_size must contain exactly one position per patch token")
    feature_map = embeddings.float().transpose(1, 2).reshape(batch, dimensions, height, width)
    pooled = F.avg_pool2d(
        feature_map,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return F.normalize(pooled.flatten(2).transpose(1, 2), dim=-1)


def changed_patch_mask(
    original: torch.Tensor,
    transformed: torch.Tensor,
    *,
    grid_size: tuple[int, int],
    difference_threshold: float = 0.015,
    dilation: int = 1,
) -> torch.Tensor:
    """Locate patch tokens touched by a synthetic image transformation.

    The pixel-space maximum absolute channel difference is max-pooled into the feature extractor's
    patch grid. This deliberately marks a patch when *any* pixel changed enough, which is suitable
    for short curves and thin lines that average pooling could erase. Optional patch-grid dilation
    includes immediate context around the edited region.

    Args:
        original: Clean images shaped ``[batch, channels, height, width]`` in the ``[0, 1]`` range.
        transformed: Synthetic images with exactly the same shape as ``original``.
        grid_size: Feature-token grid as ``(height, width)``.
        difference_threshold: Minimum absolute pixel change required to mark a patch.
        dilation: Number of patch-grid neighbors to include around each changed patch.

    Returns:
        Boolean mask shaped ``[batch, grid_height * grid_width]`` in token order.
    """
    if original.ndim != 4 or transformed.ndim != 4:
        raise ValueError("Expected original and transformed images shaped [B,C,H,W]")
    if original.shape != transformed.shape:
        raise ValueError("Original and transformed image shapes must match")
    if len(grid_size) != 2 or grid_size[0] < 1 or grid_size[1] < 1:
        raise ValueError("grid_size must contain two positive integers")
    if difference_threshold < 0:
        raise ValueError("difference_threshold cannot be negative")
    if dilation < 0:
        raise ValueError("dilation cannot be negative")
    pixel_difference = (transformed - original).abs().amax(dim=1, keepdim=True)
    pooled_difference = F.adaptive_max_pool2d(pixel_difference, grid_size)
    mask = pooled_difference >= difference_threshold
    if dilation:
        kernel_size = 2 * dilation + 1
        mask = F.max_pool2d(
            mask.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=dilation,
        ).bool()
    return mask.flatten(1)


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
    top_fraction: float | None = None,
    distance_metric: str = "euclidean",
    distance_chunk_size: int = 2048,
) -> torch.Tensor:
    """Score each image by its top-k largest nearest-normal-patch distances.

    Chunking bounds the temporary distance matrix. Both tensors must contain the same embedding
    dimension; the memory bank is moved to the embedding device for the calculation.
    """
    return patch_memory_features(
        embeddings,
        memory_bank,
        top_k=top_k,
        top_fraction=top_fraction,
        distance_metric=distance_metric,
        distance_chunk_size=distance_chunk_size,
    )[:, -1]


def patch_memory_features(
    embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    *,
    top_k: int = 3,
    top_fraction: float | None = None,
    distance_metric: str = "euclidean",
    distance_chunk_size: int = 2048,
) -> torch.Tensor:
    """Summarize each image's nearest-memory patch-distance distribution.

    The six returned columns are mean, standard deviation, maximum, 90th percentile, 99th
    percentile, and upper-tail mean. With ``top_fraction=0.01``, the last value is the mean of the
    most anomalous 1% of patches used by AnomalyDINO. Otherwise, ``top_k`` determines the tail size.
    The final column exactly matches :func:`patch_memory_scores`.
    """
    patch_scores = patch_memory_distances(
        embeddings,
        memory_bank,
        distance_metric=distance_metric,
        distance_chunk_size=distance_chunk_size,
    )
    return summarize_patch_distances(
        patch_scores,
        top_k=top_k,
        top_fraction=top_fraction,
    )


def patch_memory_distances(
    embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    *,
    distance_metric: str = "euclidean",
    distance_chunk_size: int = 2048,
) -> torch.Tensor:
    """Return each query patch's distance to its nearest normal-memory patch.

    The output retains the image and patch axes, enabling multiple neighborhood degrees to be
    fused at patch level before image-level tail aggregation.
    """
    if embeddings.ndim != 3 or memory_bank.ndim != 2:
        raise ValueError("Expected embeddings [B,P,D] and memory_bank [M,D]")
    if embeddings.shape[-1] != memory_bank.shape[-1]:
        raise ValueError("Embedding and memory-bank dimensions differ")
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive")
    if distance_metric not in {"euclidean", "cosine"}:
        raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
    batch, patches, dimensions = embeddings.shape
    flat = embeddings.float().reshape(-1, dimensions)
    bank = memory_bank.to(flat.device, dtype=torch.float32)
    normalized_bank = F.normalize(bank, dim=-1) if distance_metric == "cosine" else None
    nearest = []
    for start in range(0, len(flat), distance_chunk_size):
        queries = flat[start : start + distance_chunk_size]
        if distance_metric == "cosine":
            queries = F.normalize(queries, dim=-1)
            assert normalized_bank is not None
            values = 1 - queries @ normalized_bank.T
            nearest.append(values.min(dim=1).values.clamp(0, 2))
        else:
            distances = torch.cdist(queries, bank)
            nearest.append(distances.min(dim=1).values)
    return torch.cat(nearest).reshape(batch, patches)


def summarize_patch_distances(
    patch_scores: torch.Tensor,
    *,
    top_k: int = 3,
    top_fraction: float | None = None,
) -> torch.Tensor:
    """Summarize per-patch anomaly distances into six image-level statistics."""
    if patch_scores.ndim != 2:
        raise ValueError("Expected patch_scores shaped [B,P]")
    if top_fraction is not None and not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    _, patches = patch_scores.shape
    count = (
        max(1, math.ceil(patches * top_fraction))
        if top_fraction is not None
        else min(max(1, top_k), patches)
    )
    quantiles = torch.quantile(
        patch_scores,
        torch.tensor((0.90, 0.99), device=patch_scores.device),
        dim=1,
    )
    return torch.stack(
        (
            patch_scores.mean(dim=1),
            patch_scores.std(dim=1, unbiased=False),
            patch_scores.max(dim=1).values,
            quantiles[0],
            quantiles[1],
            patch_scores.topk(count, dim=1).values.mean(dim=1),
        ),
        dim=1,
    )


def fit_positive_evidence_head(
    normal_features: torch.Tensor,
    synthetic_features: torch.Tensor,
    *,
    seed: int = 42,
    normal_margin_quantile: float = 0.95,
    synthetic_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    """Fit a standardized logistic head and store a normal-evidence activation margin.

    The head learns positive signals from synthetic defects. At inference,
    :func:`positive_evidence_scores` subtracts a training-normal logit margin and clamps at zero,
    ensuring a low synthetic-defect logit never becomes negative evidence for anomaly detection.
    When ``synthetic_weights`` is supplied, its values define the relative contribution of the
    synthetic rows while the normal and synthetic classes each retain half of the total fit weight.
    Pass ``None`` to preserve the original balanced, unweighted fit exactly.
    """
    normal = normal_features.detach().float().cpu().numpy()
    synthetic = synthetic_features.detach().float().cpu().numpy()
    if normal.ndim != 2 or synthetic.ndim != 2 or normal.shape[1] != synthetic.shape[1]:
        raise ValueError("normal_features and synthetic_features must be 2D with equal width")
    if not len(normal) or not len(synthetic):
        raise ValueError("Both normal and synthetic feature sets must be non-empty")
    if not 0 < normal_margin_quantile < 1:
        raise ValueError("normal_margin_quantile must be between zero and one")
    features = np.concatenate((normal, synthetic), axis=0)
    labels = np.concatenate((np.zeros(len(normal), dtype=int), np.ones(len(synthetic), dtype=int)))
    fit_weights = None
    classifier_class_weight: str | None = "balanced"
    if synthetic_weights is None:
        center = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0), 1e-6)
    else:
        weights = synthetic_weights.detach().float().cpu().numpy()
        if weights.ndim != 1 or len(weights) != len(synthetic):
            raise ValueError("synthetic_weights must contain one value per synthetic feature")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("synthetic_weights must be finite, non-negative, and sum above zero")
        synthetic_fit_weights = 0.5 * weights / weights.sum()
        normal_fit_weights = np.full(len(normal), 0.5 / len(normal), dtype=np.float64)
        fit_weights = np.concatenate((normal_fit_weights, synthetic_fit_weights))
        center = np.average(features, axis=0, weights=fit_weights)
        variance = np.average((features - center) ** 2, axis=0, weights=fit_weights)
        scale = np.maximum(np.sqrt(variance), 1e-6)
        # Preserve the usual effective regularization scale used by scikit-learn.
        fit_weights = fit_weights * len(fit_weights)
        classifier_class_weight = None
    standardized = (features - center) / scale
    classifier = LogisticRegression(
        class_weight=classifier_class_weight,
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    ).fit(standardized, labels, sample_weight=fit_weights)
    normal_logits = ((normal - center) / scale) @ classifier.coef_[0] + classifier.intercept_[0]
    return {
        "center": torch.tensor(center, dtype=torch.float32),
        "scale": torch.tensor(scale, dtype=torch.float32),
        "coefficient": torch.tensor(classifier.coef_[0], dtype=torch.float32),
        "intercept": float(classifier.intercept_[0]),
        "normal_margin": float(np.quantile(normal_logits, normal_margin_quantile)),
    }


def positive_evidence_scores(
    features: torch.Tensor,
    head: dict[str, torch.Tensor | float],
) -> torch.Tensor:
    """Return non-negative learned evidence for known synthetic anomaly signals."""
    values = features.detach().float().cpu()
    standardized = (values - head["center"]) / head["scale"]
    logits = standardized @ head["coefficient"] + float(head["intercept"])
    return torch.relu(logits - float(head["normal_margin"]))


def fixed_count_rank_fusion(
    base_scores: Sequence[float],
    patch_scores: Sequence[float],
    *,
    anomaly_count: int,
    patch_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse two score rankings and label an exact number of highest-ranked samples.

    Rank fusion removes incompatible numeric scales between an open-set distance score and a
    discriminative patch head. The returned continuous score is an ordinal percentile mixture;
    only its ordering is meaningful. Stable sorting makes tied-score behavior reproducible.

    Args:
        base_scores: Primary anomaly scores, normally the calibrated DINO ranking signal.
        patch_scores: Scores from the synthetic local-defect patch adapter.
        anomaly_count: Exact number of samples to label anomalous.
        patch_weight: Patch-rank contribution in ``[0, 1]``; zero is base only.

    Returns:
        ``(labels, fused_ranks)`` arrays in original row order.
    """
    base = np.asarray(base_scores, dtype=np.float64)
    patch = np.asarray(patch_scores, dtype=np.float64)
    if base.ndim != 1 or patch.ndim != 1 or base.shape != patch.shape:
        raise ValueError("base_scores and patch_scores must be equal-length vectors")
    if not len(base):
        raise ValueError("At least one score is required")
    if not np.isfinite(base).all() or not np.isfinite(patch).all():
        raise ValueError("Scores must be finite")
    if not 0 <= anomaly_count <= len(base):
        raise ValueError("anomaly_count must be between zero and the number of scores")
    if not 0 <= patch_weight <= 1:
        raise ValueError("patch_weight must be between zero and one")

    def percentile_ranks(values: np.ndarray) -> np.ndarray:
        # Average ranks prevent a large zero-evidence tie from acquiring a false ordering based on
        # CSV row position. This matters because the margin-clamped adapter legitimately emits zero
        # for most normal-looking images.
        return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)

    fused = (1 - patch_weight) * percentile_ranks(base) + patch_weight * percentile_ranks(patch)
    labels = np.zeros(len(fused), dtype=np.int64)
    if anomaly_count:
        selected = np.argsort(fused, kind="mergesort")[-anomaly_count:]
        labels[selected] = 1
    return labels, fused


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


def synthetic_anomaly_defaults() -> dict[str, dict[str, object]]:
    """Return a mutable copy of the recommended synthetic-anomaly parameters."""
    return {method: dict(parameters) for method, parameters in SYNTHETIC_ANOMALY_DEFAULTS.items()}


def _random_partner_images(
    images: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Pair each image with a different randomly ordered image from the same batch."""
    if len(images) == 1:
        return torch.flip(images, (-1,))
    order = torch.randperm(len(images), generator=generator)
    partner_indices = torch.empty_like(order)
    partner_indices[order] = torch.roll(order, shifts=1)
    return images[partner_indices.to(images.device)]


def _sample_foreground_center(
    image: torch.Tensor,
    *,
    margin_x: int,
    margin_y: int,
    generator: torch.Generator,
) -> tuple[int, int]:
    """Sample an internal point that differs from the estimated border background."""
    _, height, width = image.shape
    cpu_image = image.detach().float().cpu()
    border = torch.cat(
        (
            cpu_image[:, 0, :],
            cpu_image[:, -1, :],
            cpu_image[:, :, 0],
            cpu_image[:, :, -1],
        ),
        dim=1,
    )
    background = border.median(dim=1).values.view(3, 1, 1)
    distance = (cpu_image - background).abs().mean(dim=0)
    valid = torch.zeros((height, width), dtype=torch.bool)
    valid[margin_y : height - margin_y, margin_x : width - margin_x] = True
    valid_distances = distance[valid]
    adaptive_threshold = max(0.06, float(torch.quantile(valid_distances, 0.80)))
    candidates = torch.nonzero(valid & (distance >= adaptive_threshold), as_tuple=False)
    if len(candidates):
        selected = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
        return int(selected[1]), int(selected[0])
    center_x = int(torch.randint(margin_x, width - margin_x, (1,), generator=generator))
    center_y = int(torch.randint(margin_y, height - margin_y, (1,), generator=generator))
    return center_x, center_y


def _validate_fraction_range(values: tuple[float, float], name: str) -> None:
    """Validate an increasing fractional interval inside zero and one."""
    if not 0 < values[0] <= values[1] < 1:
        raise ValueError(f"{name} must satisfy 0 < low <= high < 1")


def _rasterize_path_mask(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
    *,
    batch_index: int,
    masks: torch.Tensor,
    width_fraction: float,
) -> None:
    """Rasterize and thicken one internal path into a batch mask in place."""
    height, width = masks.shape[-2:]
    x = points_x.round().long().clamp(0, width - 1)
    y = points_y.round().long().clamp(0, height - 1)
    masks[batch_index, 0, y, x] = 1
    radius = max(0, round(min(height, width) * width_fraction / 2))
    if radius:
        masks[batch_index : batch_index + 1] = F.max_pool2d(
            masks[batch_index : batch_index + 1],
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )


def _colored_curve_batch(
    images: torch.Tensor,
    *,
    seed: int,
    length_fraction_range: tuple[float, float],
    width_fraction: float,
    color_range: tuple[float, float],
    opacity_range: tuple[float, float],
) -> torch.Tensor:
    """Blend thin gray-scale quadratic curves into foreground regions."""
    _validate_fraction_range(length_fraction_range, "colored_curve_length_fraction_range")
    if not 0 < width_fraction < 1:
        raise ValueError("colored_curve_width_fraction must be between zero and one")
    if not 0 <= color_range[0] <= color_range[1] <= 1:
        raise ValueError("curve_color_range must satisfy 0 <= low <= high <= 1")
    if not 0 < opacity_range[0] <= opacity_range[1] <= 1:
        raise ValueError("curve_opacity_range must satisfy 0 < low <= high <= 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, _, height, width = images.shape
    masks = torch.zeros((batch, 1, height, width), dtype=torch.float32)
    colors = torch.empty((batch, 1, 1, 1), dtype=torch.float32)
    opacities = torch.empty((batch, 1, 1, 1), dtype=torch.float32)
    for index in range(batch):
        length_fraction = float(
            torch.empty(1).uniform_(*length_fraction_range, generator=generator)
        )
        length = length_fraction * min(height, width)
        safe_margin = max(3, math.ceil(length * 0.65))
        center_x, center_y = _sample_foreground_center(
            images[index],
            margin_x=safe_margin,
            margin_y=safe_margin,
            generator=generator,
        )
        angle = float(torch.empty(1).uniform_(0, 2 * math.pi, generator=generator))
        direction_x, direction_y = math.cos(angle), math.sin(angle)
        perpendicular_x, perpendicular_y = -direction_y, direction_x
        half = length / 2
        bend = float(torch.empty(1).uniform_(-0.30, 0.30, generator=generator)) * length
        x_points = torch.tensor(
            [
                center_x - direction_x * half,
                center_x + perpendicular_x * bend,
                center_x + direction_x * half,
            ]
        )
        y_points = torch.tensor(
            [
                center_y - direction_y * half,
                center_y + perpendicular_y * bend,
                center_y + direction_y * half,
            ]
        )
        time = torch.linspace(0, 1, max(32, round(length * 2)))
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
        _rasterize_path_mask(
            curve_x,
            curve_y,
            batch_index=index,
            masks=masks,
            width_fraction=width_fraction,
        )
        colors[index] = torch.empty(1).uniform_(*color_range, generator=generator)
        opacities[index] = torch.empty(1).uniform_(*opacity_range, generator=generator)
    masks = masks.to(device=images.device, dtype=images.dtype)
    colors = colors.to(device=images.device, dtype=images.dtype)
    opacities = opacities.to(device=images.device, dtype=images.dtype)
    return (images * (1 - opacities * masks) + colors * opacities * masks).clamp(0, 1)


def apply_synthetic_anomaly(
    images: torch.Tensor,
    name: str,
    *,
    seed: int = 42,
    cutpaste_area_range: tuple[float, float] = (0.03, 0.15),
    mixup_alpha_range: tuple[float, float] = (0.25, 0.45),
    cutmix_area_range: tuple[float, float] = (0.025, 0.10),
    cutmix_opacity_range: tuple[float, float] = (0.06, 0.14),
    curve_length_fraction_range: tuple[float, float] = (0.08, 0.16),
    curve_width_fraction: float = 0.012,
    curve_darkness: float = 0.75,
    line_length_fraction_range: tuple[float, float] = (0.05, 0.10),
    line_width_fraction: float = 0.002,
    line_opacity: float = 1.0,
    colored_curve_length_fraction_range: tuple[float, float] = (0.06, 0.14),
    colored_curve_width_fraction: float = 0.004,
    curve_color_range: tuple[float, float] | None = None,
    curve_opacity_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Create one deterministic synthetic anomaly from official normal images.

    ``mixup`` blends whole same-category images, while ``cutmix`` blends a small partner patch at
    low opacity. Curve and line methods place short marks inside the estimated foreground rather
    than drawing from an image edge. ``gray_curve`` and ``white_curve`` use subtle target colors
    and opacity ranges observed during private-set inspection. These outputs are synthetic
    positives; never insert them into the normal memory bank.

    Args:
        images: Float image batch shaped ``[B,C,H,W]`` with values in the 0-1 range.
        name: One of :data:`SYNTHETIC_ANOMALIES`.
        seed: Reproducibility seed for CutPaste, MixUp pairing, and curve geometry.
        cutpaste_area_range: Minimum and maximum pasted fraction of image area.
        mixup_alpha_range: Minimum and maximum contribution from the random partner image.
        cutmix_area_range: Minimum and maximum area of the blended partner patch.
        cutmix_opacity_range: Minimum and maximum opacity of the blended partner patch.
        curve_length_fraction_range: Curve length relative to the shorter image side.
        curve_width_fraction: Approximate full curve width divided by the shorter image side.
        curve_darkness: Curve opacity, where 0 changes nothing and 1 produces black pixels.
        line_length_fraction_range: White-line length relative to the shorter image side.
        line_width_fraction: Approximate full white-line width relative to the shorter image side.
        line_opacity: White-line opacity, where 1 produces pure white pixels.
        colored_curve_length_fraction_range: Colored-curve length relative to the shorter side.
        colored_curve_width_fraction: Approximate full colored-curve width relative to that side.
        curve_color_range: Optional gray-scale target intensity range for colored curves.
        curve_opacity_range: Optional colored-curve blending-opacity range.

    Returns:
        A transformed batch with the same shape, device, and dtype as ``images``.
    """
    if images.ndim != 4:
        raise ValueError("Expected image batch shaped [B,C,H,W]")
    normalized = name.lower()
    if normalized == "cutpaste":
        return cutpaste_batch(images, area_range=cutpaste_area_range, seed=seed)
    if normalized == "mixup":
        _validate_fraction_range(mixup_alpha_range, "mixup_alpha_range")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        partners = _random_partner_images(images, generator)
        alpha = torch.empty((len(images), 1, 1, 1)).uniform_(
            *mixup_alpha_range,
            generator=generator,
        )
        alpha = alpha.to(device=images.device, dtype=images.dtype)
        return ((1 - alpha) * images + alpha * partners).clamp(0, 1)
    if normalized == "cutmix":
        _validate_fraction_range(cutmix_area_range, "cutmix_area_range")
        _validate_fraction_range(cutmix_opacity_range, "cutmix_opacity_range")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        partners = _random_partner_images(images, generator)
        output = images.clone()
        _, _, height, width = images.shape
        for index in range(len(images)):
            area = float(torch.empty(1).uniform_(*cutmix_area_range, generator=generator))
            ratio = float(torch.empty(1).uniform_(0.65, 1.55, generator=generator))
            patch_height = max(2, round((area * height * width / ratio) ** 0.5))
            patch_width = max(2, round(patch_height * ratio))
            patch_height = min(patch_height, height // 2)
            patch_width = min(patch_width, width // 2)
            half_x = patch_width // 2 + 1
            half_y = patch_height // 2 + 1
            source_x, source_y = _sample_foreground_center(
                partners[index],
                margin_x=half_x,
                margin_y=half_y,
                generator=generator,
            )
            target_x, target_y = _sample_foreground_center(
                images[index],
                margin_x=half_x,
                margin_y=half_y,
                generator=generator,
            )
            source = partners[
                index,
                :,
                source_y - patch_height // 2 : source_y - patch_height // 2 + patch_height,
                source_x - patch_width // 2 : source_x - patch_width // 2 + patch_width,
            ]
            y0 = target_y - patch_height // 2
            x0 = target_x - patch_width // 2
            opacity = float(torch.empty(1).uniform_(*cutmix_opacity_range, generator=generator))
            destination = output[index, :, y0 : y0 + patch_height, x0 : x0 + patch_width]
            output[index, :, y0 : y0 + patch_height, x0 : x0 + patch_width] = (
                1 - opacity
            ) * destination + opacity * source
        return output.clamp(0, 1)
    if normalized == "dark_curve":
        _validate_fraction_range(curve_length_fraction_range, "curve_length_fraction_range")
        if not 0 < curve_width_fraction < 1:
            raise ValueError("curve_width_fraction must be between zero and one")
        if not 0 < curve_darkness <= 1:
            raise ValueError("curve_darkness must be in (0, 1]")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        batch, _, height, width = images.shape
        masks = torch.zeros((batch, 1, height, width), dtype=torch.float32)
        for index in range(batch):
            length_fraction = float(
                torch.empty(1).uniform_(*curve_length_fraction_range, generator=generator)
            )
            length = length_fraction * min(height, width)
            safe_margin = max(3, math.ceil(length * 0.65))
            center_x, center_y = _sample_foreground_center(
                images[index],
                margin_x=safe_margin,
                margin_y=safe_margin,
                generator=generator,
            )
            angle = float(torch.empty(1).uniform_(0, 2 * math.pi, generator=generator))
            direction_x, direction_y = math.cos(angle), math.sin(angle)
            perpendicular_x, perpendicular_y = -direction_y, direction_x
            half = length / 2
            bend = float(torch.empty(1).uniform_(-0.22, 0.22, generator=generator)) * length
            x_points = torch.tensor(
                [
                    center_x - direction_x * half,
                    center_x + perpendicular_x * bend,
                    center_x + direction_x * half,
                ]
            )
            y_points = torch.tensor(
                [
                    center_y - direction_y * half,
                    center_y + perpendicular_y * bend,
                    center_y + direction_y * half,
                ]
            )
            steps = max(32, round(length * 2))
            time = torch.linspace(0, 1, steps)
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
            _rasterize_path_mask(
                curve_x,
                curve_y,
                batch_index=index,
                masks=masks,
                width_fraction=curve_width_fraction,
            )
        masks = masks.to(device=images.device, dtype=images.dtype)
        return images * (1 - curve_darkness * masks)
    if normalized in {"gray_curve", "white_curve"}:
        default_color = (0.50, 0.75) if normalized == "gray_curve" else (0.85, 1.00)
        default_opacity = (0.10, 0.25) if normalized == "gray_curve" else (0.08, 0.22)
        return _colored_curve_batch(
            images,
            seed=seed,
            length_fraction_range=colored_curve_length_fraction_range,
            width_fraction=colored_curve_width_fraction,
            color_range=curve_color_range or default_color,
            opacity_range=curve_opacity_range or default_opacity,
        )
    if normalized == "white_line":
        _validate_fraction_range(line_length_fraction_range, "line_length_fraction_range")
        if not 0 < line_width_fraction < 1:
            raise ValueError("line_width_fraction must be between zero and one")
        if not 0 < line_opacity <= 1:
            raise ValueError("line_opacity must be in (0, 1]")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        batch, _, height, width = images.shape
        masks = torch.zeros((batch, 1, height, width), dtype=torch.float32)
        for index in range(batch):
            length_fraction = float(
                torch.empty(1).uniform_(*line_length_fraction_range, generator=generator)
            )
            length = length_fraction * min(height, width)
            safe_margin = max(3, math.ceil(length * 0.55))
            center_x, center_y = _sample_foreground_center(
                images[index],
                margin_x=safe_margin,
                margin_y=safe_margin,
                generator=generator,
            )
            angle = float(torch.empty(1).uniform_(0, 2 * math.pi, generator=generator))
            half = length / 2
            points_x = torch.linspace(
                center_x - math.cos(angle) * half,
                center_x + math.cos(angle) * half,
                max(16, round(length * 2)),
            )
            points_y = torch.linspace(
                center_y - math.sin(angle) * half,
                center_y + math.sin(angle) * half,
                max(16, round(length * 2)),
            )
            _rasterize_path_mask(
                points_x,
                points_y,
                batch_index=index,
                masks=masks,
                width_fraction=line_width_fraction,
            )
        masks = masks.to(device=images.device, dtype=images.dtype)
        return (images * (1 - line_opacity * masks) + line_opacity * masks).clamp(0, 1)
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
    if normalized == "blur_mild":
        return v2.functional.gaussian_blur(
            images,
            kernel_size=[5, 5],
            sigma=[0.8, 0.8],
        )
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
