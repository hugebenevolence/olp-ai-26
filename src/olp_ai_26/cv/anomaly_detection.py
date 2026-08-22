"""Normal-only image anomaly detection with frozen timm features and patch memory banks.

The implementation is intentionally small enough for a six-hour contest. It learns one normal
feature memory bank and one decision threshold per category. Higher scores mean more anomalous.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    """Concatenate patch batches and retain a deterministic random memory subset on CPU."""
    flattened = [batch.detach().float().cpu().flatten(0, 1) for batch in patch_batches]
    if not flattened:
        raise ValueError("At least one patch-embedding batch is required")
    patches = torch.cat(flattened)
    if len(patches) <= max_patches:
        return patches.contiguous()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(patches), generator=generator)[:max_patches]
    return patches[indices].contiguous()


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
