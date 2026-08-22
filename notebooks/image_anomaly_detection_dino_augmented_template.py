# %% [markdown]
# # Task 2 - DINOv2 native-patch and local-context anomaly scoring
#
# The preceding evidence candidate reached 0.769 at threshold scale 0.85. This experiment freezes
# that selected scale and changes the representation instead of continuing threshold search. It
# averages nearest-normal distances from native DINO patches (degree 1) and 3x3 locally aggregated
# DINO patches (degree 3) before top-1% tail scoring. Separate memory banks protect both scales.

# %% [markdown]
# ## Colab bootstrap

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import hashlib
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from olp_ai_26.core.colab import (
    ColabPaths,
    dataloader_kwargs,
    gpu_report,
    mount_google_drive,
    sync_artifacts,
)
from olp_ai_26.core.config import seed_everything
from olp_ai_26.core.inspect import dataset_report
from olp_ai_26.core.split import make_split
from olp_ai_26.cv.anomaly_detection import (
    AnomalyImageDataset,
    DinoV2PatchFeatureExtractor,
    aggregate_patch_neighborhoods,
    apply_normal_augmentation,
    apply_synthetic_anomaly,
    calibrate_anomaly_threshold,
    fit_positive_evidence_head,
    load_official_training_table,
    patch_memory_distances,
    positive_evidence_scores,
    sample_memory_bank,
    select_anomaly_threshold,
    stage_official_task2_data,
    summarize_patch_distances,
    synthetic_anomaly_defaults,
    validate_anomaly_submission,
)

# %% [markdown]
# ## 0. Run configuration
#
# `category_policy` restores the previous category-specific normal transforms. For the narrowest
# ablation, choose `blur_only`. Neither profile creates anomaly labels.

# %%
TEAM_NAME = "replace_team_name"
TASK_NAME = "task2"
PHASE = "public"  # public | private
RUN_TRAINING = True
EXPERIMENT_NAME = "anomalydino_local_context_13"

OFFICIAL_DATA_SOURCE = (
    Path("/content/drive/MyDrive/olpai26/ThiChinhThucData.zip")
    if "google.colab" in sys.modules
    else Path.home() / "Downloads" / "ThiChinhThucData.zip"
)
PERSISTENT_DIR = (
    Path("/content/drive/MyDrive/olpai26/task2_artifacts")
    if "google.colab" in sys.modules
    else None
)
PRIVATE_ZIP_PASSWORD = None

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
MODEL_CHECKPOINT = None
PRETRAINED_ALLOWED = True
IMAGE_SIZE = 448
BATCH_SIZE = 8
CLEAN_MEMORY_PATCHES = 32768
AUGMENTED_MEMORY_PATCHES = 16384
TOP_FRACTION = 0.01
DISTANCE_CHUNK_SIZE = 256
NORMAL_VALID_SIZE = 0.20
# MuSc's classification ablation favored degrees {1,3}; degree 5 can smooth away small defects.
# Each degree receives a separate full-sized memory so context patches never displace native ones.
NEIGHBORHOOD_DEGREES = (1, 3)
SELECTED_THRESHOLD_SCALE = 0.85
AUGMENTATION_PROFILE = "category_policy"  # category_policy | blur_only
PERSIST_AUGMENTATION_AUDIT = True
AUGMENTATION_AUDIT_SAMPLES = 4
SHOW_AUGMENTATION_PLOTS = True

# Set this to False to isolate the multi-degree DINO representation. When True, a separate,
# non-negative synthetic-evidence term is added without changing either normal memory.
ENABLE_SYNTHETIC_EVIDENCE = True
SYNTHETIC_EVIDENCE_WEIGHT = 0.25
EVIDENCE_TRAIN_FRACTION = 0.25
EVIDENCE_NORMAL_MARGIN_QUANTILE = 0.95

SEED = 42
NUM_WORKERS = 2
if PHASE not in {"public", "private"}:
    raise ValueError("PHASE must be 'public' or 'private'")
if PHASE == "private" and RUN_TRAINING:
    raise ValueError("Private is inference-only: load the frozen public bundle")
if AUGMENTATION_PROFILE not in {"category_policy", "blur_only"}:
    raise ValueError("AUGMENTATION_PROFILE must be 'category_policy' or 'blur_only'")
if CLEAN_MEMORY_PATCHES < 1 or AUGMENTED_MEMORY_PATCHES < 1:
    raise ValueError("Both memory allocations must be positive")
if not 0 < EVIDENCE_TRAIN_FRACTION < 1:
    raise ValueError("EVIDENCE_TRAIN_FRACTION must be between zero and one")
if SYNTHETIC_EVIDENCE_WEIGHT < 0:
    raise ValueError("SYNTHETIC_EVIDENCE_WEIGHT cannot be negative")
if not NEIGHBORHOOD_DEGREES or any(
    degree < 1 or degree % 2 == 0 for degree in NEIGHBORHOOD_DEGREES
):
    raise ValueError("NEIGHBORHOOD_DEGREES must contain positive odd integers")
if 1 not in NEIGHBORHOOD_DEGREES:
    raise ValueError("NEIGHBORHOOD_DEGREES must retain native degree 1")
if SELECTED_THRESHOLD_SCALE <= 0:
    raise ValueError("SELECTED_THRESHOLD_SCALE must be positive")

if "google.colab" in sys.modules and (
    str(OFFICIAL_DATA_SOURCE).startswith("/content/drive") or PERSISTENT_DIR
):
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
official_paths = stage_official_task2_data(
    OFFICIAL_DATA_SOURCE,
    paths.data_dir / "task2_extracted",
    phase=PHASE,
    private_password=PRIVATE_ZIP_PASSWORD,
)
TRAIN_ROOT = official_paths.training_root
TEST_ROOT = official_paths.test_root
TEST_CSV = official_paths.test_csv
EXPERIMENT_DIR = paths.output_dir / EXPERIMENT_NAME
BUNDLE_NAME = f"task2_{EXPERIMENT_NAME}_bundle.pt"
BUNDLE_PATH = (
    EXPERIMENT_DIR / BUNDLE_NAME
    if RUN_TRAINING or paths.persistent_dir is None
    else paths.persistent_dir / EXPERIMENT_NAME / BUNDLE_NAME
)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

seed_everything(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
loader_options = dataloader_kwargs(DEVICE, NUM_WORKERS)
print(gpu_report())
print(dataset_report(paths.data_dir, image_limit=100)["counts"])

# %% [markdown]
# ## 1. Explicit category augmentation policy
#
# These methods expand only the second, known-normal memory. The clean bank always receives its
# entire 32,768-patch allocation. Edit one category at a time after inspecting the saved contact
# sheets. Do not place CutMix, dark curves, white lines, or other suspected defects here.

# %%
CATEGORIES = tuple(f"category_{index:02d}" for index in range(1, 7))
CATEGORY_AUGMENTATIONS = {
    "category_01": ("blur_mild", "rot180", "contrast_down", "contrast_up"),
    "category_02": ("blur_mild", "brightness_down", "brightness_up", "contrast_up"),
    "category_03": ("blur_mild", "hflip", "vflip", "rot90", "rot270"),
    "category_04": ("blur_mild", "brightness_down", "brightness_up", "contrast_up"),
    "category_05": ("blur_mild", "hflip", "vflip", "rot90", "rot270"),
    "category_06": (
        "blur_mild",
        "brightness_down",
        "brightness_up",
        "contrast_up",
        "cool",
        "warm",
    ),
}
BLUR_ONLY_AUGMENTATIONS = {category: ("blur_mild",) for category in CATEGORIES}
ACTIVE_AUGMENTATIONS = (
    CATEGORY_AUGMENTATIONS if AUGMENTATION_PROFILE == "category_policy" else BLUR_ONLY_AUGMENTATIONS
)
NORMAL_QUANTILES = {
    "category_01": 0.99,
    "category_02": 0.975,
    "category_03": 0.975,
    "category_04": 0.975,
    "category_05": 0.975,
    "category_06": 0.95,
}

# Blur stays in the normal-memory policy above. These four transformations are synthetic positives
# only: they never enter either normal bank. Remove one name from a category tuple for a clean
# method ablation. The weaker MixUp range is deliberate so the generated samples remain plausible.
SYNTHETIC_ANOMALY_POLICIES = {
    category: ("mixup", "cutmix", "dark_curve", "white_line") for category in CATEGORIES
}
SYNTHETIC_PARAMETERS_BY_CATEGORY = {
    category: synthetic_anomaly_defaults() for category in CATEGORIES
}
for category_parameters in SYNTHETIC_PARAMETERS_BY_CATEGORY.values():
    category_parameters["mixup"]["mixup_alpha_range"] = (0.10, 0.25)

config_table = pd.DataFrame(
    {
        category: {
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "clean_memory_patches": CLEAN_MEMORY_PATCHES,
            "augmented_memory_patches": AUGMENTED_MEMORY_PATCHES,
            "neighborhood_degrees": NEIGHBORHOOD_DEGREES,
            "selected_threshold_scale": SELECTED_THRESHOLD_SCALE,
            "normal_augmentations": ACTIVE_AUGMENTATIONS[category],
            "synthetic_positives": (
                SYNTHETIC_ANOMALY_POLICIES[category] if ENABLE_SYNTHETIC_EVIDENCE else ()
            ),
            "positive_evidence_weight": (
                SYNTHETIC_EVIDENCE_WEIGHT if ENABLE_SYNTHETIC_EVIDENCE else 0.0
            ),
            "normal_quantile": NORMAL_QUANTILES[category],
            "distance": "cosine_1nn",
            "image_score": f"mean_top_{TOP_FRACTION:.2%}_patches",
        }
        for category in CATEGORIES
    }
).T
print(config_table)

# %% [markdown]
# ## 2. Validate the official data contract

# %%
train_table = load_official_training_table(TRAIN_ROOT)
test_table = pd.read_csv(TEST_CSV)
required_test_columns = ["sample_id", "category", "relative_path"]
if list(test_table.columns) != required_test_columns:
    raise ValueError(f"test.csv columns must be exactly {required_test_columns}")
if test_table["sample_id"].duplicated().any():
    raise ValueError("test.csv contains duplicate sample_id values")
expected_train_counts = {
    "category_01": 664,
    "category_02": 664,
    "category_03": 302,
    "category_04": 660,
    "category_05": 660,
    "category_06": 210,
}
actual_train_counts = train_table["category"].value_counts().sort_index().to_dict()
if actual_train_counts != expected_train_counts:
    raise ValueError(f"Unexpected training counts: {actual_train_counts}")
expected_test_count = 80 if PHASE == "public" else 160
actual_test_counts = test_table["category"].value_counts().sort_index().to_dict()
if actual_test_counts != {category: expected_test_count for category in CATEGORIES}:
    raise ValueError(f"Unexpected {PHASE} counts: {actual_test_counts}")
print("Train counts:\n", train_table["category"].value_counts().sort_index())
print("Test counts:\n", test_table["category"].value_counts().sort_index())

# %% [markdown]
# ## 3. Plot and persist every configured augmentation before feature extraction

# %%


def preview_category_augmentations(
    category: str,
    *,
    selected_rows: pd.DataFrame,
    save_path: Path,
):
    """Render the exact normal and synthetic transforms configured for one category.

    The figure is a required pre-training audit: use it to judge whether severity and geometry
    resemble the observed competition images before spending Colab GPU time.
    """
    dataset = AnomalyImageDataset(selected_rows, root=TRAIN_ROOT, image_size=IMAGE_SIZE)
    images = torch.stack([dataset[index] for index in range(len(dataset))])
    panels = [("original", images[0])]
    panels.extend(
        (method, apply_normal_augmentation(images, method)[0])
        for method in ACTIVE_AUGMENTATIONS[category]
    )
    if ENABLE_SYNTHETIC_EVIDENCE:
        panels.extend(
            (
                f"synthetic: {method}",
                apply_synthetic_anomaly(
                    images,
                    method,
                    seed=SEED + method_index,
                    **SYNTHETIC_PARAMETERS_BY_CATEGORY[category][method],
                )[0],
            )
            for method_index, method in enumerate(SYNTHETIC_ANOMALY_POLICIES[category])
        )
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    flat_axes = axes.reshape(-1)
    for axis, (title, image) in zip(flat_axes, panels, strict=False):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(title)
        axis.axis("off")
    for axis in flat_axes[len(panels) :]:
        axis.axis("off")
    figure.suptitle(f"{category}: normal-memory versus synthetic-positive transforms", fontsize=14)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def export_normal_augmentation_audit() -> pd.DataFrame:
    """Save representative PNGs, contact sheets, and the exact augmentation manifest.

    Clean and known-normal outputs are stored separately from synthetic-positive outputs so it is
    visually obvious which samples enter normal memory and which train only the auxiliary head.
    """
    payload = json.dumps(
        {
            "profile": AUGMENTATION_PROFILE,
            "augmentations": ACTIVE_AUGMENTATIONS,
            "synthetic_evidence_enabled": ENABLE_SYNTHETIC_EVIDENCE,
            "synthetic_policies": SYNTHETIC_ANOMALY_POLICIES,
            "synthetic_parameters": SYNTHETIC_PARAMETERS_BY_CATEGORY,
            "synthetic_evidence_weight": SYNTHETIC_EVIDENCE_WEIGHT,
            "clean_memory_patches": CLEAN_MEMORY_PATCHES,
            "augmented_memory_patches": AUGMENTED_MEMORY_PATCHES,
            "neighborhood_degrees": NEIGHBORHOOD_DEGREES,
            "selected_threshold_scale": SELECTED_THRESHOLD_SCALE,
        },
        sort_keys=True,
    )
    config_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    audit_root = EXPERIMENT_DIR / "augmentation_audit" / config_id
    records = []
    for category in CATEGORIES:
        category_rows = train_table.loc[train_table["category"] == category]
        count = min(AUGMENTATION_AUDIT_SAMPLES, len(category_rows))
        positions = np.linspace(0, len(category_rows) - 1, num=count, dtype=int)
        selected = category_rows.iloc[positions].reset_index(drop=True)
        dataset = AnomalyImageDataset(selected, root=TRAIN_ROOT, image_size=IMAGE_SIZE)
        images = torch.stack([dataset[index] for index in range(len(dataset))])
        transforms = [("identity", images)]
        transforms.extend(
            (method, apply_normal_augmentation(images, method))
            for method in ACTIVE_AUGMENTATIONS[category]
        )
        if ENABLE_SYNTHETIC_EVIDENCE:
            transforms.extend(
                (
                    method,
                    apply_synthetic_anomaly(
                        images,
                        method,
                        seed=SEED + method_index,
                        **SYNTHETIC_PARAMETERS_BY_CATEGORY[category][method],
                    ),
                )
                for method_index, method in enumerate(SYNTHETIC_ANOMALY_POLICIES[category])
            )
        for method, transformed in transforms:
            if method == "identity":
                transform_type = "clean"
            elif method in ACTIVE_AUGMENTATIONS[category]:
                transform_type = "augmented_normal"
            else:
                transform_type = "synthetic_positive"
            for index, row in selected.iterrows():
                output = audit_root / category / transform_type / method / f"{row.sample_id}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                save_image(transformed[index], output)
                records.append(
                    {
                        "config_id": config_id,
                        "category": category,
                        "sample_id": row.sample_id,
                        "transform_type": transform_type,
                        "method": method,
                        "output_relative_path": output.relative_to(EXPERIMENT_DIR).as_posix(),
                    }
                )
        figure = preview_category_augmentations(
            category,
            selected_rows=selected,
            save_path=audit_root / category / "contact_sheet.png",
        )
        if SHOW_AUGMENTATION_PLOTS:
            print(f"Normal and synthetic augmentation preview: {category}")
            plt.show()
        plt.close(figure)
    manifest = pd.DataFrame(records)
    manifest.to_csv(audit_root / "augmentation_manifest.csv", index=False)
    (audit_root / "augmentation_config.json").write_text(payload, encoding="utf-8")
    copied = sync_artifacts(
        paths.output_dir,
        paths.persistent_dir,
        patterns=("*.png", "augmentation_manifest.csv", "augmentation_config.json"),
    )
    print("Augmentation audit:", audit_root)
    print("Files copied to persistent storage:", len(copied))
    return manifest


augmentation_manifest = None
if RUN_TRAINING and PERSIST_AUGMENTATION_AUDIT:
    augmentation_manifest = export_normal_augmentation_audit()
    print(augmentation_manifest.head())

# %% [markdown]
# ## 4. DINOv2 memory and scoring helpers

# %%


def build_extractor(*, load_pretrained: bool) -> DinoV2PatchFeatureExtractor:
    """Build the frozen DINOv2-S/14 extractor used by both public and private inference."""
    return (
        DinoV2PatchFeatureExtractor(
            MODEL_NAME,
            image_size=IMAGE_SIZE,
            pretrained_allowed=PRETRAINED_ALLOWED and load_pretrained,
            checkpoint_path=MODEL_CHECKPOINT if load_pretrained else None,
        )
        .to(DEVICE)
        .eval()
    )


@torch.inference_mode()
def iter_memory_patch_batches(
    frame: pd.DataFrame,
    extractor: DinoV2PatchFeatureExtractor,
    augmentations: tuple[str, ...],
    *,
    neighborhood_degree: int,
):
    """Yield one neighborhood degree of DINO tokens for known-normal transformations."""
    dataset = AnomalyImageDataset(frame, root=TRAIN_ROOT, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    for images in loader:
        images = images.to(DEVICE)
        for augmentation in augmentations:
            transformed = apply_normal_augmentation(images, augmentation)
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                embeddings = extractor(transformed).float()
            yield aggregate_patch_neighborhoods(
                embeddings,
                kernel_size=neighborhood_degree,
            ).cpu()


def combined_memory_bank(
    clean_memory: torch.Tensor,
    augmented_memory: torch.Tensor,
) -> torch.Tensor:
    """Concatenate protected clean and augmented banks for exact nearest-neighbour lookup."""
    if clean_memory.ndim != 2 or augmented_memory.ndim != 2:
        raise ValueError("Both memory banks must be matrices")
    if clean_memory.shape[1] != augmented_memory.shape[1]:
        raise ValueError("Clean and augmented memory dimensions differ")
    return torch.cat((clean_memory, augmented_memory), dim=0)


@torch.inference_mode()
def score_frame(
    frame: pd.DataFrame,
    root: Path,
    extractor: DinoV2PatchFeatureExtractor,
    clean_memories: dict[int, torch.Tensor],
    augmented_memories: dict[int, torch.Tensor],
    *,
    synthetic_method: str | None = None,
) -> tuple[np.ndarray, torch.Tensor, dict[int, np.ndarray]]:
    """Return fused DINO scores, summary features, and per-degree scores.

    Args:
        frame: Metadata rows containing each image's relative path.
        root: Directory against which the relative paths are resolved.
        extractor: Frozen DINOv2 patch-token extractor.
        clean_memories: Protected original-normal memory for every neighborhood degree.
        augmented_memories: Known-normal transformed memory for every neighborhood degree.
        synthetic_method: Optional synthetic-positive transform applied before feature extraction.

    Returns:
        The fused top-1%-tail score, six fused patch-distance summaries, and a mapping containing
        each degree's standalone image scores. Fusion happens per patch before tail aggregation.
    """
    dataset = AnomalyImageDataset(frame, root=root, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    banks = {
        degree: combined_memory_bank(clean_memories[degree], augmented_memories[degree]).to(DEVICE)
        for degree in NEIGHBORHOOD_DEGREES
    }
    scores = []
    feature_rows = []
    degree_score_rows = {degree: [] for degree in NEIGHBORHOOD_DEGREES}
    for batch_index, images in enumerate(loader):
        if synthetic_method is not None:
            # MixUp needs a distinct partner. Supply the first dataset image if the final loader
            # batch contains only one row; otherwise the shared helper pairs rows within the batch.
            single_mixup = synthetic_method == "mixup" and len(images) == 1
            transform_input = (
                torch.cat((images, dataset[0].unsqueeze(0)), dim=0) if single_mixup else images
            )
            transformed = apply_synthetic_anomaly(
                transform_input,
                synthetic_method,
                seed=SEED + batch_index,
                **SYNTHETIC_PARAMETERS_BY_CATEGORY[str(frame["category"].iloc[0])][
                    synthetic_method
                ],
            )
            images = transformed[:1] if single_mixup else transformed
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            embeddings = extractor(images.to(DEVICE)).float()
        degree_distances = []
        for degree in NEIGHBORHOOD_DEGREES:
            aggregated = aggregate_patch_neighborhoods(embeddings, kernel_size=degree)
            patch_distances = patch_memory_distances(
                aggregated,
                banks[degree],
                distance_metric="cosine",
                distance_chunk_size=DISTANCE_CHUNK_SIZE,
            )
            degree_distances.append(patch_distances)
            degree_features = summarize_patch_distances(
                patch_distances,
                top_k=1,
                top_fraction=TOP_FRACTION,
            )
            degree_score_rows[degree].extend(degree_features[:, -1].cpu().tolist())
        fused_distances = torch.stack(degree_distances).mean(dim=0)
        features = summarize_patch_distances(
            fused_distances,
            top_k=1,
            top_fraction=TOP_FRACTION,
        ).cpu()
        feature_rows.append(features)
        scores.extend(features[:, -1].tolist())
    return (
        np.asarray(scores, dtype=np.float64),
        torch.cat(feature_rows),
        {
            degree: np.asarray(values, dtype=np.float64)
            for degree, values in degree_score_rows.items()
        },
    )


def combine_anomaly_evidence(
    dino_scores: np.ndarray,
    features: torch.Tensor,
    artifact: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Add known-defect evidence without ever subtracting from the open-set DINO score.

    The logistic head recognizes the synthetic directions only. Its margin-adjusted output is
    clamped at zero, scaled into DINO-score units, and then added. Consequently, an unknown anomaly
    that matches none of MixUp, CutMix, dark-curve, or white-line evidence keeps its original DINO
    score instead of being pushed toward normal.

    Returns:
        ``(positive_evidence, combined_score)`` vectors in input-row order.
    """
    base = np.asarray(dino_scores, dtype=np.float64)
    head = artifact.get("positive_evidence_head")
    weight = float(artifact.get("positive_evidence_weight", 0.0))
    if head is None or weight <= 0:
        return np.zeros_like(base), base.copy()
    evidence = positive_evidence_scores(features, head).numpy().astype(np.float64)
    score_scale = max(float(artifact["dino_score_std"]), 1e-6)
    return evidence, base + weight * score_scale * evidence


# %% [markdown]
# ## 5. Fit normal memories, optional positive-evidence heads, and thresholds
#
# The outer validation images never enter either memory. With synthetic evidence enabled, only 25%
# of that held-out set trains the small six-feature logistic head; the remaining 75% independently
# calibrates the final threshold. The synthetic samples are diagnostic proxy positives, never normal
# memory. Threshold selection remains normal-only, so proxy separability cannot choose the cutoff.

# %%
if RUN_TRAINING:
    extractor = build_extractor(load_pretrained=True)
    model_state = {name: value.cpu() for name, value in extractor.state_dict().items()}
    category_artifacts = {}
    for category, category_rows in train_table.groupby("category", sort=True):
        split = make_split(category_rows, valid_size=NORMAL_VALID_SIZE, seed=SEED)
        clean_memories = {
            degree: sample_memory_bank(
                iter_memory_patch_batches(
                    split.train,
                    extractor,
                    ("identity",),
                    neighborhood_degree=degree,
                ),
                max_patches=CLEAN_MEMORY_PATCHES,
                seed=SEED if degree == 1 else SEED + degree * 100,
            )
            for degree in NEIGHBORHOOD_DEGREES
        }
        augmented_memories = {
            degree: sample_memory_bank(
                iter_memory_patch_batches(
                    split.train,
                    extractor,
                    ACTIVE_AUGMENTATIONS[category],
                    neighborhood_degree=degree,
                ),
                max_patches=AUGMENTED_MEMORY_PATCHES,
                seed=SEED + 1000 if degree == 1 else SEED + 1000 + degree * 100,
            )
            for degree in NEIGHBORHOOD_DEGREES
        }
        evidence_head = None
        dino_score_std = 1.0
        positive_weight = SYNTHETIC_EVIDENCE_WEIGHT if ENABLE_SYNTHETIC_EVIDENCE else 0.0
        calibration_rows = split.valid
        if ENABLE_SYNTHETIC_EVIDENCE:
            evidence_split = make_split(
                split.valid,
                valid_size=1.0 - EVIDENCE_TRAIN_FRACTION,
                seed=SEED + 1,
            )
            evidence_dino_scores, evidence_normal_features, evidence_degree_scores = score_frame(
                evidence_split.train,
                TRAIN_ROOT,
                extractor,
                clean_memories,
                augmented_memories,
            )
            synthetic_training_features = torch.cat(
                [
                    score_frame(
                        evidence_split.train,
                        TRAIN_ROOT,
                        extractor,
                        clean_memories,
                        augmented_memories,
                        synthetic_method=method,
                    )[1]
                    for method in SYNTHETIC_ANOMALY_POLICIES[category]
                ]
            )
            evidence_head = fit_positive_evidence_head(
                evidence_normal_features,
                synthetic_training_features,
                seed=SEED,
                normal_margin_quantile=EVIDENCE_NORMAL_MARGIN_QUANTILE,
            )
            dino_score_std = max(float(evidence_dino_scores.std()), 1e-6)
            calibration_rows = evidence_split.valid

        artifact_for_scoring = {
            "positive_evidence_head": evidence_head,
            "positive_evidence_weight": positive_weight,
            "dino_score_std": dino_score_std,
        }
        normal_dino_scores, normal_features, normal_degree_scores = score_frame(
            calibration_rows,
            TRAIN_ROOT,
            extractor,
            clean_memories,
            augmented_memories,
        )
        normal_evidence, normal_scores = combine_anomaly_evidence(
            normal_dino_scores,
            normal_features,
            artifact_for_scoring,
        )
        control_normal_scores = (
            np.concatenate((evidence_dino_scores, normal_dino_scores))
            if ENABLE_SYNTHETIC_EVIDENCE
            else normal_dino_scores
        )
        control_calibration = calibrate_anomaly_threshold(
            control_normal_scores,
            normal_quantile=NORMAL_QUANTILES[category],
        )
        dino_only_threshold = select_anomaly_threshold(control_calibration, "normal_quantile")
        native_normal_scores = (
            np.concatenate((evidence_degree_scores[1], normal_degree_scores[1]))
            if ENABLE_SYNTHETIC_EVIDENCE
            else normal_degree_scores[1]
        )
        native_calibration = calibrate_anomaly_threshold(
            native_normal_scores,
            normal_quantile=NORMAL_QUANTILES[category],
        )
        native_threshold = select_anomaly_threshold(native_calibration, "normal_quantile")
        synthetic_score_parts = []
        if ENABLE_SYNTHETIC_EVIDENCE:
            for method in SYNTHETIC_ANOMALY_POLICIES[category]:
                synthetic_dino_scores, synthetic_features, _ = score_frame(
                    calibration_rows,
                    TRAIN_ROOT,
                    extractor,
                    clean_memories,
                    augmented_memories,
                    synthetic_method=method,
                )
                _, synthetic_scores = combine_anomaly_evidence(
                    synthetic_dino_scores,
                    synthetic_features,
                    artifact_for_scoring,
                )
                synthetic_score_parts.append(synthetic_scores)
        synthetic_scores = np.concatenate(synthetic_score_parts) if synthetic_score_parts else None
        calibration = calibrate_anomaly_threshold(
            normal_scores,
            synthetic_scores,
            normal_quantile=NORMAL_QUANTILES[category],
        )
        calibration["selected_threshold"] = select_anomaly_threshold(calibration, "normal_quantile")
        calibration.update(
            {
                "normal_score_mean": float(normal_scores.mean()),
                "normal_score_std": float(normal_scores.std()),
                "normal_dino_score_mean": float(normal_dino_scores.mean()),
                "normal_dino_score_std": float(normal_dino_scores.std()),
                "normal_positive_evidence_mean": float(normal_evidence.mean()),
                "normal_positive_evidence_max": float(normal_evidence.max()),
                "normal_validation_images": len(normal_scores),
                "dino_only_validation_images": len(control_normal_scores),
                "dino_only_threshold": dino_only_threshold,
                "native_degree_one_threshold": native_threshold,
                "evidence_training_images": (
                    len(evidence_split.train) if ENABLE_SYNTHETIC_EVIDENCE else 0
                ),
                "clean_memory_patches_by_degree": {
                    degree: len(memory) for degree, memory in clean_memories.items()
                },
                "augmented_memory_patches_by_degree": {
                    degree: len(memory) for degree, memory in augmented_memories.items()
                },
            }
        )
        category_artifacts[category] = {
            "clean_memories": clean_memories,
            "augmented_memories": augmented_memories,
            "normal_augmentations": ACTIVE_AUGMENTATIONS[category],
            "synthetic_anomalies": (
                SYNTHETIC_ANOMALY_POLICIES[category] if ENABLE_SYNTHETIC_EVIDENCE else ()
            ),
            "synthetic_parameters": SYNTHETIC_PARAMETERS_BY_CATEGORY[category],
            "positive_evidence_head": evidence_head,
            "positive_evidence_weight": positive_weight,
            "dino_score_std": dino_score_std,
            "dino_only_threshold": dino_only_threshold,
            "native_degree_one_threshold": native_threshold,
            "normal_quantile": NORMAL_QUANTILES[category],
            "calibration": calibration,
        }
        print(
            category,
            {
                "augmentations": ACTIVE_AUGMENTATIONS[category],
                "synthetic_positives": (
                    SYNTHETIC_ANOMALY_POLICIES[category] if ENABLE_SYNTHETIC_EVIDENCE else ()
                ),
                "positive_evidence_weight": positive_weight,
                "clean_memory_by_degree": {
                    degree: len(memory) for degree, memory in clean_memories.items()
                },
                "augmented_memory_by_degree": {
                    degree: len(memory) for degree, memory in augmented_memories.items()
                },
                "threshold": calibration["selected_threshold"],
                "dino_only_control_threshold": dino_only_threshold,
                "native_degree_one_threshold": native_threshold,
                "normal_mean": calibration["normal_score_mean"],
                "normal_std": calibration["normal_score_std"],
                "normal_positive_evidence_mean": calibration["normal_positive_evidence_mean"],
                "proxy_balanced_accuracy": calibration["proxy_balanced_accuracy"],
            },
        )
    bundle = {
        "experiment_name": EXPERIMENT_NAME,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "top_fraction": TOP_FRACTION,
        "neighborhood_degrees": NEIGHBORHOOD_DEGREES,
        "selected_threshold_scale": SELECTED_THRESHOLD_SCALE,
        "synthetic_evidence_enabled": ENABLE_SYNTHETIC_EVIDENCE,
        "synthetic_evidence_weight": (
            SYNTHETIC_EVIDENCE_WEIGHT if ENABLE_SYNTHETIC_EVIDENCE else 0.0
        ),
        "seed": SEED,
        "model_state": model_state,
        "categories": category_artifacts,
    }
    torch.save(bundle, BUNDLE_PATH)
    (EXPERIMENT_DIR / "task2_calibration.json").write_text(
        json.dumps(
            {
                category: artifact["calibration"]
                for category, artifact in category_artifacts.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sync_artifacts(paths.output_dir, paths.persistent_dir)
    print("Frozen bundle:", BUNDLE_PATH)
else:
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=True)
    if bundle["experiment_name"] != EXPERIMENT_NAME:
        raise ValueError("Loaded bundle does not match EXPERIMENT_NAME")
    if bundle["augmentation_profile"] != AUGMENTATION_PROFILE:
        raise ValueError("Loaded bundle does not match AUGMENTATION_PROFILE")
    if bundle["synthetic_evidence_enabled"] != ENABLE_SYNTHETIC_EVIDENCE:
        raise ValueError("Loaded bundle does not match ENABLE_SYNTHETIC_EVIDENCE")
    if tuple(bundle["neighborhood_degrees"]) != NEIGHBORHOOD_DEGREES:
        raise ValueError("Loaded bundle does not match NEIGHBORHOOD_DEGREES")
    if bundle["selected_threshold_scale"] != SELECTED_THRESHOLD_SCALE:
        raise ValueError("Loaded bundle does not match SELECTED_THRESHOLD_SCALE")
    extractor = build_extractor(load_pretrained=False)
    extractor.load_state_dict(bundle["model_state"])
    category_artifacts = bundle["categories"]
    print("Loaded frozen categories:", sorted(category_artifacts))

# %% [markdown]
# ## 6. Public/private inference and score audit

# %%
all_dino_scores = np.zeros(len(test_table), dtype=np.float64)
all_positive_evidence = np.zeros(len(test_table), dtype=np.float64)
all_scores = np.zeros(len(test_table), dtype=np.float64)
all_labels = np.zeros(len(test_table), dtype=np.int64)
all_dino_only_labels = np.zeros(len(test_table), dtype=np.int64)
all_native_only_labels = np.zeros(len(test_table), dtype=np.int64)
all_degree_scores = {
    degree: np.zeros(len(test_table), dtype=np.float64) for degree in NEIGHBORHOOD_DEGREES
}
for category, category_rows in test_table.groupby("category", sort=True):
    artifact = category_artifacts[category]
    row_dino_scores, row_features, row_degree_scores = score_frame(
        category_rows,
        TEST_ROOT,
        extractor,
        artifact["clean_memories"],
        artifact["augmented_memories"],
    )
    row_evidence, row_scores = combine_anomaly_evidence(
        row_dino_scores,
        row_features,
        artifact,
    )
    threshold = artifact["calibration"]["selected_threshold"] * SELECTED_THRESHOLD_SCALE
    positions = category_rows.index.to_numpy()
    all_dino_scores[positions] = row_dino_scores
    all_positive_evidence[positions] = row_evidence
    all_scores[positions] = row_scores
    all_labels[positions] = (row_scores >= threshold).astype(np.int64)
    all_dino_only_labels[positions] = (
        row_dino_scores >= artifact["dino_only_threshold"] * SELECTED_THRESHOLD_SCALE
    ).astype(np.int64)
    all_native_only_labels[positions] = (
        row_degree_scores[1] >= artifact["native_degree_one_threshold"] * SELECTED_THRESHOLD_SCALE
    ).astype(np.int64)
    for degree, degree_scores in row_degree_scores.items():
        all_degree_scores[degree][positions] = degree_scores
    print(
        category,
        {
            "threshold": threshold,
            "threshold_scale": SELECTED_THRESHOLD_SCALE,
            "degree_score_medians": {
                degree: float(np.median(values)) for degree, values in row_degree_scores.items()
            },
            "dino_score_median": float(np.median(row_dino_scores)),
            "positive_evidence_images": int((row_evidence > 0).sum()),
            "positive_evidence_max": float(row_evidence.max()),
            "combined_score_median": float(np.median(row_scores)),
            "predicted_anomalies": int((row_scores >= threshold).sum()),
        },
    )

audit = test_table.copy()
for degree, degree_scores in all_degree_scores.items():
    audit[f"dino_degree_{degree}_score"] = degree_scores
audit["dino_score"] = all_dino_scores
audit["positive_evidence"] = all_positive_evidence
audit["combined_score"] = all_scores
audit["label"] = all_labels
audit.to_csv(EXPERIMENT_DIR / f"{TASK_NAME}_{PHASE}_scores.csv", index=False)

# %% [markdown]
# ## 7. Exact submission writer

# %%


def write_submission_candidate(
    labels: np.ndarray,
    tag: str = "main",
) -> tuple[pd.DataFrame, Path]:
    """Write one isolated candidate with the exact official CSV and ZIP structure."""
    candidate_dir = EXPERIMENT_DIR / f"candidate_{tag}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    submission = test_table[["sample_id", "category"]].copy()
    submission["label"] = np.asarray(labels, dtype=np.int64)
    validate_anomaly_submission(submission, test_table)
    csv_name = f"{TASK_NAME}_{PHASE}_output.csv"
    csv_path = candidate_dir / csv_name
    submission.to_csv(csv_path, index=False, encoding="utf-8")
    zip_suffix = "pub" if PHASE == "public" else "pri"
    zip_path = candidate_dir / f"{TEAM_NAME}_{TASK_NAME}_{zip_suffix}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_name)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != [csv_name]:
            raise RuntimeError("Submission ZIP must contain exactly the required CSV")
    return submission, zip_path


submission, zip_path = write_submission_candidate(all_labels)
if ENABLE_SYNTHETIC_EVIDENCE:
    control_submission, control_zip_path = write_submission_candidate(
        all_dino_only_labels,
        tag="multidegree_no_synthetic",
    )
native_submission, native_zip_path = write_submission_candidate(
    all_native_only_labels,
    tag="native_degree_1_no_synthetic",
)
sync_artifacts(
    paths.output_dir,
    paths.persistent_dir,
    patterns=("*.pt", "*.json", "*.csv", "*.zip"),
)
print(zip_path, submission["label"].value_counts().to_dict())
if ENABLE_SYNTHETIC_EVIDENCE:
    print(
        control_zip_path,
        control_submission["label"].value_counts().to_dict(),
        "Multi-degree no-synthetic ablation",
    )
print(
    native_zip_path,
    native_submission["label"].value_counts().to_dict(),
    "Native degree-1 no-synthetic ablation",
)
submission.head()

# %% [markdown]
# ## 8. Candidate interpretation
#
# Threshold scale 0.85 is frozen from the previous run. The three candidates differ by method:
# `main` uses multi-degree context plus synthetic evidence; `multidegree_no_synthetic` isolates
# context; `native_degree_1_no_synthetic` removes both context and synthetic evidence. Do not resume
# a dense scale sweep: submit these candidates to measure representation changes.

# %%
candidate_summary = pd.DataFrame(
    {
        "candidate_main": all_labels,
        "multidegree_no_synthetic": all_dino_only_labels,
        "native_degree_1_no_synthetic": all_native_only_labels,
    }
).sum()
print("Predicted anomalies by method at fixed scale 0.85:")
print(candidate_summary)
