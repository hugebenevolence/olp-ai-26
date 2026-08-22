# %% [markdown]
# # Task 2 - DINOv2 local-context scoring with three frozen 0.85 modes
#
# The preceding candidate reached 0.769 at threshold scale 0.85. This notebook freezes that scale
# and exposes three controlled modes: the original public baseline, the private-inspection-driven
# augmentation without TTA, and the same new augmentation with matched calibration-time TTA. Each
# run trains or loads one isolated bundle and writes exactly one submission ZIP.

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
    changed_patch_mask,
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
# Change only `EXPERIMENT_MODE` for the requested comparison. `category_policy` controls
# known-normal memory transforms; `blur_only` is retained as a diagnostic. Neither normal profile
# creates anomaly labels. The synthetic profile is selected by the mode and is not edited manually.

# %%
TEAM_NAME = "replace_team_name"
TASK_NAME = "task2"
PHASE = "public"  # public | private
RUN_TRAINING = True
# Choose exactly one. Train each mode on public first. The two new-augmentation modes can later be
# rerun with PHASE="private" and RUN_TRAINING=False to load their frozen public bundles.
EXPERIMENT_MODE = "085_new_aug_tta"
EXPERIMENT_MODES = {
    "085_original_public": {
        "synthetic_profile": "original",
        "tta": False,
        "public_only": True,
    },
    "085_new_aug_no_tta": {
        "synthetic_profile": "private_inspection_v1",
        "tta": False,
        "public_only": False,
    },
    "085_new_aug_tta": {
        "synthetic_profile": "private_inspection_v1",
        "tta": True,
        "public_only": False,
    },
}
if EXPERIMENT_MODE not in EXPERIMENT_MODES:
    raise ValueError(f"EXPERIMENT_MODE must be one of {tuple(EXPERIMENT_MODES)}")
MODE_CONFIG = EXPERIMENT_MODES[EXPERIMENT_MODE]
SYNTHETIC_PROFILE = str(MODE_CONFIG["synthetic_profile"])
ENABLE_TTA = bool(MODE_CONFIG["tta"])
EXPERIMENT_NAME = f"anomalydino_{EXPERIMENT_MODE}"

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

# The patch adapter is a diagnostic only and does not enter the accepted scale-0.85 score. Leave it
# disabled for the three requested runs to save Colab time. If enabled for analysis, MixUp remains
# excluded because nearly every pixel changes and has no credible local patch boundary.
ENABLE_PATCH_ADAPTER = False
PATCH_ADAPTER_FEATURES_PER_CLASS = 8192
PATCH_ADAPTER_FEATURES_PER_METHOD = 2048
PATCH_ADAPTER_DIFFERENCE_THRESHOLD = 0.015 if SYNTHETIC_PROFILE == "original" else 0.008
PATCH_ADAPTER_MASK_DILATION = 1
PATCH_ADAPTER_NORMAL_MARGIN_QUANTILE = 0.95

SEED = 42
NUM_WORKERS = 2
if PHASE not in {"public", "private"}:
    raise ValueError("PHASE must be 'public' or 'private'")
if PHASE == "private" and RUN_TRAINING:
    raise ValueError("Private is inference-only: load the frozen public bundle")
if bool(MODE_CONFIG["public_only"]) and PHASE != "public":
    raise ValueError("085_original_public is intentionally restricted to the public phase")
if AUGMENTATION_PROFILE not in {"category_policy", "blur_only"}:
    raise ValueError("AUGMENTATION_PROFILE must be 'category_policy' or 'blur_only'")
if CLEAN_MEMORY_PATCHES < 1 or AUGMENTED_MEMORY_PATCHES < 1:
    raise ValueError("Both memory allocations must be positive")
if not 0 < EVIDENCE_TRAIN_FRACTION < 1:
    raise ValueError("EVIDENCE_TRAIN_FRACTION must be between zero and one")
if SYNTHETIC_EVIDENCE_WEIGHT < 0:
    raise ValueError("SYNTHETIC_EVIDENCE_WEIGHT cannot be negative")
if PATCH_ADAPTER_FEATURES_PER_CLASS < 1:
    raise ValueError("PATCH_ADAPTER_FEATURES_PER_CLASS must be positive")
if PATCH_ADAPTER_FEATURES_PER_METHOD < 1:
    raise ValueError("PATCH_ADAPTER_FEATURES_PER_METHOD must be positive")
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
# Use only geometry already accepted as normal for that category. Categories 02, 04, and 06 keep
# identity-only inference because brightness/color changes could erase real anomaly evidence.
CATEGORY_TTA = {
    "category_01": ("identity",),
    "category_02": ("identity",),
    "category_03": ("identity", "hflip", "vflip"),
    "category_04": ("identity",),
    "category_05": ("identity", "hflip", "vflip"),
    "category_06": ("identity",),
}
if any(not methods or methods[0] != "identity" for methods in CATEGORY_TTA.values()):
    raise ValueError("Every CATEGORY_TTA policy must begin with identity")
NORMAL_QUANTILES = {
    "category_01": 0.99,
    "category_02": 0.975,
    "category_03": 0.975,
    "category_04": 0.975,
    "category_05": 0.975,
    "category_06": 0.95,
}

# Blur and flips stay in the normal-memory policy above. Synthetic methods never enter either
# normal bank. The original profile reproduces the previous public candidate exactly; the new
# profile prioritizes MixUp and replaces strong black/straight marks with subtle colored curves.
if SYNTHETIC_PROFILE == "original":
    SYNTHETIC_METHODS = ("mixup", "cutmix", "dark_curve", "white_line")
    PATCH_ADAPTER_METHODS = ("cutmix", "dark_curve", "white_line")
    IMAGE_EVIDENCE_METHOD_WEIGHTS = {method: 0.25 for method in SYNTHETIC_METHODS}
    PATCH_ADAPTER_METHOD_WEIGHTS = {method: 1 / 3 for method in PATCH_ADAPTER_METHODS}
else:
    SYNTHETIC_METHODS = ("mixup", "cutmix", "gray_curve", "white_curve")
    PATCH_ADAPTER_METHODS = ("cutmix", "gray_curve", "white_curve")
    IMAGE_EVIDENCE_METHOD_WEIGHTS = {
        "mixup": 0.50,
        "cutmix": 0.10,
        "gray_curve": 0.20,
        "white_curve": 0.20,
    }
    PATCH_ADAPTER_METHOD_WEIGHTS = {
        "cutmix": 0.15,
        "gray_curve": 0.425,
        "white_curve": 0.425,
    }
SYNTHETIC_ANOMALY_POLICIES = {category: SYNTHETIC_METHODS for category in CATEGORIES}
SYNTHETIC_PARAMETERS_BY_CATEGORY = {
    category: synthetic_anomaly_defaults() for category in CATEGORIES
}
for category_parameters in SYNTHETIC_PARAMETERS_BY_CATEGORY.values():
    category_parameters["mixup"]["mixup_alpha_range"] = (0.10, 0.25)
    if SYNTHETIC_PROFILE != "original":
        category_parameters["cutmix"].update(
            {
                "cutmix_area_range": (0.02, 0.07),
                "cutmix_opacity_range": (0.04, 0.10),
            }
        )
if not set(PATCH_ADAPTER_METHODS).issubset(set(SYNTHETIC_METHODS) - {"mixup"}):
    raise ValueError("PATCH_ADAPTER_METHODS must be local methods from SYNTHETIC_METHODS")
if not np.isclose(sum(IMAGE_EVIDENCE_METHOD_WEIGHTS.values()), 1.0):
    raise ValueError("IMAGE_EVIDENCE_METHOD_WEIGHTS must sum to one")
if not np.isclose(sum(PATCH_ADAPTER_METHOD_WEIGHTS.values()), 1.0):
    raise ValueError("PATCH_ADAPTER_METHOD_WEIGHTS must sum to one")
AUDIT_SYNTHETIC_METHODS = tuple(
    dict.fromkeys(
        (SYNTHETIC_ANOMALY_POLICIES[CATEGORIES[0]] if ENABLE_SYNTHETIC_EVIDENCE else ())
        + (PATCH_ADAPTER_METHODS if ENABLE_PATCH_ADAPTER else ())
    )
)

config_table = pd.DataFrame(
    {
        category: {
            "experiment_mode": EXPERIMENT_MODE,
            "synthetic_profile": SYNTHETIC_PROFILE,
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
            "patch_adapter_methods": PATCH_ADAPTER_METHODS if ENABLE_PATCH_ADAPTER else (),
            "patch_adapter_features_per_class": PATCH_ADAPTER_FEATURES_PER_CLASS,
            "image_evidence_method_weights": IMAGE_EVIDENCE_METHOD_WEIGHTS,
            "patch_adapter_method_weights": PATCH_ADAPTER_METHOD_WEIGHTS,
            "patch_adapter_mask_dilation": PATCH_ADAPTER_MASK_DILATION,
            "tta_methods": CATEGORY_TTA[category] if ENABLE_TTA else ("identity",),
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
    for method_index, method in enumerate(AUDIT_SYNTHETIC_METHODS):
        transformed = apply_synthetic_anomaly(
            images,
            method,
            seed=SEED + method_index,
            **SYNTHETIC_PARAMETERS_BY_CATEGORY[category][method],
        )
        panels.append((f"synthetic: {method}", transformed[0]))
        if ENABLE_PATCH_ADAPTER and method in PATCH_ADAPTER_METHODS:
            grid_side = IMAGE_SIZE // 14
            mask = changed_patch_mask(
                images,
                transformed,
                grid_size=(grid_side, grid_side),
                difference_threshold=PATCH_ADAPTER_DIFFERENCE_THRESHOLD,
                dilation=PATCH_ADAPTER_MASK_DILATION,
            )[0].reshape(grid_side, grid_side)
            panels.append((f"adapter mask: {method}", mask))
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    flat_axes = axes.reshape(-1)
    for axis, (title, image) in zip(flat_axes, panels, strict=False):
        if image.ndim == 2:
            axis.imshow(image, cmap="magma", vmin=0, vmax=1)
        else:
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
            "experiment_mode": EXPERIMENT_MODE,
            "profile": AUGMENTATION_PROFILE,
            "synthetic_profile": SYNTHETIC_PROFILE,
            "augmentations": ACTIVE_AUGMENTATIONS,
            "synthetic_evidence_enabled": ENABLE_SYNTHETIC_EVIDENCE,
            "synthetic_policies": SYNTHETIC_ANOMALY_POLICIES,
            "synthetic_parameters": SYNTHETIC_PARAMETERS_BY_CATEGORY,
            "synthetic_evidence_weight": SYNTHETIC_EVIDENCE_WEIGHT,
            "clean_memory_patches": CLEAN_MEMORY_PATCHES,
            "augmented_memory_patches": AUGMENTED_MEMORY_PATCHES,
            "neighborhood_degrees": NEIGHBORHOOD_DEGREES,
            "selected_threshold_scale": SELECTED_THRESHOLD_SCALE,
            "patch_adapter_enabled": ENABLE_PATCH_ADAPTER,
            "patch_adapter_methods": PATCH_ADAPTER_METHODS,
            "patch_adapter_features_per_class": PATCH_ADAPTER_FEATURES_PER_CLASS,
            "patch_adapter_features_per_method": PATCH_ADAPTER_FEATURES_PER_METHOD,
            "image_evidence_method_weights": IMAGE_EVIDENCE_METHOD_WEIGHTS,
            "patch_adapter_method_weights": PATCH_ADAPTER_METHOD_WEIGHTS,
            "patch_adapter_difference_threshold": PATCH_ADAPTER_DIFFERENCE_THRESHOLD,
            "patch_adapter_mask_dilation": PATCH_ADAPTER_MASK_DILATION,
            "patch_adapter_normal_margin_quantile": PATCH_ADAPTER_NORMAL_MARGIN_QUANTILE,
            "tta_enabled": ENABLE_TTA,
            "category_tta": CATEGORY_TTA,
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
        if AUDIT_SYNTHETIC_METHODS:
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
                for method_index, method in enumerate(AUDIT_SYNTHETIC_METHODS)
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
            if ENABLE_PATCH_ADAPTER and method in PATCH_ADAPTER_METHODS:
                grid_side = IMAGE_SIZE // 14
                masks = changed_patch_mask(
                    images,
                    transformed,
                    grid_size=(grid_side, grid_side),
                    difference_threshold=PATCH_ADAPTER_DIFFERENCE_THRESHOLD,
                    dilation=PATCH_ADAPTER_MASK_DILATION,
                )
                for index, row in selected.iterrows():
                    output = audit_root / category / "patch_adapter_mask" / method
                    output = output / f"{row.sample_id}.png"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    save_image(masks[index].reshape(1, grid_side, grid_side).float(), output)
                    records.append(
                        {
                            "config_id": config_id,
                            "category": category,
                            "sample_id": row.sample_id,
                            "transform_type": "patch_adapter_mask",
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


@torch.inference_mode()
def iter_synthetic_patch_batches(
    frame: pd.DataFrame,
    extractor: DinoV2PatchFeatureExtractor,
    category: str,
    method: str | None = None,
):
    """Yield DINO tokens only from locally edited synthetic-positive regions.

    Pixel differences create patch labels, so this iterator accepts one configured local method but
    intentionally rejects MixUp. A one-patch dilation supplies context around thin marks. Keeping
    methods separate allows explicit total method weights rather than letting large CutMix regions
    dominate merely because they contain more changed patches.

    Args:
        frame: Held-out normal rows used exclusively for adapter fitting.
        extractor: Frozen DINOv2 feature extractor.
        category: Category whose tuned synthetic parameters should be applied.
        method: One local synthetic method. ``None`` preserves the original interleaved policy.

    Yields:
        Tensors shaped ``[1, selected_patches, embedding_dimensions]``.
    """
    if method is not None and method not in PATCH_ADAPTER_METHODS:
        raise ValueError(f"Patch method must be one of {PATCH_ADAPTER_METHODS}")
    methods = PATCH_ADAPTER_METHODS if method is None else (method,)
    dataset = AnomalyImageDataset(frame, root=TRAIN_ROOT, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    for batch_index, clean_images in enumerate(loader):
        for current_method in methods:
            method_index = PATCH_ADAPTER_METHODS.index(current_method)
            transformed = apply_synthetic_anomaly(
                clean_images,
                current_method,
                seed=SEED + batch_index * len(PATCH_ADAPTER_METHODS) + method_index,
                **SYNTHETIC_PARAMETERS_BY_CATEGORY[category][current_method],
            )
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                embeddings = extractor(transformed.to(DEVICE)).float()
            patch_count = embeddings.shape[1]
            grid_side = int(round(patch_count**0.5))
            if grid_side * grid_side != patch_count:
                raise RuntimeError("Patch adapter currently requires a square DINO token grid")
            mask = changed_patch_mask(
                clean_images,
                transformed,
                grid_size=(grid_side, grid_side),
                difference_threshold=PATCH_ADAPTER_DIFFERENCE_THRESHOLD,
                dilation=PATCH_ADAPTER_MASK_DILATION,
            )
            selected = embeddings[mask.to(embeddings.device)]
            if len(selected):
                yield selected.unsqueeze(0).cpu()


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
    patch_adapter_head: dict[str, torch.Tensor | float] | None = None,
    tta_method: str = "identity",
) -> tuple[np.ndarray, torch.Tensor, dict[int, np.ndarray], np.ndarray]:
    """Return fused DINO, summary, per-degree, and optional patch-adapter scores.

    Args:
        frame: Metadata rows containing each image's relative path.
        root: Directory against which the relative paths are resolved.
        extractor: Frozen DINOv2 patch-token extractor.
        clean_memories: Protected original-normal memory for every neighborhood degree.
        augmented_memories: Known-normal transformed memory for every neighborhood degree.
        synthetic_method: Optional synthetic-positive transform applied before feature extraction.
        patch_adapter_head: Optional logistic head evaluated independently on native patch tokens.
        tta_method: Label-preserving view transform. Views are scored separately and averaged only
            at image-score level, so rotated/flipped patch grids never become spatially misaligned.

    Returns:
        The fused top-1%-tail score, six fused patch-distance summaries, a mapping containing each
        degree's standalone image scores, and the patch adapter's top-1% positive-evidence score.
        DINO degree fusion happens per patch before tail aggregation.
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
    patch_adapter_rows = []
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
        images = apply_normal_augmentation(images, tta_method)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            embeddings = extractor(images.to(DEVICE)).float()
        if patch_adapter_head is None:
            patch_adapter_rows.extend([0.0] * len(embeddings))
        else:
            patch_evidence = positive_evidence_scores(
                embeddings.reshape(-1, embeddings.shape[-1]),
                patch_adapter_head,
            ).reshape(embeddings.shape[:2])
            patch_adapter_features = summarize_patch_distances(
                patch_evidence,
                top_k=1,
                top_fraction=TOP_FRACTION,
            )
            patch_adapter_rows.extend(patch_adapter_features[:, -1].tolist())
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
        np.asarray(patch_adapter_rows, dtype=np.float64),
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


def score_frame_views(
    frame: pd.DataFrame,
    root: Path,
    extractor: DinoV2PatchFeatureExtractor,
    clean_memories: dict[int, torch.Tensor],
    augmented_memories: dict[int, torch.Tensor],
    artifact: dict[str, object],
    *,
    tta_methods: tuple[str, ...],
    synthetic_method: str | None = None,
) -> dict[str, object]:
    """Score independent TTA views and average only image-level evidence.

    This helper is used for both public calibration and test inference. Consequently, the TTA mode
    never applies a single-view threshold to averaged multi-view scores.
    """
    view_results = []
    for tta_method in tta_methods:
        dino_scores, features, degree_scores, patch_scores = score_frame(
            frame,
            root,
            extractor,
            clean_memories,
            augmented_memories,
            synthetic_method=synthetic_method,
            patch_adapter_head=artifact.get("patch_adapter_head"),
            tta_method=tta_method,
        )
        evidence, combined_scores = combine_anomaly_evidence(dino_scores, features, artifact)
        view_results.append(
            {
                "dino": dino_scores,
                "evidence": evidence,
                "combined": combined_scores,
                "patch": patch_scores,
                "degree": degree_scores,
            }
        )
    return {
        "dino": np.mean(np.stack([result["dino"] for result in view_results]), axis=0),
        "evidence": np.mean(np.stack([result["evidence"] for result in view_results]), axis=0),
        "combined": np.mean(np.stack([result["combined"] for result in view_results]), axis=0),
        "patch": np.mean(np.stack([result["patch"] for result in view_results]), axis=0),
        "degree": {
            degree: np.mean(
                np.stack([result["degree"][degree] for result in view_results]),
                axis=0,
            )
            for degree in NEIGHBORHOOD_DEGREES
        },
    }


# %% [markdown]
# ## 5. Fit normal memories, synthetic evidence, local patch adapters, and thresholds
#
# The outer validation images never enter either memory. With synthetic evidence enabled, only 25%
# of that held-out set trains the small six-feature logistic head and patch adapter; the remaining
# 75% independently calibrates the legacy threshold. Adapter negatives are sampled from the already
# protected clean and known-normal memories. Its positives contain only DINO tokens overlapping the
# local synthetic edit mask. Threshold selection remains normal-only and uses the exact inference
# views of the selected mode, so TTA scores are never compared against a single-view threshold.

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
        patch_adapter_head = None
        patch_adapter_normal_count = 0
        patch_adapter_positive_count = 0
        patch_adapter_method_counts = {}
        dino_score_std = 1.0
        positive_weight = SYNTHETIC_EVIDENCE_WEIGHT if ENABLE_SYNTHETIC_EVIDENCE else 0.0
        calibration_rows = split.valid
        if ENABLE_SYNTHETIC_EVIDENCE or ENABLE_PATCH_ADAPTER:
            evidence_split = make_split(
                split.valid,
                valid_size=1.0 - EVIDENCE_TRAIN_FRACTION,
                seed=SEED + 1,
            )
            calibration_rows = evidence_split.valid
        if ENABLE_SYNTHETIC_EVIDENCE:
            evidence_dino_scores, evidence_normal_features, evidence_degree_scores, _ = score_frame(
                evidence_split.train,
                TRAIN_ROOT,
                extractor,
                clean_memories,
                augmented_memories,
            )
            synthetic_training_parts = {
                method: score_frame(
                    evidence_split.train,
                    TRAIN_ROOT,
                    extractor,
                    clean_memories,
                    augmented_memories,
                    synthetic_method=method,
                )[1]
                for method in SYNTHETIC_ANOMALY_POLICIES[category]
            }
            synthetic_training_features = torch.cat(list(synthetic_training_parts.values()))
            image_synthetic_weights = None
            if SYNTHETIC_PROFILE != "original":
                image_synthetic_weights = torch.cat(
                    [
                        torch.full(
                            (len(features),),
                            IMAGE_EVIDENCE_METHOD_WEIGHTS[method] / len(features),
                        )
                        for method, features in synthetic_training_parts.items()
                    ]
                )
            evidence_head = fit_positive_evidence_head(
                evidence_normal_features,
                synthetic_training_features,
                seed=SEED,
                normal_margin_quantile=EVIDENCE_NORMAL_MARGIN_QUANTILE,
                synthetic_weights=image_synthetic_weights,
            )
            dino_score_std = max(float(evidence_dino_scores.std()), 1e-6)
        if ENABLE_PATCH_ADAPTER:
            patch_normal_features = sample_memory_bank(
                (
                    clean_memories[1].unsqueeze(0),
                    augmented_memories[1].unsqueeze(0),
                ),
                max_patches=PATCH_ADAPTER_FEATURES_PER_CLASS,
                seed=SEED + 2000,
            )
            patch_synthetic_weights = None
            if SYNTHETIC_PROFILE == "original":
                patch_positive_features = sample_memory_bank(
                    iter_synthetic_patch_batches(
                        evidence_split.train,
                        extractor,
                        category,
                    ),
                    max_patches=PATCH_ADAPTER_FEATURES_PER_CLASS,
                    seed=SEED + 3000,
                )
                patch_adapter_method_counts = {"interleaved": len(patch_positive_features)}
            else:
                patch_parts = {
                    method: sample_memory_bank(
                        iter_synthetic_patch_batches(
                            evidence_split.train,
                            extractor,
                            category,
                            method,
                        ),
                        max_patches=PATCH_ADAPTER_FEATURES_PER_METHOD,
                        seed=SEED + 3000 + method_index * 100,
                    )
                    for method_index, method in enumerate(PATCH_ADAPTER_METHODS)
                }
                patch_positive_features = torch.cat(list(patch_parts.values()))
                patch_synthetic_weights = torch.cat(
                    [
                        torch.full(
                            (len(features),),
                            PATCH_ADAPTER_METHOD_WEIGHTS[method] / len(features),
                        )
                        for method, features in patch_parts.items()
                    ]
                )
                patch_adapter_method_counts = {
                    method: len(features) for method, features in patch_parts.items()
                }
            patch_adapter_head = fit_positive_evidence_head(
                patch_normal_features,
                patch_positive_features,
                seed=SEED,
                normal_margin_quantile=PATCH_ADAPTER_NORMAL_MARGIN_QUANTILE,
                synthetic_weights=patch_synthetic_weights,
            )
            patch_adapter_normal_count = len(patch_normal_features)
            patch_adapter_positive_count = len(patch_positive_features)

        artifact_for_scoring = {
            "positive_evidence_head": evidence_head,
            "positive_evidence_weight": positive_weight,
            "patch_adapter_head": patch_adapter_head,
            "dino_score_std": dino_score_std,
        }
        mode_tta_methods = CATEGORY_TTA[category] if ENABLE_TTA else ("identity",)
        normal_view_scores = score_frame_views(
            calibration_rows,
            TRAIN_ROOT,
            extractor,
            clean_memories,
            augmented_memories,
            artifact_for_scoring,
            tta_methods=mode_tta_methods,
        )
        normal_dino_scores = np.asarray(normal_view_scores["dino"])
        normal_evidence = np.asarray(normal_view_scores["evidence"])
        normal_scores = np.asarray(normal_view_scores["combined"])
        normal_degree_scores = normal_view_scores["degree"]
        control_normal_scores = normal_dino_scores
        control_calibration = calibrate_anomaly_threshold(
            control_normal_scores,
            normal_quantile=NORMAL_QUANTILES[category],
        )
        dino_only_threshold = select_anomaly_threshold(control_calibration, "normal_quantile")
        native_normal_scores = normal_degree_scores[1]
        native_calibration = calibrate_anomaly_threshold(
            native_normal_scores,
            normal_quantile=NORMAL_QUANTILES[category],
        )
        native_threshold = select_anomaly_threshold(native_calibration, "normal_quantile")
        synthetic_score_parts = []
        if ENABLE_SYNTHETIC_EVIDENCE:
            for method in SYNTHETIC_ANOMALY_POLICIES[category]:
                synthetic_view_scores = score_frame_views(
                    calibration_rows,
                    TRAIN_ROOT,
                    extractor,
                    clean_memories,
                    augmented_memories,
                    artifact_for_scoring,
                    tta_methods=mode_tta_methods,
                    synthetic_method=method,
                )
                synthetic_score_parts.append(np.asarray(synthetic_view_scores["combined"]))
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
                    len(evidence_split.train)
                    if ENABLE_SYNTHETIC_EVIDENCE or ENABLE_PATCH_ADAPTER
                    else 0
                ),
                "patch_adapter_normal_features": patch_adapter_normal_count,
                "patch_adapter_positive_features": patch_adapter_positive_count,
                "patch_adapter_method_counts": patch_adapter_method_counts,
                "tta_methods": mode_tta_methods,
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
            "patch_adapter_head": patch_adapter_head,
            "patch_adapter_methods": PATCH_ADAPTER_METHODS if ENABLE_PATCH_ADAPTER else (),
            "image_evidence_method_weights": IMAGE_EVIDENCE_METHOD_WEIGHTS,
            "patch_adapter_method_weights": PATCH_ADAPTER_METHOD_WEIGHTS,
            "patch_adapter_normal_features": patch_adapter_normal_count,
            "patch_adapter_positive_features": patch_adapter_positive_count,
            "patch_adapter_method_counts": patch_adapter_method_counts,
            "dino_score_std": dino_score_std,
            "dino_only_threshold": dino_only_threshold,
            "native_degree_one_threshold": native_threshold,
            "normal_quantile": NORMAL_QUANTILES[category],
            "tta_methods": mode_tta_methods,
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
                "patch_adapter_methods": PATCH_ADAPTER_METHODS if ENABLE_PATCH_ADAPTER else (),
                "patch_adapter_feature_counts": {
                    "normal": patch_adapter_normal_count,
                    "positive": patch_adapter_positive_count,
                    "by_method": patch_adapter_method_counts,
                },
                "tta_methods": mode_tta_methods,
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
        "experiment_mode": EXPERIMENT_MODE,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "synthetic_profile": SYNTHETIC_PROFILE,
        "tta_enabled": ENABLE_TTA,
        "category_tta": CATEGORY_TTA,
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "top_fraction": TOP_FRACTION,
        "neighborhood_degrees": NEIGHBORHOOD_DEGREES,
        "selected_threshold_scale": SELECTED_THRESHOLD_SCALE,
        "synthetic_evidence_enabled": ENABLE_SYNTHETIC_EVIDENCE,
        "synthetic_evidence_weight": (
            SYNTHETIC_EVIDENCE_WEIGHT if ENABLE_SYNTHETIC_EVIDENCE else 0.0
        ),
        "patch_adapter_enabled": ENABLE_PATCH_ADAPTER,
        "patch_adapter_methods": PATCH_ADAPTER_METHODS if ENABLE_PATCH_ADAPTER else (),
        "patch_adapter_features_per_class": PATCH_ADAPTER_FEATURES_PER_CLASS,
        "patch_adapter_features_per_method": PATCH_ADAPTER_FEATURES_PER_METHOD,
        "image_evidence_method_weights": IMAGE_EVIDENCE_METHOD_WEIGHTS,
        "patch_adapter_method_weights": PATCH_ADAPTER_METHOD_WEIGHTS,
        "patch_adapter_difference_threshold": PATCH_ADAPTER_DIFFERENCE_THRESHOLD,
        "patch_adapter_mask_dilation": PATCH_ADAPTER_MASK_DILATION,
        "patch_adapter_normal_margin_quantile": PATCH_ADAPTER_NORMAL_MARGIN_QUANTILE,
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
    if bundle["experiment_mode"] != EXPERIMENT_MODE:
        raise ValueError("Loaded bundle does not match EXPERIMENT_MODE")
    if bundle["augmentation_profile"] != AUGMENTATION_PROFILE:
        raise ValueError("Loaded bundle does not match AUGMENTATION_PROFILE")
    if bundle["synthetic_profile"] != SYNTHETIC_PROFILE:
        raise ValueError("Loaded bundle does not match SYNTHETIC_PROFILE")
    if bundle["tta_enabled"] != ENABLE_TTA:
        raise ValueError("Loaded bundle does not match ENABLE_TTA")
    if bundle["synthetic_evidence_enabled"] != ENABLE_SYNTHETIC_EVIDENCE:
        raise ValueError("Loaded bundle does not match ENABLE_SYNTHETIC_EVIDENCE")
    if bundle["patch_adapter_enabled"] != ENABLE_PATCH_ADAPTER:
        raise ValueError("Loaded bundle does not match ENABLE_PATCH_ADAPTER")
    if tuple(bundle["patch_adapter_methods"]) != (
        PATCH_ADAPTER_METHODS if ENABLE_PATCH_ADAPTER else ()
    ):
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_METHODS")
    if bundle["patch_adapter_features_per_class"] != PATCH_ADAPTER_FEATURES_PER_CLASS:
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_FEATURES_PER_CLASS")
    if bundle["patch_adapter_features_per_method"] != PATCH_ADAPTER_FEATURES_PER_METHOD:
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_FEATURES_PER_METHOD")
    if bundle["image_evidence_method_weights"] != IMAGE_EVIDENCE_METHOD_WEIGHTS:
        raise ValueError("Loaded bundle does not match IMAGE_EVIDENCE_METHOD_WEIGHTS")
    if bundle["patch_adapter_method_weights"] != PATCH_ADAPTER_METHOD_WEIGHTS:
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_METHOD_WEIGHTS")
    if bundle["patch_adapter_difference_threshold"] != PATCH_ADAPTER_DIFFERENCE_THRESHOLD:
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_DIFFERENCE_THRESHOLD")
    if bundle["patch_adapter_mask_dilation"] != PATCH_ADAPTER_MASK_DILATION:
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_MASK_DILATION")
    if bundle["patch_adapter_normal_margin_quantile"] != PATCH_ADAPTER_NORMAL_MARGIN_QUANTILE:
        raise ValueError("Loaded bundle does not match PATCH_ADAPTER_NORMAL_MARGIN_QUANTILE")
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
all_patch_adapter_scores = np.zeros(len(test_table), dtype=np.float64)
all_scores = np.zeros(len(test_table), dtype=np.float64)
all_labels = np.zeros(len(test_table), dtype=np.int64)
all_degree_scores = {
    degree: np.zeros(len(test_table), dtype=np.float64) for degree in NEIGHBORHOOD_DEGREES
}
for category, category_rows in test_table.groupby("category", sort=True):
    artifact = category_artifacts[category]
    expected_tta_methods = CATEGORY_TTA[category] if ENABLE_TTA else ("identity",)
    artifact_tta_methods = tuple(artifact["tta_methods"])
    if artifact_tta_methods != expected_tta_methods:
        raise ValueError(
            f"{category} bundle TTA methods {artifact_tta_methods} do not match "
            f"selected mode {expected_tta_methods}"
        )
    view_scores = score_frame_views(
        category_rows,
        TEST_ROOT,
        extractor,
        artifact["clean_memories"],
        artifact["augmented_memories"],
        artifact,
        tta_methods=artifact_tta_methods,
    )
    row_dino_scores = np.asarray(view_scores["dino"])
    row_evidence = np.asarray(view_scores["evidence"])
    row_patch_adapter_scores = np.asarray(view_scores["patch"])
    row_scores = np.asarray(view_scores["combined"])
    row_degree_scores = view_scores["degree"]
    threshold = artifact["calibration"]["selected_threshold"] * SELECTED_THRESHOLD_SCALE
    positions = category_rows.index.to_numpy()
    all_dino_scores[positions] = row_dino_scores
    all_positive_evidence[positions] = row_evidence
    all_patch_adapter_scores[positions] = row_patch_adapter_scores
    all_scores[positions] = row_scores
    all_labels[positions] = (row_scores >= threshold).astype(np.int64)
    for degree, degree_scores in row_degree_scores.items():
        all_degree_scores[degree][positions] = degree_scores
    print(
        category,
        {
            "threshold": threshold,
            "threshold_scale": SELECTED_THRESHOLD_SCALE,
            "tta_methods": artifact_tta_methods,
            "degree_score_medians": {
                degree: float(np.median(values)) for degree, values in row_degree_scores.items()
            },
            "dino_score_median": float(np.median(row_dino_scores)),
            "positive_evidence_images": int((row_evidence > 0).sum()),
            "positive_evidence_max": float(row_evidence.max()),
            "patch_adapter_score_median": float(np.median(row_patch_adapter_scores)),
            "patch_adapter_score_max": float(row_patch_adapter_scores.max()),
            "combined_patch_spearman": float(
                pd.Series(row_scores).corr(pd.Series(row_patch_adapter_scores), method="spearman")
            ),
            "combined_score_median": float(np.median(row_scores)),
            "predicted_anomalies": int((row_scores >= threshold).sum()),
        },
    )

audit = test_table.copy()
for degree, degree_scores in all_degree_scores.items():
    audit[f"dino_degree_{degree}_score"] = degree_scores
audit["dino_score"] = all_dino_scores
audit["positive_evidence"] = all_positive_evidence
audit["patch_adapter_score"] = all_patch_adapter_scores
audit["combined_score"] = all_scores
audit["experiment_mode"] = EXPERIMENT_MODE
audit["synthetic_profile"] = SYNTHETIC_PROFILE
audit["tta_enabled"] = ENABLE_TTA
audit["label"] = all_labels

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


submission, submission_zip_path = write_submission_candidate(
    all_labels,
    tag=EXPERIMENT_MODE,
)
audit.to_csv(EXPERIMENT_DIR / f"{TASK_NAME}_{PHASE}_scores.csv", index=False)
sync_artifacts(
    paths.output_dir,
    paths.persistent_dir,
    patterns=("*.pt", "*.json", "*.csv", "*.zip"),
)
print(
    submission_zip_path,
    "mode=",
    EXPERIMENT_MODE,
    "synthetic_profile=",
    SYNTHETIC_PROFILE,
    "threshold_scale=",
    SELECTED_THRESHOLD_SCALE,
    "tta=",
    ENABLE_TTA,
    "total=",
    int(submission["label"].sum()),
    "by_category=",
    submission.groupby("category")["label"].sum().to_dict(),
)
submission.head()

# %% [markdown]
# ## 8. Mode interpretation
#
# Each run produces one ZIP. Compare the three modes as separate Colab runs; do not reuse a bundle
# across modes. The TTA mode averages identity/horizontal/vertical views only for categories 03 and
# 05, and its threshold was calibrated with those same views. All modes use threshold scale 0.85.

# %%
candidate_summary = pd.DataFrame(
    [
        {
            "experiment_mode": EXPERIMENT_MODE,
            "phase": PHASE,
            "synthetic_profile": SYNTHETIC_PROFILE,
            "threshold_scale": SELECTED_THRESHOLD_SCALE,
            "tta": ENABLE_TTA,
            "predicted_anomalies": int(all_labels.sum()),
            "submission_zip": str(submission_zip_path),
        }
    ]
).set_index("experiment_mode")
print("Completed one isolated mode; change EXPERIMENT_MODE and rerun for the next candidate.")
print(candidate_summary)
