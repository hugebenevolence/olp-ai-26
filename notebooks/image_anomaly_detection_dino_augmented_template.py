# %% [markdown]
# # Task 2 - DINOv2 with protected clean and augmented normal memories
#
# This is a separate controlled candidate derived from the 0.719 `anomalydino_448` run. DINOv2,
# cosine 1-NN, top-1% tail scoring, square 448-pixel preprocessing, and normal-only quantile
# calibration remain unchanged. The only methodological change is a second memory bank containing
# known-normal transformations. Clean patches retain their full allocation and are never replaced
# by augmented patches. Synthetic defects and the auxiliary evidence head are deliberately disabled.

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
    apply_normal_augmentation,
    calibrate_anomaly_threshold,
    load_official_training_table,
    patch_memory_features,
    sample_memory_bank,
    select_anomaly_threshold,
    stage_official_task2_data,
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
EXPERIMENT_NAME = "anomalydino_normal_augmented"

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
AUGMENTATION_PROFILE = "category_policy"  # category_policy | blur_only
PERSIST_AUGMENTATION_AUDIT = True
AUGMENTATION_AUDIT_SAMPLES = 4
SHOW_AUGMENTATION_PLOTS = True

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
config_table = pd.DataFrame(
    {
        category: {
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "clean_memory_patches": CLEAN_MEMORY_PATCHES,
            "augmented_memory_patches": AUGMENTED_MEMORY_PATCHES,
            "normal_augmentations": ACTIVE_AUGMENTATIONS[category],
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
# ## 3. Persist the exact normal augmentations before feature extraction

# %%


def preview_category_augmentations(category, *, selected_rows, save_path):
    """Render one original and every active known-normal transform for a category."""
    dataset = AnomalyImageDataset(selected_rows, root=TRAIN_ROOT, image_size=IMAGE_SIZE)
    images = torch.stack([dataset[index] for index in range(len(dataset))])
    panels = [("original", images[0])]
    panels.extend(
        (method, apply_normal_augmentation(images, method)[0])
        for method in ACTIVE_AUGMENTATIONS[category]
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
    figure.suptitle(f"{category}: DINO normal-memory transforms", fontsize=14)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def export_normal_augmentation_audit():
    """Save representative transformed PNGs, contact sheets, and a configuration manifest."""
    payload = json.dumps(
        {
            "profile": AUGMENTATION_PROFILE,
            "augmentations": ACTIVE_AUGMENTATIONS,
            "clean_memory_patches": CLEAN_MEMORY_PATCHES,
            "augmented_memory_patches": AUGMENTED_MEMORY_PATCHES,
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
        for method, transformed in transforms:
            transform_type = "clean" if method == "identity" else "augmented_normal"
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
            print(f"Normal augmentation preview: {category}")
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


def build_extractor(*, load_pretrained):
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
def iter_memory_patch_batches(frame, extractor, augmentations):
    """Yield DINOv2 patch batches for only the requested known-normal transformations."""
    dataset = AnomalyImageDataset(frame, root=TRAIN_ROOT, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    for images in loader:
        images = images.to(DEVICE)
        for augmentation in augmentations:
            transformed = apply_normal_augmentation(images, augmentation)
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                yield extractor(transformed).float().cpu()


def combined_memory_bank(clean_memory, augmented_memory):
    """Concatenate protected clean and augmented banks for exact nearest-neighbour lookup."""
    if clean_memory.ndim != 2 or augmented_memory.ndim != 2:
        raise ValueError("Both memory banks must be matrices")
    if clean_memory.shape[1] != augmented_memory.shape[1]:
        raise ValueError("Clean and augmented memory dimensions differ")
    return torch.cat((clean_memory, augmented_memory), dim=0)


@torch.inference_mode()
def score_frame(frame, root, extractor, clean_memory, augmented_memory):
    """Score rows using the nearest cosine match across both normal memory banks."""
    dataset = AnomalyImageDataset(frame, root=root, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    bank = combined_memory_bank(clean_memory, augmented_memory).to(DEVICE)
    scores = []
    feature_rows = []
    for images in loader:
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            embeddings = extractor(images.to(DEVICE)).float()
        features = patch_memory_features(
            embeddings,
            bank,
            top_k=1,
            top_fraction=TOP_FRACTION,
            distance_metric="cosine",
            distance_chunk_size=DISTANCE_CHUNK_SIZE,
        ).cpu()
        feature_rows.append(features)
        scores.extend(features[:, -1].tolist())
    return np.asarray(scores, dtype=np.float64), torch.cat(feature_rows)


# %% [markdown]
# ## 5. Fit protected clean/augmented memories and normal-only thresholds
#
# The validation images are never inserted into either memory. Synthetic anomaly calibration is
# absent, so the only experiment change relative to the 0.719 run is expanded known-normal memory.

# %%
if RUN_TRAINING:
    extractor = build_extractor(load_pretrained=True)
    model_state = {name: value.cpu() for name, value in extractor.state_dict().items()}
    category_artifacts = {}
    for category, category_rows in train_table.groupby("category", sort=True):
        split = make_split(category_rows, valid_size=NORMAL_VALID_SIZE, seed=SEED)
        clean_memory = sample_memory_bank(
            iter_memory_patch_batches(split.train, extractor, ("identity",)),
            max_patches=CLEAN_MEMORY_PATCHES,
            seed=SEED,
        )
        augmented_memory = sample_memory_bank(
            iter_memory_patch_batches(split.train, extractor, ACTIVE_AUGMENTATIONS[category]),
            max_patches=AUGMENTED_MEMORY_PATCHES,
            seed=SEED + 1000,
        )
        normal_scores, _ = score_frame(
            split.valid,
            TRAIN_ROOT,
            extractor,
            clean_memory,
            augmented_memory,
        )
        calibration = calibrate_anomaly_threshold(
            normal_scores,
            normal_quantile=NORMAL_QUANTILES[category],
        )
        calibration["selected_threshold"] = select_anomaly_threshold(calibration, "normal_quantile")
        calibration.update(
            {
                "normal_score_mean": float(normal_scores.mean()),
                "normal_score_std": float(normal_scores.std()),
                "normal_validation_images": len(normal_scores),
                "clean_memory_patches": len(clean_memory),
                "augmented_memory_patches": len(augmented_memory),
            }
        )
        category_artifacts[category] = {
            "clean_memory": clean_memory,
            "augmented_memory": augmented_memory,
            "normal_augmentations": ACTIVE_AUGMENTATIONS[category],
            "normal_quantile": NORMAL_QUANTILES[category],
            "calibration": calibration,
        }
        print(
            category,
            {
                "augmentations": ACTIVE_AUGMENTATIONS[category],
                "clean_memory": len(clean_memory),
                "augmented_memory": len(augmented_memory),
                "threshold": calibration["selected_threshold"],
                "normal_mean": calibration["normal_score_mean"],
                "normal_std": calibration["normal_score_std"],
            },
        )
    bundle = {
        "experiment_name": EXPERIMENT_NAME,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "top_fraction": TOP_FRACTION,
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
    extractor = build_extractor(load_pretrained=False)
    extractor.load_state_dict(bundle["model_state"])
    category_artifacts = bundle["categories"]
    print("Loaded frozen categories:", sorted(category_artifacts))

# %% [markdown]
# ## 6. Public/private inference and score audit

# %%
all_scores = np.zeros(len(test_table), dtype=np.float64)
all_labels = np.zeros(len(test_table), dtype=np.int64)
for category, category_rows in test_table.groupby("category", sort=True):
    artifact = category_artifacts[category]
    row_scores, _ = score_frame(
        category_rows,
        TEST_ROOT,
        extractor,
        artifact["clean_memory"],
        artifact["augmented_memory"],
    )
    threshold = artifact["calibration"]["selected_threshold"]
    positions = category_rows.index.to_numpy()
    all_scores[positions] = row_scores
    all_labels[positions] = (row_scores >= threshold).astype(np.int64)
    print(
        category,
        {
            "threshold": threshold,
            "score_min": float(row_scores.min()),
            "score_median": float(np.median(row_scores)),
            "score_max": float(row_scores.max()),
            "predicted_anomalies": int((row_scores >= threshold).sum()),
        },
    )

audit = test_table.copy()
audit["anomaly_score"] = all_scores
audit["label"] = all_labels
audit.to_csv(EXPERIMENT_DIR / f"{TASK_NAME}_{PHASE}_scores.csv", index=False)

# %% [markdown]
# ## 7. Exact submission writer

# %%


def write_submission_candidate(labels, tag="main"):
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
sync_artifacts(
    paths.output_dir,
    paths.persistent_dir,
    patterns=("*.pt", "*.json", "*.csv", "*.zip"),
)
print(zip_path, submission["label"].value_counts().to_dict())
submission.head()

# %% [markdown]
# ## 8. Public threshold candidates with exact category counts
#
# Submit the main candidate first. These files require no retraining, but they are threshold-policy
# experiments and should not be interpreted as augmentation ablations.

# %%
PUBLIC_SWEEP_SCALES = (0.90, 0.95, 1.05)
if PHASE == "public":
    for scale in PUBLIC_SWEEP_SCALES:
        candidate_labels = np.zeros(len(test_table), dtype=np.int64)
        category_counts = {}
        for category, rows in test_table.groupby("category", sort=True):
            artifact = category_artifacts[category]
            threshold = artifact["calibration"]["selected_threshold"] * scale
            labels = (all_scores[rows.index] >= threshold).astype(np.int64)
            candidate_labels[rows.index] = labels
            category_counts[category] = int(labels.sum())
        _, candidate_zip = write_submission_candidate(
            candidate_labels,
            tag=f"global_scale_{scale:.2f}",
        )
        print(
            candidate_zip,
            {"scale": scale, "total": int(candidate_labels.sum()), "by_category": category_counts},
        )
