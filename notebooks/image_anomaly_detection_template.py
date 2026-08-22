# %% [markdown]
# # Task 2 - category-specialized normal-only anomaly detection
#
# The supplied executed runs scoring 0.577 and 0.590 both used one `wide_resnet50_2`, identity-only
# normal memory, and CutPaste calibration for every category. This revision keeps that design as a
# preset and adds controlled category specialization plus persisted augmentation evidence. Do not
# manually label public images; use only aggregate PublicScore feedback.

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
    TimmPatchFeatureExtractor,
    apply_normal_augmentation,
    apply_synthetic_anomaly,
    calibrate_anomaly_threshold,
    load_official_training_table,
    patch_memory_scores,
    sample_memory_bank,
    select_anomaly_threshold,
    stage_official_task2_data,
    validate_anomaly_submission,
)

# %% [markdown]
# ## 0. Run and data configuration
#
# Recommended public experiment order:
#
# 1. Keep the submitted `baseline_0577` result as reference; do not spend another submission on it.
# 2. Run `category_models_only` to isolate the model-choice effect.
# 3. Run `category_augmented` to measure the added limited-augmentation/calibration effect.
# 4. Use global threshold candidates only on the better approach.

# %%
TEAM_NAME = "replace_team_name"
TASK_NAME = "task2"
PHASE = "public"  # public | private
RUN_TRAINING = True
EXPERIMENT_PRESET = "category_augmented"
# choices: baseline_0577 | category_models_only | category_augmented

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
PERSIST_AUGMENTATION_AUDIT = True
AUGMENTATION_AUDIT_MODE = "sample"  # sample | all
AUGMENTATION_AUDIT_SAMPLES = 4  # per category when mode="sample"
SHOW_ALL_AUGMENTATION_PLOTS = True
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
EXPERIMENT_DIR = paths.output_dir / EXPERIMENT_PRESET
BUNDLE_NAME = f"task2_{EXPERIMENT_PRESET}_bundle.pt"
BUNDLE_PATH = (
    EXPERIMENT_DIR / BUNDLE_NAME
    if RUN_TRAINING or paths.persistent_dir is None
    else paths.persistent_dir / EXPERIMENT_PRESET / BUNDLE_NAME
)

PRETRAINED_ALLOWED = True  # explicitly permitted by the task
MODEL_CHECKPOINTS = {
    "wide_resnet50_2": None,
    "convnext_tiny": None,
    "resnet50": None,
}
SEED = 42
NUM_WORKERS = 2

if PHASE not in {"public", "private"}:
    raise ValueError("PHASE must be 'public' or 'private'")
if PHASE == "private" and RUN_TRAINING:
    raise ValueError("Private is inference-only: load a frozen public bundle")
seed_everything(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
loader_options = dataloader_kwargs(DEVICE, NUM_WORKERS)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
print(gpu_report())
print(dataset_report(paths.data_dir, image_limit=100)["counts"])

# %% [markdown]
# ## 1. Category model and augmentation presets
#
# Augmentations expand only the **normal memory bank**. They are deliberately deterministic and
# mild. Validation/test images remain unaugmented. `normal_quantile` and `threshold_mode` are
# calibration choices; changes to them are separate from backbone changes.

# %%

DEFAULT_SYNTHETIC_PARAMETERS = {
    "cutpaste": {"cutpaste_area_range": (0.03, 0.15)},
    "mixup": {"mixup_alpha_range": (0.25, 0.45)},
    "blur": {"blur_sigma": 2.5},
    "dark_curve": {"curve_width_fraction": 0.025, "curve_darkness": 0.85},
}


def category_config(
    model_name,
    *,
    image_size=256,
    batch_size=16,
    memory_patches=4096,
    top_k=3,
    augmentations=("identity",),
    normal_quantile=0.99,
    threshold_mode="synthetic",
    threshold_scale=1.0,
    synthetic_anomalies=("cutpaste",),
    synthetic_parameters=None,
):
    """Create one explicit, serializable category experiment configuration."""
    parameters = synthetic_parameters or DEFAULT_SYNTHETIC_PARAMETERS
    return {
        "model_name": model_name,
        "out_indices": (2, 3),
        "image_size": image_size,
        "batch_size": batch_size,
        "projection_dim": 128,
        "max_memory_patches": memory_patches,
        "top_k": top_k,
        "normal_valid_size": 0.20,
        "normal_quantile": normal_quantile,
        "threshold_mode": threshold_mode,
        "threshold_scale": threshold_scale,
        "normal_augmentations": tuple(augmentations),
        "synthetic_anomalies": tuple(synthetic_anomalies),
        "synthetic_parameters": {
            method: dict(method_parameters) for method, method_parameters in parameters.items()
        },
    }


CATEGORIES = tuple(f"category_{index:02d}" for index in range(1, 7))
BASELINE_CONFIGS = {category: category_config("wide_resnet50_2") for category in CATEGORIES}

# Model-only preset: augmentation/calibration stays identical to the 0.577 run.
MODEL_CHOICES = {
    "category_01": ("wide_resnet50_2", 288, 12, 6144, 5),
    "category_02": ("convnext_tiny", 320, 12, 8192, 5),
    "category_03": ("resnet50", 288, 16, 6144, 5),
    "category_04": ("wide_resnet50_2", 320, 10, 8192, 5),
    "category_05": ("resnet50", 288, 16, 6144, 5),
    "category_06": ("convnext_tiny", 288, 16, 6144, 5),
}
MODELS_ONLY_CONFIGS = {
    category: category_config(
        model_name,
        image_size=image_size,
        batch_size=batch_size,
        memory_patches=memory_patches,
        top_k=top_k,
    )
    for category, (
        model_name,
        image_size,
        batch_size,
        memory_patches,
        top_k,
    ) in MODEL_CHOICES.items()
}

# Specialized preset derived from aggregate train/public acquisition statistics and category-level
# visual structure. These are hypotheses to test, not claimed anomaly labels.
CATEGORY_AUGMENTATIONS = {
    "category_01": ("identity", "rot180", "contrast_down", "contrast_up"),
    "category_02": ("identity", "brightness_down", "brightness_up", "contrast_up"),
    "category_03": ("identity", "hflip", "vflip", "rot90", "rot270"),
    "category_04": ("identity", "brightness_down", "brightness_up", "contrast_up"),
    "category_05": ("identity", "hflip", "vflip", "rot90", "rot270"),
    "category_06": (
        "identity",
        "brightness_down",
        "brightness_up",
        "contrast_up",
        "cool",
        "warm",
    ),
}
SYNTHETIC_ANOMALY_POLICIES = {
    "category_01": ("cutpaste", "mixup", "blur", "dark_curve"),
    "category_02": ("cutpaste", "mixup", "blur", "dark_curve"),
    "category_03": ("cutpaste", "mixup", "blur", "dark_curve"),
    "category_04": ("cutpaste", "mixup", "blur", "dark_curve"),
    "category_05": ("cutpaste", "mixup", "blur", "dark_curve"),
    "category_06": ("cutpaste", "mixup", "blur", "dark_curve"),
}
SYNTHETIC_PARAMETERS_BY_CATEGORY = {
    category: {
        method: dict(parameters) for method, parameters in DEFAULT_SYNTHETIC_PARAMETERS.items()
    }
    for category in CATEGORIES
}
# Example per-category severity edit:
# SYNTHETIC_PARAMETERS_BY_CATEGORY["category_06"]["mixup"]["mixup_alpha_range"] = (0.15, 0.30)
NORMAL_QUANTILES = {
    "category_01": 0.99,
    "category_02": 0.975,
    "category_03": 0.975,
    "category_04": 0.975,
    "category_05": 0.975,
    "category_06": 0.95,
}
CATEGORY_AUGMENTED_CONFIGS = {
    category: {
        **MODELS_ONLY_CONFIGS[category],
        "normal_augmentations": CATEGORY_AUGMENTATIONS[category],
        "synthetic_anomalies": SYNTHETIC_ANOMALY_POLICIES[category],
        "synthetic_parameters": SYNTHETIC_PARAMETERS_BY_CATEGORY[category],
        "normal_quantile": NORMAL_QUANTILES[category],
        "threshold_mode": "min_synthetic_quantile",
    }
    for category in CATEGORIES
}
PRESETS = {
    "baseline_0577": BASELINE_CONFIGS,
    "category_models_only": MODELS_ONLY_CONFIGS,
    "category_augmented": CATEGORY_AUGMENTED_CONFIGS,
}
try:
    CATEGORY_CONFIGS = PRESETS[EXPERIMENT_PRESET]
except KeyError as error:
    raise ValueError(f"Unknown preset; choose {sorted(PRESETS)}") from error
print(pd.DataFrame(CATEGORY_CONFIGS).T)

# %% [markdown]
# ## 2. Validate official tables

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

# The archive README says real-valued score while the official PDF says binary label. This notebook
# keeps scores for audit but follows the PDF's binary submission contract pending clarification.

# %% [markdown]
# ## 2.1 Preview one category's configured transformations
#
# Normal-memory transforms and synthetic anomalies are shown in separate rows conceptually and
# are never mixed in training. Change `PREVIEW_CATEGORY` and rerun this cell before a full run.

# %%
PREVIEW_CATEGORY = "category_06"
SHOW_TRANSFORM_PREVIEW = False  # all categories are rendered by the persisted audit below


def preview_category_transforms(category, *, rows=None, save_path=None):
    """Plot and optionally persist the exact transforms configured for one category."""
    config = CATEGORY_CONFIGS[category]
    preview_rows = (
        train_table.loc[train_table["category"] == category].head(2)
        if rows is None
        else rows.head(2)
    )
    preview_batch = torch.stack(
        [
            AnomalyImageDataset(preview_rows, root=TRAIN_ROOT, image_size=config["image_size"])[i]
            for i in range(len(preview_rows))
        ]
    )
    panels = [("original", preview_batch[0])]
    panels.extend(
        (f"normal: {name}", apply_normal_augmentation(preview_batch, name)[0])
        for name in config["normal_augmentations"]
        if name != "identity"
    )
    panels.extend(
        (
            f"synthetic: {name}\n{config['synthetic_parameters'].get(name, {})}",
            apply_synthetic_anomaly(
                preview_batch,
                name,
                seed=SEED,
                **config["synthetic_parameters"].get(name, {}),
            )[0],
        )
        for name in config["synthetic_anomalies"]
    )
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    for axis, (title, image) in zip(np.asarray(axes).reshape(-1), panels, strict=False):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(title)
        axis.axis("off")
    for axis in np.asarray(axes).reshape(-1)[len(panels) :]:
        axis.axis("off")
    figure.suptitle(f"{category}: verify before training", fontsize=14)
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


if SHOW_TRANSFORM_PREVIEW:
    preview_category_transforms(PREVIEW_CATEGORY)

# %% [markdown]
# ## 2.2 Persist augmentation evidence
#
# This block saves individual PNGs, one contact sheet per category, the exact method parameters,
# and a manifest. A configuration hash creates a new folder whenever severity or methods change,
# so two experiments can be compared without overwriting each other. With the default Drive-backed
# `PERSISTENT_DIR`, the audit survives a Colab runtime reset.

# %%


def export_augmentation_audit():
    """Persist representative originals/transforms and return their audit manifest."""
    if AUGMENTATION_AUDIT_MODE not in {"sample", "all"}:
        raise ValueError("AUGMENTATION_AUDIT_MODE must be 'sample' or 'all'")
    if AUGMENTATION_AUDIT_SAMPLES < 1:
        raise ValueError("AUGMENTATION_AUDIT_SAMPLES must be positive")
    payload = json.dumps(CATEGORY_CONFIGS, sort_keys=True)
    config_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    audit_root = EXPERIMENT_DIR / "augmentation_audit" / config_id
    records = []
    for category in CATEGORIES:
        config = CATEGORY_CONFIGS[category]
        category_rows = train_table.loc[train_table["category"] == category]
        count = (
            len(category_rows)
            if AUGMENTATION_AUDIT_MODE == "all"
            else min(AUGMENTATION_AUDIT_SAMPLES, len(category_rows))
        )
        positions = np.linspace(0, len(category_rows) - 1, num=count, dtype=int)
        selected = category_rows.iloc[positions].reset_index(drop=True)
        dataset = AnomalyImageDataset(selected, root=TRAIN_ROOT, image_size=config["image_size"])
        batch_size = min(config["batch_size"], len(dataset))
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch_rows = selected.iloc[start:stop].reset_index(drop=True)
            images = torch.stack([dataset[index] for index in range(start, stop)])
            transforms = [("original", "identity", images, {})]
            transforms.extend(
                ("normal", method, apply_normal_augmentation(images, method), {})
                for method in config["normal_augmentations"]
                if method != "identity"
            )
            transforms.extend(
                (
                    "synthetic_anomaly",
                    method,
                    apply_synthetic_anomaly(
                        images,
                        method,
                        seed=SEED + start,
                        **config["synthetic_parameters"].get(method, {}),
                    ),
                    config["synthetic_parameters"].get(method, {}),
                )
                for method in config["synthetic_anomalies"]
            )
            for transform_type, method, transformed, parameters in transforms:
                for index, row in batch_rows.iterrows():
                    output = (
                        audit_root / category / transform_type / method / f"{row.sample_id}.png"
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    save_image(transformed[index], output)
                    records.append(
                        {
                            "config_id": config_id,
                            "category": category,
                            "sample_id": row.sample_id,
                            "source_relative_path": row.relative_path,
                            "transform_type": transform_type,
                            "method": method,
                            "parameters": json.dumps(parameters, sort_keys=True),
                            "output_relative_path": output.relative_to(EXPERIMENT_DIR).as_posix(),
                        }
                    )
        figure = preview_category_transforms(
            category,
            rows=selected,
            save_path=audit_root / category / "contact_sheet.png",
        )
        if SHOW_ALL_AUGMENTATION_PLOTS:
            print(f"Augmentation preview: {category}")
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
    print("Local augmentation audit:", audit_root)
    if paths.persistent_dir is not None:
        persistent_root = paths.persistent_dir / audit_root.relative_to(paths.output_dir)
        print("Persistent augmentation audit:", persistent_root)
        print("Files copied to persistent storage:", len(copied))
    else:
        print("WARNING: PERSISTENT_DIR=None; PNGs disappear when the runtime resets")
    return manifest


augmentation_manifest = None
if RUN_TRAINING and PERSIST_AUGMENTATION_AUDIT:
    augmentation_manifest = export_augmentation_audit()
    print(augmentation_manifest.head())

# %% [markdown]
# ## 3. Reusable category training and scoring helpers

# %%


def extractor_key(config):
    """Create a stable key so identical backbone/projection states are stored only once."""
    payload = {key: config[key] for key in ("model_name", "out_indices", "projection_dim")}
    payload["seed"] = SEED
    return json.dumps(payload, sort_keys=True)


def build_extractor(config, *, load_pretrained):
    """Build one configured frozen feature extractor on the active device."""
    checkpoint = MODEL_CHECKPOINTS.get(config["model_name"])
    return (
        TimmPatchFeatureExtractor(
            config["model_name"],
            out_indices=tuple(config["out_indices"]),
            projection_dim=config["projection_dim"],
            pretrained_allowed=PRETRAINED_ALLOWED and load_pretrained,
            checkpoint_path=checkpoint if load_pretrained else None,
            seed=SEED,
        )
        .to(DEVICE)
        .eval()
    )


@torch.inference_mode()
def iter_normal_patch_batches(frame, root, extractor, config):
    """Yield augmented normal patch embeddings without accumulating all candidates in RAM."""
    dataset = AnomalyImageDataset(frame, root=root, image_size=config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, **loader_options)
    for images in loader:
        images = images.to(DEVICE)
        for augmentation in config["normal_augmentations"]:
            transformed = apply_normal_augmentation(images, augmentation)
            with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                yield extractor(transformed).float().cpu()


@torch.inference_mode()
def score_frame(frame, root, extractor, memory_bank, config, *, synthetic_method=None):
    """Score table rows in order using deterministic validation/test preprocessing."""
    dataset = AnomalyImageDataset(frame, root=root, image_size=config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, **loader_options)
    bank = memory_bank.to(DEVICE)
    scores = []
    for batch_index, images in enumerate(loader):
        if synthetic_method is not None:
            images = apply_synthetic_anomaly(
                images,
                synthetic_method,
                seed=SEED + batch_index,
                **config["synthetic_parameters"].get(synthetic_method, {}),
            )
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            embeddings = extractor(images.to(DEVICE)).float()
        values = patch_memory_scores(embeddings, bank, top_k=config["top_k"])
        scores.extend(values.cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


# %% [markdown]
# ## 4. Fit category memories and calibrate thresholds
# `proxy_balanced_accuracy` uses synthetic CutPaste positives and is not the official metric.

# %%
model_states = {}
category_artifacts = {}
extractor_cache = {}
if RUN_TRAINING:
    for category, category_rows in train_table.groupby("category", sort=True):
        config = CATEGORY_CONFIGS[category]
        key = extractor_key(config)
        if key not in extractor_cache:
            extractor_cache[key] = build_extractor(config, load_pretrained=True)
            model_states[key] = {
                name: value.cpu() for name, value in extractor_cache[key].state_dict().items()
            }
        extractor = extractor_cache[key]
        split = make_split(category_rows, valid_size=config["normal_valid_size"], seed=SEED)
        bank = sample_memory_bank(
            iter_normal_patch_batches(split.train, TRAIN_ROOT, extractor, config),
            max_patches=config["max_memory_patches"],
            seed=SEED,
        )
        normal_scores = score_frame(split.valid, TRAIN_ROOT, extractor, bank, config)
        synthetic_scores = np.concatenate(
            [
                score_frame(
                    split.valid,
                    TRAIN_ROOT,
                    extractor,
                    bank,
                    config,
                    synthetic_method=method,
                )
                for method in config["synthetic_anomalies"]
            ]
        )
        calibrated = calibrate_anomaly_threshold(
            normal_scores,
            synthetic_scores,
            normal_quantile=config["normal_quantile"],
        )
        calibrated["selected_threshold"] = select_anomaly_threshold(
            calibrated, config["threshold_mode"]
        )
        calibrated.update(
            {
                "normal_score_mean": float(normal_scores.mean()),
                "normal_score_std": float(normal_scores.std()),
                "normal_validation_images": len(normal_scores),
            }
        )
        category_artifacts[category] = {
            "config": config,
            "model_key": key,
            "memory_bank": bank,
            "calibration": calibrated,
        }
        print(category, config["model_name"], config["normal_augmentations"], calibrated)

# %% [markdown]
# ## 5. Freeze or load the complete experiment bundle

# %%
if RUN_TRAINING:
    bundle = {
        "experiment_name": EXPERIMENT_PRESET,
        "seed": SEED,
        "model_states": model_states,
        "categories": category_artifacts,
    }
    torch.save(bundle, BUNDLE_PATH)
    calibration_report = {
        category: artifact["calibration"] for category, artifact in category_artifacts.items()
    }
    (EXPERIMENT_DIR / "task2_calibration.json").write_text(
        json.dumps(calibration_report, indent=2), encoding="utf-8"
    )
    (EXPERIMENT_DIR / "task2_experiment_config.json").write_text(
        json.dumps(
            {"experiment_name": EXPERIMENT_PRESET, "categories": CATEGORY_CONFIGS},
            indent=2,
        ),
        encoding="utf-8",
    )
    sync_artifacts(paths.output_dir, paths.persistent_dir)
    print("Frozen bundle:", BUNDLE_PATH)
else:
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=True)
    if bundle["experiment_name"] != EXPERIMENT_PRESET:
        raise ValueError("Loaded bundle experiment does not match EXPERIMENT_PRESET")
    model_states = bundle["model_states"]
    category_artifacts = bundle["categories"]
    print("Loaded frozen categories:", sorted(category_artifacts))

# %% [markdown]
# ## 6. Public/private inference

# %%
all_scores = np.zeros(len(test_table), dtype=np.float64)
all_labels = np.zeros(len(test_table), dtype=np.int64)
inference_extractors = {}
for category, category_rows in test_table.groupby("category", sort=True):
    artifact = category_artifacts[category]
    config = artifact["config"]
    key = artifact["model_key"]
    if key not in inference_extractors:
        inference_extractors[key] = build_extractor(config, load_pretrained=False)
        inference_extractors[key].load_state_dict(model_states[key])
    row_scores = score_frame(
        category_rows,
        TEST_ROOT,
        inference_extractors[key],
        artifact["memory_bank"],
        config,
    )
    threshold = artifact["calibration"]["selected_threshold"] * config["threshold_scale"]
    positions = category_rows.index.to_numpy()
    all_scores[positions] = row_scores
    all_labels[positions] = (row_scores >= threshold).astype(np.int64)
    print(
        category,
        {
            "model": config["model_name"],
            "threshold": threshold,
            "predicted_anomalies": int((row_scores >= threshold).sum()),
        },
    )

audit = test_table.copy()
audit["anomaly_score"] = all_scores
audit["label"] = all_labels
audit.to_csv(EXPERIMENT_DIR / f"{TASK_NAME}_{PHASE}_scores.csv", index=False)

# %% [markdown]
# ## 7. Exact CSV and ZIP writer

# %%


def write_submission_candidate(labels, tag="main"):
    """Write one isolated candidate directory with the exact required CSV/ZIP names."""
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
    paths.output_dir, paths.persistent_dir, patterns=("*.pt", "*.json", "*.csv", "*.zip")
)
print(zip_path, submission["label"].value_counts().to_dict())
submission.head()

# %% [markdown]
# ## 8. Public global threshold candidates
# Lower scales predict more anomalies. Submit only controlled experiments within the 20-run limit.

# %%
PUBLIC_SWEEP_SCALES = (0.90, 0.95, 1.05)
if PHASE == "public":
    for scale in PUBLIC_SWEEP_SCALES:
        candidate_labels = np.zeros(len(test_table), dtype=np.int64)
        for category, rows in test_table.groupby("category", sort=True):
            artifact = category_artifacts[category]
            threshold = artifact["calibration"]["selected_threshold"] * scale
            candidate_labels[rows.index] = (all_scores[rows.index] >= threshold).astype(np.int64)
        _, candidate_zip = write_submission_candidate(
            candidate_labels, tag=f"global_scale_{scale:.2f}"
        )
        print(candidate_zip)
