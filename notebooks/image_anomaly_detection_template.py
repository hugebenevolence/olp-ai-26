# %% [markdown]
# # Task 2 - normal-only image anomaly detection
#
# This notebook matches the official task: six independent categories, only normal training
# images, binary image-level output, and macro per-category balanced accuracy. The baseline is a
# small PatchCore-style system: permitted frozen pretrained features, a normal patch memory bank
# per category, and thresholds calibrated with held-out normal plus allowed CutPaste anomalies.

# %% [markdown]
# ## Colab bootstrap
# Keep repeated image reads under `/content`; Drive is only for input archives and durable outputs.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import json
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
    calibrate_anomaly_threshold,
    cutpaste_batch,
    load_official_training_table,
    patch_memory_scores,
    sample_memory_bank,
    stage_official_task2_data,
    validate_anomaly_submission,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration
#
# Public development: `PHASE="public"`, `RUN_TRAINING=True` for the first run. After choosing a
# threshold scale from public aggregate feedback, rerun the freeze cell.
#
# Private final: set `PHASE="private"`, `RUN_TRAINING=False`, and point `BUNDLE_PATH` at the frozen
# public artifact. The notebook refuses to train or calibrate from private data.

# %%
TEAM_NAME = "replace_team_name"
TASK_NAME = "task2"
PHASE = "public"  # public | private
RUN_TRAINING = True

OFFICIAL_DATA_SOURCE = (
    Path("/content/drive/MyDrive/olpai26/ThiChinhThucData.zip")
    if "google.colab" in sys.modules
    else Path.home() / "Downloads" / "ThiChinhThucData.zip"
)
PERSISTENT_DIR = None  # e.g. Path("/content/drive/MyDrive/olpai26/task2_artifacts")
PRIVATE_ZIP_PASSWORD = None  # set only after the organizer releases it in the final hour
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
BUNDLE_PATH = (
    paths.output_dir / "task2_anomaly_bundle.pt"
    if RUN_TRAINING or paths.persistent_dir is None
    else paths.persistent_dir / "task2_anomaly_bundle.pt"
)

MODEL_NAME = "wide_resnet50_2"  # faster: resnet18; middle: resnet50
MODEL_CHECKPOINT = None  # optional staged local timm checkpoint
PRETRAINED_ALLOWED = True  # explicitly permitted by this task
OUT_INDICES = (2, 3)
IMAGE_SIZE = 256
BATCH_SIZE = 16
PROJECTION_DIM = 128
MAX_MEMORY_PATCHES = 4096  # per category; lower to 2048 for faster inference
ANOMALY_TOP_K = 3
NORMAL_VALID_SIZE = 0.20
NORMAL_QUANTILE = 0.99
THRESHOLD_SCALE = 1.00  # tune algorithmically from PublicScore; freeze before private
SEED = 42
NUM_WORKERS = 2

if PHASE not in {"public", "private"}:
    raise ValueError("PHASE must be 'public' or 'private'")
if PHASE == "private" and RUN_TRAINING:
    raise ValueError("Private is inference-only: set RUN_TRAINING=False and load the frozen bundle")
seed_everything(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
loader_options = dataloader_kwargs(DEVICE, NUM_WORKERS)
print(gpu_report())
print(dataset_report(paths.data_dir, image_limit=100)["counts"])

# %% [markdown]
# ## 1. Inspect official structure
# Expected train counts are 664, 664, 302, 660, 660, and 210. Stop if categories or paths differ;
# a silent path error can otherwise look like a surprisingly fast training run.

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
    raise ValueError(
        f"Train structure/counts differ from the task statement: {actual_train_counts}"
    )
expected_test_per_category = 80 if PHASE == "public" else 160
actual_test_counts = test_table["category"].value_counts().sort_index().to_dict()
expected_test_counts = {category: expected_test_per_category for category in expected_train_counts}
if actual_test_counts != expected_test_counts:
    raise ValueError(
        f"{PHASE} test structure/counts differ from the task statement: {actual_test_counts}"
    )
print("Train counts:\n", train_table["category"].value_counts().sort_index())
print("Test counts:\n", test_table["category"].value_counts().sort_index())
print(test_table.head())

# The extracted public README asks for a real-valued anomaly score, but the official problem PDF
# requires binary values in the `label` column. This notebook follows the PDF contract. Confirm any
# later organizer clarification before changing the final submission serializer.

# %% [markdown]
# ## 2. Feature extractor and reusable scoring helpers
# The encoder is frozen. "Training" fits category-specific normal memory banks and thresholds.

# %%
extractor = (
    TimmPatchFeatureExtractor(
        MODEL_NAME,
        out_indices=OUT_INDICES,
        projection_dim=PROJECTION_DIM,
        pretrained_allowed=PRETRAINED_ALLOWED and RUN_TRAINING,
        checkpoint_path=MODEL_CHECKPOINT if RUN_TRAINING else None,
        seed=SEED,
    )
    .to(DEVICE)
    .eval()
)


@torch.inference_mode()
def collect_patch_batches(frame, root):
    """Extract CPU patch batches for fitting one category memory bank."""
    dataset = AnomalyImageDataset(frame, root=root, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    outputs = []
    for images in loader:
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            outputs.append(extractor(images.to(DEVICE)).float().cpu())
    return outputs


@torch.inference_mode()
def score_frame(frame, root, memory_bank, synthetic=False):
    """Score table rows in order, optionally after deterministic CutPaste corruption."""
    dataset = AnomalyImageDataset(frame, root=root, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_options)
    memory_bank = memory_bank.to(DEVICE)
    scores = []
    for batch_index, images in enumerate(loader):
        if synthetic:
            images = cutpaste_batch(images, seed=SEED + batch_index)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            embeddings = extractor(images.to(DEVICE)).float()
        batch_scores = patch_memory_scores(embeddings, memory_bank, top_k=ANOMALY_TOP_K)
        scores.extend(batch_scores.cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


# %% [markdown]
# ## 3. Fit six normal memory banks and calibrate six thresholds
# The reported balanced accuracy is a **synthetic proxy**, not the hidden official score.

# %%
memory_banks = {}
calibration = {}
if RUN_TRAINING:
    for category, category_rows in train_table.groupby("category", sort=True):
        split = make_split(category_rows, valid_size=NORMAL_VALID_SIZE, seed=SEED)
        patch_batches = collect_patch_batches(split.train, TRAIN_ROOT)
        bank = sample_memory_bank(patch_batches, max_patches=MAX_MEMORY_PATCHES, seed=SEED)
        normal_scores = score_frame(split.valid, TRAIN_ROOT, bank)
        synthetic_scores = score_frame(split.valid, TRAIN_ROOT, bank, synthetic=True)
        calibrated = calibrate_anomaly_threshold(
            normal_scores,
            synthetic_scores,
            normal_quantile=NORMAL_QUANTILE,
        )
        memory_banks[category] = bank
        calibration[category] = {
            **calibrated,
            "normal_score_mean": float(normal_scores.mean()),
            "normal_score_std": float(normal_scores.std()),
            "normal_validation_images": len(normal_scores),
        }
        print(category, calibration[category])

# %% [markdown]
# ## 4. Freeze the complete artifact
# This stores the exact encoder weights, random projection, memory banks, thresholds, and selected
# public threshold scale. Rerun this cell after changing `THRESHOLD_SCALE`, before private opens.

# %%
if RUN_TRAINING:
    bundle = {
        "config": {
            "model_name": MODEL_NAME,
            "out_indices": OUT_INDICES,
            "image_size": IMAGE_SIZE,
            "projection_dim": PROJECTION_DIM,
            "anomaly_top_k": ANOMALY_TOP_K,
            "threshold_scale": THRESHOLD_SCALE,
            "seed": SEED,
        },
        "extractor_state": {key: value.cpu() for key, value in extractor.state_dict().items()},
        "memory_banks": memory_banks,
        "calibration": calibration,
    }
    torch.save(bundle, BUNDLE_PATH)
    (paths.output_dir / "task2_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    sync_artifacts(paths.output_dir, paths.persistent_dir)
    print("Frozen bundle:", BUNDLE_PATH)
else:
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=True)
    frozen = bundle["config"]
    # Rebuild the architecture without downloading weights, then restore the exact frozen state.
    extractor = (
        TimmPatchFeatureExtractor(
            frozen["model_name"],
            out_indices=tuple(frozen["out_indices"]),
            projection_dim=frozen["projection_dim"],
            pretrained_allowed=False,
            seed=frozen["seed"],
        )
        .to(DEVICE)
        .eval()
    )
    extractor.load_state_dict(bundle["extractor_state"])
    memory_banks = bundle["memory_banks"]
    calibration = bundle["calibration"]
    IMAGE_SIZE = frozen["image_size"]
    ANOMALY_TOP_K = frozen["anomaly_top_k"]
    THRESHOLD_SCALE = frozen["threshold_scale"]
    print("Loaded frozen threshold scale:", THRESHOLD_SCALE)

# %% [markdown]
# ## 5. Public/private inference
# Inference fits nothing. Predictions are written back to the original `test.csv` row positions.

# %%
all_scores = np.zeros(len(test_table), dtype=np.float64)
all_labels = np.zeros(len(test_table), dtype=np.int64)
for category, category_rows in test_table.groupby("category", sort=True):
    if category not in memory_banks:
        raise KeyError(f"No frozen memory bank for test category {category}")
    row_scores = score_frame(category_rows, TEST_ROOT, memory_banks[category])
    threshold = calibration[category]["threshold"] * THRESHOLD_SCALE
    positions = category_rows.index.to_numpy()
    all_scores[positions] = row_scores
    all_labels[positions] = (row_scores >= threshold).astype(np.int64)
    print(
        category,
        {"threshold": threshold, "predicted_anomalies": int((row_scores >= threshold).sum())},
    )

audit = test_table.copy()
audit["anomaly_score"] = all_scores
audit["label"] = all_labels
audit.to_csv(paths.output_dir / f"{TASK_NAME}_{PHASE}_scores.csv", index=False)

# %% [markdown]
# ## 6. Exact CSV and ZIP contract
# The ZIP contains exactly one CSV. The audit score file stays outside the submission archive.

# %%
submission = test_table[["sample_id", "category"]].copy()
submission["label"] = all_labels
validate_anomaly_submission(submission, test_table)
csv_name = f"{TASK_NAME}_{PHASE}_output.csv"
csv_path = paths.output_dir / csv_name
submission.to_csv(csv_path, index=False, encoding="utf-8")
zip_suffix = "pub" if PHASE == "public" else "pri"
zip_path = paths.output_dir / f"{TEAM_NAME}_{TASK_NAME}_{zip_suffix}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(csv_path, arcname=csv_name)
with zipfile.ZipFile(zip_path) as archive:
    if archive.namelist() != [csv_name]:
        raise RuntimeError("Submission ZIP must contain exactly the required CSV")
sync_artifacts(
    paths.output_dir, paths.persistent_dir, patterns=("*.pt", "*.json", "*.csv", "*.zip")
)
print(csv_path, zip_path, submission["label"].value_counts().to_dict())
submission.head()

# %% [markdown]
# ## 7. Public threshold-scale sweep (optional; never run on private)
# PublicScore may be used to tune algorithmic thresholds, but labels must never be assigned by
# manual per-image inspection. Generate candidates sparingly within the 20-submission limit, select
# one scale from aggregate feedback, then rerun the freeze cell so private uses that exact scale.

# %%
PUBLIC_SWEEP_SCALES = (0.90, 1.00, 1.10)
if PHASE == "public":
    for scale in PUBLIC_SWEEP_SCALES:
        candidate = test_table[["sample_id", "category"]].copy()
        candidate_labels = np.zeros(len(test_table), dtype=np.int64)
        for category, rows in test_table.groupby("category", sort=True):
            threshold = calibration[category]["threshold"] * scale
            candidate_labels[rows.index] = (all_scores[rows.index] >= threshold).astype(np.int64)
        candidate["label"] = candidate_labels
        validate_anomaly_submission(candidate, test_table)
        candidate.to_csv(
            paths.output_dir / f"{TASK_NAME}_public_scale_{scale:.2f}.csv", index=False
        )
    print("Candidate CSVs created; submit only deliberate experiments.")
