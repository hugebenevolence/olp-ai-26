# %% [markdown]
# # Adversarial robustness baseline (FGSM and PGD)
# Use only when the task explicitly asks for attacks/defenses or robustness evaluation.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ("timm",)
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.core.metrics import evaluate_metric
from olp_ai_26.core.report import record_run
from olp_ai_26.cv.adversarial import fgsm, perturbation_statistics, pgd
from olp_ai_26.cv.classification import (
    ImageTableDataset,
    build_image_classifier,
    build_image_transforms,
)

# %% [markdown]
# ## 0. Drag-and-plug configuration
# Inputs to these helpers must be in the clipping range below. If the classifier consumes
# normalized tensors, attack before normalization or convert bounds/epsilon into normalized space.

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/adversarial")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "adversarial_robustness"
paths.output_dir.mkdir(parents=True, exist_ok=True)
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, paths.data_dir)
SOURCE_RUN = paths.runtime_dir / "outputs" / "cv_classification"
IMAGE_COLUMN, TARGET_COLUMN = "image", "label"
BATCH_SIZE = 8
ATTACK = "pgd"  # fgsm | pgd
EPSILON = 8 / 255
PGD_STEP_SIZE, PGD_STEPS = 2 / 255, 10
CLIP_MIN, CLIP_MAX = 0.0, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(gpu_report())

# %% [markdown]
# ## 1. Load the trained clean classifier

# %%
if not (SOURCE_RUN / "run.json").is_file() or not (SOURCE_RUN / "best.pt").is_file():
    raise FileNotFoundError(
        "Run the image-classification baseline first and point SOURCE_RUN at its bundle"
    )
recipe = json.loads((SOURCE_RUN / "run.json").read_text(encoding="utf-8"))
label_values = recipe["inference_options"]["label_values"]
label_to_index = {value: index for index, value in enumerate(label_values)}
IMAGE_SIZE = recipe["inference_options"]["image_size"]
base_model = build_image_classifier(recipe["model"], len(label_values), pretrained_allowed=False)
payload = torch.load(SOURCE_RUN / "best.pt", map_location="cpu", weights_only=True)
base_model.load_state_dict(payload["model"])


class PixelSpaceClassifier(nn.Module):
    """Keep attacks in [0,1] pixel space and normalize inside the trained classifier."""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images):
        return self.classifier((images - self.mean) / self.std)


model = PixelSpaceClassifier(base_model).to(DEVICE).eval()

# %% [markdown]
# ## 2. Real held-out data and bounded attacks
# The baseline's recorded validation fold is reused. Do not substitute its training rows.

# %%
train = pd.read_csv(paths.data_dir / "train.csv")
valid = train.iloc[recipe["split"]["valid_indices"]].reset_index(drop=True)
dataset = ImageTableDataset(
    valid,
    image_column=IMAGE_COLUMN,
    target_column=TARGET_COLUMN,
    root=paths.data_dir,
    transform=build_image_transforms(size=IMAGE_SIZE, training=False, normalize=False),
    label_to_index=label_to_index,
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
truth, clean_predictions, robust_predictions = [], [], []
for clean_images, labels in loader:
    clean_images, labels = clean_images.to(DEVICE), labels.to(DEVICE)
    if ATTACK == "fgsm":
        attacked = fgsm(
            model, clean_images, labels, epsilon=EPSILON, clip_min=CLIP_MIN, clip_max=CLIP_MAX
        )
    elif ATTACK == "pgd":
        attacked = pgd(
            model,
            clean_images,
            labels,
            epsilon=EPSILON,
            step_size=PGD_STEP_SIZE,
            steps=PGD_STEPS,
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX,
        )
    else:
        raise ValueError("ATTACK must be fgsm or pgd")
    statistics = perturbation_statistics(clean_images, attacked)
    if statistics["linf"] > EPSILON + 1e-6:
        raise ValueError("Attack violates its declared pixel-space perturbation budget")
    with torch.inference_mode():
        clean_predictions.extend(model(clean_images).argmax(1).cpu().tolist())
        robust_predictions.extend(model(attacked).argmax(1).cpu().tolist())
    truth.extend(labels.cpu().tolist())
metrics = {
    "clean_accuracy": evaluate_metric("accuracy", truth, clean_predictions),
    "robust_accuracy": evaluate_metric("accuracy", truth, robust_predictions),
}
print(metrics)
record_run(
    paths.output_dir,
    task="robustness_validation_drill",
    model=recipe["model"],
    configuration={
        "attack": ATTACK,
        "epsilon": EPSILON,
        "pgd_steps": PGD_STEPS,
        "step_size": PGD_STEP_SIZE,
    },
    validation_metrics=metrics,
    checkpoint=str(SOURCE_RUN / "best.pt"),
    notes=["Held-out robustness drill; adapt official attack and output contracts separately."],
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
