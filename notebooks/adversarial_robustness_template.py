# %% [markdown]
# # Adversarial robustness baseline (FGSM and PGD)
# Use only when the task explicitly asks for attacks/defenses or robustness evaluation.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import torch

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.cv.adversarial import fgsm, perturbation_statistics, pgd
from olp_ai_26.cv.classification import build_image_classifier

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
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, paths.data_dir)
MODEL_NAME, NUM_CLASSES = "resnet18", 10
ATTACK = "pgd"  # fgsm | pgd
EPSILON = 8 / 255
PGD_STEP_SIZE, PGD_STEPS = 2 / 255, 10
CLIP_MIN, CLIP_MAX = 0.0, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(gpu_report())

# %% [markdown]
# ## 1. Load the trained clean classifier

# %%
model = build_image_classifier(MODEL_NAME, NUM_CLASSES, pretrained_allowed=False).to(DEVICE)
checkpoint = paths.output_dir / "best.pt"  # point at the clean model checkpoint
if checkpoint.exists():
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
model.eval()

# %% [markdown]
# ## 2. Attack block
# Replace `clean_images, labels` with a real batch from the classification loader.

# %%
clean_images = torch.rand(8, 3, 224, 224, device=DEVICE)
labels = torch.randint(NUM_CLASSES, (8,), device=DEVICE)
if ATTACK == "fgsm":
    attacked = fgsm(
        model, clean_images, labels, epsilon=EPSILON, clip_min=CLIP_MIN, clip_max=CLIP_MAX
    )
else:
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
with torch.inference_mode():
    clean_accuracy = (model(clean_images).argmax(1) == labels).float().mean().item()
    robust_accuracy = (model(attacked).argmax(1) == labels).float().mean().item()
print({"clean_accuracy": clean_accuracy, "robust_accuracy": robust_accuracy})
print(perturbation_statistics(clean_images, attacked))
sync_artifacts(paths.output_dir, paths.persistent_dir)
