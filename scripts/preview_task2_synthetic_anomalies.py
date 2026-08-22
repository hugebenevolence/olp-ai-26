"""Render CPU previews for Task 2 normal blur and synthetic positive signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from olp_ai_26.cv.anomaly_detection import (
    AnomalyImageDataset,
    apply_normal_augmentation,
    apply_synthetic_anomaly,
    discover_normal_images,
    synthetic_anomaly_defaults,
)

METHODS = ("blur_mild", "cutmix", "dark_curve", "white_line")
IMAGE_SIZES = {
    "category_01": 288,
    "category_02": 320,
    "category_03": 288,
    "category_04": 320,
    "category_05": 288,
    "category_06": 288,
}


def _tensor_image(image: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor into a clipped HWC array for Matplotlib."""
    return image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def render_previews(train_root: Path, output_dir: Path, *, seed: int = 42) -> list[Path]:
    """Render actual augmentations and amplified difference maps for all six categories."""
    frame = discover_normal_images(train_root)
    defaults = synthetic_anomaly_defaults()
    categories = sorted(frame["category"].unique())
    figure, axes = plt.subplots(
        len(categories),
        len(METHODS) + 1,
        figsize=(20, 4 * len(categories)),
        squeeze=False,
    )
    difference_figure, difference_axes = plt.subplots(
        len(categories),
        len(METHODS),
        figsize=(16, 4 * len(categories)),
        squeeze=False,
    )
    for row_index, category in enumerate(categories):
        category_rows = frame.loc[frame["category"] == category].head(2)
        dataset = AnomalyImageDataset(
            category_rows,
            root=train_root,
            image_size=IMAGE_SIZES.get(category, 288),
        )
        images = torch.stack([dataset[index] for index in range(len(dataset))])
        base = images[0]
        axes[row_index, 0].imshow(_tensor_image(base))
        axes[row_index, 0].set_title(f"{category}\noriginal")
        axes[row_index, 0].axis("off")
        for column_index, method in enumerate(METHODS, start=1):
            transformed = (
                apply_normal_augmentation(images, method)[0]
                if method == "blur_mild"
                else apply_synthetic_anomaly(
                    images,
                    method,
                    seed=seed,
                    **defaults[method],
                )[0]
            )
            axes[row_index, column_index].imshow(_tensor_image(transformed))
            axes[row_index, column_index].set_title(method)
            axes[row_index, column_index].axis("off")
            difference = torch.abs(transformed - base).mean(dim=0)
            difference_axes[row_index, column_index - 1].imshow(
                difference.numpy(),
                cmap="magma",
                vmin=0,
                vmax=0.25,
            )
            difference_axes[row_index, column_index - 1].set_title(
                f"{category}\n{method} difference"
            )
            difference_axes[row_index, column_index - 1].axis("off")
    figure.suptitle("Task 2 CPU preview: normal blur and synthetic positive signals", fontsize=18)
    difference_figure.suptitle(
        "Task 2 CPU preview: difference maps (fixed 0-0.25 scale)",
        fontsize=18,
    )
    figure.tight_layout()
    difference_figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "all_categories_augmentations.png"
    difference_path = output_dir / "all_categories_differences.png"
    figure.savefig(preview_path, dpi=160, bbox_inches="tight")
    difference_figure.savefig(difference_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    plt.close(difference_figure)
    return [preview_path, difference_path]


def main() -> None:
    """Parse paths, render the CPU previews, and print their locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task2_cpu_augmentation_preview"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for output in render_previews(args.train_root, args.output_dir, seed=args.seed):
        print(output.resolve())


if __name__ == "__main__":
    main()
