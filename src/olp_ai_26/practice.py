"""Offline synthetic task packs for pipeline drills, never for official contest training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from olp_ai_26.core.artifacts import save_json


def create_practice_pack(destination: Path | str, *, task: str = "text", seed: int = 42) -> Path:
    """Create a small reproducible pack with shuffled sample IDs and separate grading answers.

    Refuse to overwrite a nonempty pack. Synthetic scores exercise plumbing, not real task quality.
    """
    if task not in {"text", "tabular", "image", "segmentation"}:
        raise ValueError("Unknown practice task")
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Practice pack is not empty: {root}")
    data, grading = root / "data", root / "grading"
    data.mkdir(parents=True, exist_ok=True)
    grading.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count = 240 if task in {"text", "tabular"} else 48
    rows = []
    masks = []
    for index in range(count):
        row = {"id": f"case-{index:04d}"}
        if task == "text":
            label = index % 2
            vocabulary = ("tốt hay thích dễ hiểu", "tệ khó chán không thích")
            row.update(text=f"{vocabulary[label]} môn học ví dụ {index}", label=label)
        elif task == "tabular":
            first, second = rng.normal(size=2)
            region = str(rng.choice(["north", "central", "south"]))
            label = int(first + 0.6 * second + (0.5 if region == "south" else 0) > 0)
            row.update(feature_a=first, feature_b=second, region=region, label=label)
            if index % 11 == 0:
                row["feature_b"] = np.nan
        else:
            (data / "images").mkdir(exist_ok=True)
            label = index % 2
            image = Image.new("RGB", (40, 28), (180, 20, 20) if label else (20, 20, 180))
            relative = f"images/{row['id']}.png"
            row.update(image=relative, label=label)
            if task == "segmentation":
                (data / "masks").mkdir(exist_ok=True)
                mask = Image.new("L", image.size, 0)
                x, y = int(rng.integers(3, 14)), int(rng.integers(3, 8))
                rectangle = (x, y, x + 15, y + 10)
                ImageDraw.Draw(mask).rectangle(rectangle, fill=255)
                ImageDraw.Draw(image).rectangle(rectangle, fill=(220, 220, 220))
                row["mask"] = f"masks/{row['id']}.png"
                mask.save(data / row["mask"])
                masks.append(np.asarray(mask) > 0)
            image.save(data / relative)
        rows.append(row)
    frame = pd.DataFrame(rows)
    cut = count * 5 // 6
    train, test = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
    target = "rle" if task == "segmentation" else "label"
    if task == "segmentation":
        from olp_ai_26.cv.segmentation import mask_to_rle

        answers = pd.DataFrame(
            {"id": test["id"], target: [mask_to_rle(mask) for mask in masks[cut:]]}
        )
        train = train.drop(columns="label")
        test = test.drop(columns=["label", "mask"])
    else:
        answers = test[["id", "label"]].copy()
        test = test.drop(columns="label")
    train.to_csv(data / "train.csv", index=False, encoding="utf-8")
    test.to_csv(data / "test.csv", index=False, encoding="utf-8")
    sample = test[["id"]].sample(frac=1, random_state=seed).reset_index(drop=True)
    sample[target] = "" if task == "segmentation" else 0
    sample.to_csv(data / "sample_submission.csv", index=False, encoding="utf-8")
    answers.to_csv(grading / "answers.csv", index=False, encoding="utf-8")
    save_json(
        {
            "task": task,
            "seed": seed,
            "target": target,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        root / "pack.json",
    )
    (root / "README.md").write_text(
        "# Synthetic practice pack\n\nSet notebook DATA_DIR to this pack's data/ directory.\n"
        "The sample deliberately permutes test IDs. Carry prediction IDs through inference.\n"
        "Use only train.csv for fitting and validation.\n"
        "Keep grading/answers.csv closed until delivery.\n"
        "These toy scores check the workflow; they do not predict competition performance.\n"
        "Never use this synthetic pack as additional official contest training data.\n",
        encoding="utf-8",
    )
    return root
