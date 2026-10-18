"""Tiny offline fixtures with historical Kaggle schemas; real data must be acquired separately."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

LEARNING_FAMILIES = (
    "01_credit",
    "02_prices",
    "03_toxic",
    "04_leaves",
    "05_salt",
    "06_wheat",
    "07_matching",
)
TOXIC_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def create_learning_pack(destination: Path | str, *, family: str, seed: int = 42) -> Path:
    """Create a small schema fixture, separate held-out labels and deliberately permuted sample IDs.

    These artificial rows prove execution and integrity only, never competition performance.
    No competition data, credentials, downloaded assets or external training examples are used.
    """
    if family not in LEARNING_FAMILIES:
        raise ValueError(f"Choose one of {LEARNING_FAMILIES}")
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, box_rows = [], []
    count = 240 if family in LEARNING_FAMILIES[:3] else 60
    id_column, target = "id", "label"
    for i in range(count):
        key = f"case-{i:04d}"
        if family == "01_credit":
            id_column, target = "SK_ID_CURR", "TARGET"
            label = int(i % 8 == 0)
            rows.append(
                {
                    id_column: key,
                    target: label,
                    "AMT_INCOME_TOTAL": 20 + 10 * label + rng.normal(),
                    "AMT_CREDIT": 30 + rng.normal(),
                    "NAME_CONTRACT_TYPE": "cash" if i % 3 else "loan",
                }
            )
        elif family == "02_prices":
            id_column, target = "Id", "SalePrice"
            size = 100 + rng.uniform(0, 200)
            rows.append(
                {
                    id_column: key,
                    target: np.exp(8 + size / 150 + rng.normal(0, 0.02)),
                    "GrLivArea": size,
                    "OverallQual": i % 5 + 1,
                    "Neighborhood": f"area-{i % 3}",
                }
            )
        elif family == "03_toxic":
            active = [(i >> j) & 1 for j in range(6)]
            words = " ".join(f"signal{j}" for j, value in enumerate(active) if value)
            rows.append(
                {
                    "id": key,
                    "comment_text": f"ordinary text {words}",
                    **dict(zip(TOXIC_LABELS, active, strict=True)),
                }
            )
        elif family == "07_matching":
            id_column, target = "posting_id", "matches"
            group = i // 3
            rows.append(
                {
                    id_column: key,
                    "label_group": str(group),
                    "title": f"brand item{group} product",
                    "image": f"{key}.png",
                    "image_phash": f"phash-{group}",
                }
            )
            folder = root / "train_images"  # test images are moved below after the split
            folder.mkdir(exist_ok=True)
            Image.new("RGB", (32, 24), (group * 17 % 255, 80, 120)).save(folder / f"{key}.png")
        else:
            training = i < count * 4 // 5
            folder_name = "train_images" if family == "04_leaves" else "train/images"
            test_folder = "test_images" if family == "04_leaves" else "test/images"
            if family == "06_wheat":
                folder_name, test_folder = "train", "test"
            folder = root / (folder_name if training else test_folder)
            folder.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (32, 24), (25, 30, 35))
            if family == "04_leaves":
                id_column, target = "image_id", "label"
                label = i % 5
                color = [
                    (200, 20, 20),
                    (20, 200, 20),
                    (20, 20, 200),
                    (200, 200, 20),
                    (180, 20, 180),
                ]
                image.paste(color[label], (0, 0, 32, 24))
                rows.append({id_column: f"{key}.png", target: label})
            elif family == "05_salt":
                id_column, target = "id", "rle_mask"
                mask = Image.new("L", image.size, 0)
                if i % 3:
                    ImageDraw.Draw(mask).rectangle((6, 5, 22, 18), fill=255)
                    ImageDraw.Draw(image).rectangle((6, 5, 22, 18), fill=(220, 220, 220))
                (root / "train/masks").mkdir(parents=True, exist_ok=True)
                if training:
                    mask.save(root / "train/masks" / f"{key}.png")
                from olp_ai_26.cv.segmentation import mask_to_rle

                rows.append({id_column: key, target: mask_to_rle(np.asarray(mask) > 0)})
            else:
                id_column, target = "image_id", "PredictionString"
                rectangle = (5, 4, 19, 16)
                ImageDraw.Draw(image).rectangle(rectangle, fill=(220, 190, 30))
                rows.append({id_column: key, target: "1 5 4 14 12"})
                if training:
                    box_rows.append(
                        {
                            id_column: key,
                            "width": 32,
                            "height": 24,
                            "source": f"camera-{i % 8}",
                            "bbox": "[5, 4, 14, 12]",
                        }
                    )
            image.save(folder / f"{key}.png")
    frame = pd.DataFrame(rows)
    cut = count * 4 // 5
    train, test = frame.iloc[:cut].copy(), frame.iloc[cut:].copy()
    if family == "07_matching":
        (root / "test_images").mkdir(exist_ok=True)
        for filename in test["image"]:
            (root / "train_images" / filename).replace(root / "test_images" / filename)
        target_values = [
            " ".join(test.loc[test["label_group"] == group, id_column])
            for group in test["label_group"]
        ]
        answers = pd.DataFrame({id_column: test[id_column], target: target_values})
        test = test.drop(columns="label_group")
        sample = test[[id_column]].copy()
        sample[target] = ""
    else:
        targets = list(TOXIC_LABELS) if family == "03_toxic" else [target]
        answers = test[[id_column, *targets]].copy()
        test = test.drop(columns=targets)
        sample = test[[id_column]].copy()
        for column in targets:
            sample[column] = "" if family in {"05_salt", "06_wheat"} else 0
    train_file = "application_train.csv" if family == "01_credit" else "train.csv"
    test_file = "application_test.csv" if family == "01_credit" else "test.csv"
    if family == "01_credit":
        pd.DataFrame(
            {
                "SK_ID_CURR": frame[id_column],
                "AMT_CREDIT_SUM": rng.uniform(1, 100, count),
                "DAYS_CREDIT": -rng.integers(10, 1000, count),
            }
        ).to_csv(root / "bureau.csv", index=False)
    if family == "05_salt":
        train = train[[id_column]]
    if family == "06_wheat":
        train = pd.DataFrame(box_rows)
    train.to_csv(root / train_file, index=False, encoding="utf-8")
    test.to_csv(root / test_file, index=False, encoding="utf-8")
    sample.sample(frac=1, random_state=seed).to_csv(root / "sample_submission.csv", index=False)
    (root / "grading").mkdir()
    answers.to_csv(root / "grading/answers.csv", index=False, encoding="utf-8")
    (root / "FIXTURE_ONLY.txt").write_text(
        "Synthetic schema fixture only. Scores measure plumbing, not competition quality.\n"
        "Keep grading answers closed during practice. Never add this to contest training.\n",
        encoding="utf-8",
    )
    return root
