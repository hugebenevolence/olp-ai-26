from __future__ import annotations

import pandas as pd
import torch
from PIL import Image

from olp_ai_26.cv.adversarial import fgsm, perturbation_statistics
from olp_ai_26.cv.classification import (
    ImageTableDataset,
    build_image_classifier,
    build_image_transforms,
)


def test_image_dataset_and_model_forward(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 32), color=(10, 20, 30)).save(image_path)
    frame = pd.DataFrame({"image": [image_path.name], "label": ["a"]})
    dataset = ImageTableDataset(
        frame,
        image_column="image",
        target_column="label",
        root=tmp_path,
        transform=build_image_transforms(size=32, training=False),
        label_to_index={"a": 0},
    )
    tensor, target = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert target.item() == 0
    model = build_image_classifier("resnet18", 2, pretrained_allowed=False).eval()
    with torch.inference_mode():
        assert model(tensor.unsqueeze(0)).shape == (1, 2)


def test_fgsm_respects_epsilon():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    images = torch.full((2, 1, 2, 2), 0.5)
    labels = torch.tensor([0, 1])
    attacked = fgsm(model, images, labels, epsilon=0.1)
    statistics = perturbation_statistics(images, attacked)
    assert statistics["linf"] <= 0.100001
    assert attacked.min() >= 0
    assert attacked.max() <= 1
