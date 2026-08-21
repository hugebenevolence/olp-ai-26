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
from olp_ai_26.cv.detection import BoxDetectionDataset, detection_collate
from olp_ai_26.cv.instance_segmentation import InstanceMaskDataset
from olp_ai_26.cv.model_catalog import MODEL_PRESETS, list_supported_classifiers
from olp_ai_26.cv.multitask import ClassificationSegmentationLoss, ClassificationSegmentationModel
from olp_ai_26.cv.segmentation import mask_to_rle, rle_to_mask
from olp_ai_26.cv.tta import build_classification_tta, predict_segmentation_tta


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


def test_model_catalog_and_classification_tta():
    assert MODEL_PRESETS["resnet18"].image_size == 224
    assert "resnet18" in list_supported_classifiers("resnet18")
    image = torch.arange(12).reshape(1, 1, 3, 4)
    identity, horizontal = build_classification_tta(("identity", "hflip"))
    assert torch.equal(identity(image), image)
    assert torch.equal(horizontal(horizontal(image)), image)


def test_segmentation_rle_roundtrip_and_tta():
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.uint8).numpy()
    assert (rle_to_mask(mask_to_rle(mask), mask.shape) == mask).all()
    model = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
    loader = [(torch.rand(2, 3, 8, 8), torch.zeros(2, 1, 8, 8))]
    output = predict_segmentation_tta(
        model, loader, device="cpu", names=("identity", "hflip"), mixed_precision=False
    )
    assert output.shape == (2, 1, 8, 8)


def test_detection_and_instance_datasets(tmp_path):
    Image.new("RGB", (20, 16), color=(10, 20, 30)).save(tmp_path / "image.png")
    mask = Image.new("L", (20, 16), color=0)
    for x in range(4, 10):
        for y in range(3, 8):
            mask.putpixel((x, y), 255)
    mask.save(tmp_path / "mask.png")
    boxes = pd.DataFrame(
        {"image": ["image.png"], "label": [1], "xmin": [4], "ymin": [3], "xmax": [10], "ymax": [8]}
    )
    box_dataset = BoxDetectionDataset(boxes, root=tmp_path)
    image, target = box_dataset[0]
    assert image.shape == (3, 16, 20)
    assert target["boxes"].shape == (1, 4)
    collated = detection_collate([box_dataset[0]])
    assert len(collated[0]) == 1
    instances = InstanceMaskDataset(
        pd.DataFrame({"image": ["image.png"], "mask": ["mask.png"], "label": [1]}),
        root=tmp_path,
    )
    _, instance_target = instances[0]
    assert instance_target["masks"].shape == (1, 16, 20)
    assert instance_target["boxes"].tolist() == [[4.0, 3.0, 10.0, 8.0]]


def test_classification_segmentation_model_and_loss():
    model = ClassificationSegmentationModel("resnet18", 3, decoder_channels=8)
    images = torch.rand(2, 3, 64, 64)
    outputs = model(images)
    assert outputs["class_logits"].shape == (2, 3)
    assert outputs["mask_logits"].shape == (2, 1, 64, 64)
    loss, parts = ClassificationSegmentationLoss()(
        outputs, torch.tensor([0, 1]), torch.zeros(2, 1, 64, 64)
    )
    assert loss.isfinite()
    assert set(parts) == {"classification_loss", "segmentation_loss"}
