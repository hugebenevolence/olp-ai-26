from __future__ import annotations

import zipfile

import pandas as pd
import torch
from PIL import Image

from olp_ai_26.cv.adversarial import fgsm, perturbation_statistics
from olp_ai_26.cv.anomaly_detection import (
    AnomalyImageDataset,
    TimmPatchFeatureExtractor,
    calibrate_anomaly_threshold,
    cutpaste_batch,
    discover_normal_images,
    load_official_training_table,
    patch_memory_scores,
    sample_memory_bank,
    stage_official_task2_data,
    validate_anomaly_submission,
)
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


def test_anomaly_feature_memory_and_calibration(tmp_path):
    category = tmp_path / "category_01"
    category.mkdir()
    Image.new("RGB", (24, 20), color=(10, 20, 30)).save(category / "normal.png")
    frame = discover_normal_images(tmp_path)
    assert frame.to_dict("records") == [
        {"category": "category_01", "relative_path": "category_01/normal.png"}
    ]
    image = AnomalyImageDataset(frame, root=tmp_path, image_size=32)[0]
    assert image.shape == (3, 32, 32)
    extractor = TimmPatchFeatureExtractor(
        "resnet18", out_indices=(1, 2), projection_dim=8, pretrained_allowed=False
    ).eval()
    with torch.inference_mode():
        embeddings = extractor(image.unsqueeze(0))
    bank = sample_memory_bank([embeddings], max_patches=10, seed=42)
    scores = patch_memory_scores(embeddings, bank, top_k=2)
    assert scores.shape == (1,)
    assert scores.item() >= 0
    attacked = cutpaste_batch(image.unsqueeze(0), seed=42)
    assert attacked.shape == image.unsqueeze(0).shape
    calibrated = calibrate_anomaly_threshold([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    assert 0.3 < calibrated["threshold"] < 0.7
    assert calibrated["proxy_balanced_accuracy"] == 1.0


def test_anomaly_submission_contract():
    test = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "category": ["category_01", "category_02"],
            "relative_path": ["images/a.png", "images/b.png"],
        }
    )
    submission = test[["sample_id", "category"]].copy()
    submission["label"] = [0, 1]
    validate_anomaly_submission(submission, test)


def test_stage_official_nested_task2_archive(tmp_path):
    source_image = tmp_path / "normal.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(source_image)
    train_zip = tmp_path / "dataset_train.zip"
    with zipfile.ZipFile(train_zip, "w") as archive:
        for csv_name in ("train1_6.csv", "train2_5.csv", "train3_4.csv"):
            rows = "sample_id,category,relative_path\n"
            if csv_name == "train1_6.csv":
                rows += "normal_1,category_01,train/category_01/normal.png\n"
            archive.writestr(f"dataset_train/{csv_name}", rows)
        archive.write(source_image, arcname="dataset_train/train/category_01/normal.png")
    public_zip = tmp_path / "public_test.zip"
    with zipfile.ZipFile(public_zip, "w") as archive:
        archive.writestr(
            "public_test/test.csv",
            "sample_id,category,relative_path\ntest_1,category_01,images/category_01/test.png\n",
        )
        archive.write(source_image, arcname="public_test/images/category_01/test.png")
    outer_zip = tmp_path / "official.zip"
    with zipfile.ZipFile(outer_zip, "w") as archive:
        archive.write(
            train_zip,
            arcname="CV_Data/training_dataset/dataset_train.zip",
        )
        archive.write(
            public_zip,
            arcname="CV_Data/public_test/public_test.zip",
        )
    resolved = stage_official_task2_data(outer_zip, tmp_path / "expanded", phase="public")
    assert resolved.training_root.name == "dataset_train"
    assert resolved.test_csv.is_file()
    training = load_official_training_table(resolved.training_root)
    assert training["sample_id"].tolist() == ["normal_1"]
