from __future__ import annotations

import zipfile

import pandas as pd
import torch
from PIL import Image

from olp_ai_26.cv.adversarial import fgsm, perturbation_statistics
from olp_ai_26.cv.anomaly_detection import (
    AnomalyImageDataset,
    DinoV2PatchFeatureExtractor,
    TimmPatchFeatureExtractor,
    aggregate_patch_neighborhoods,
    apply_normal_augmentation,
    apply_synthetic_anomaly,
    calibrate_anomaly_threshold,
    changed_patch_mask,
    cutpaste_batch,
    discover_normal_images,
    fit_positive_evidence_head,
    fixed_count_rank_fusion,
    load_official_training_table,
    patch_memory_distances,
    patch_memory_features,
    patch_memory_scores,
    positive_evidence_scores,
    sample_memory_bank,
    select_anomaly_threshold,
    stage_official_task2_data,
    summarize_patch_distances,
    synthetic_anomaly_defaults,
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
    feature_statistics = patch_memory_features(embeddings, bank, top_k=2)
    assert feature_statistics.shape == (1, 6)
    scores = patch_memory_scores(embeddings, bank, top_k=2)
    assert scores.shape == (1,)
    assert torch.equal(scores, feature_statistics[:, -1])
    assert scores.item() >= 0
    attacked = cutpaste_batch(image.unsqueeze(0), seed=42)
    assert attacked.shape == image.unsqueeze(0).shape
    calibrated = calibrate_anomaly_threshold([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    assert 0.3 < calibrated["threshold"] < 0.7
    assert calibrated["proxy_balanced_accuracy"] == 1.0
    assert select_anomaly_threshold(calibrated, "min_synthetic_quantile") <= calibrated["threshold"]
    augmented = apply_normal_augmentation(image.unsqueeze(0), "contrast_up")
    assert augmented.shape == image.unsqueeze(0).shape
    assert augmented.min() >= 0 and augmented.max() <= 1
    normal_blur = apply_normal_augmentation(image.unsqueeze(0), "blur_mild")
    assert normal_blur.shape == image.unsqueeze(0).shape
    for synthetic_name in (
        "cutpaste",
        "mixup",
        "cutmix",
        "dark_curve",
        "white_line",
        "gray_curve",
        "white_curve",
    ):
        synthetic = apply_synthetic_anomaly(
            image.unsqueeze(0).repeat(2, 1, 1, 1), synthetic_name, seed=42
        )
        repeated = apply_synthetic_anomaly(
            image.unsqueeze(0).repeat(2, 1, 1, 1), synthetic_name, seed=42
        )
        assert synthetic.shape == (2, 3, 32, 32)
        assert synthetic.min() >= 0 and synthetic.max() <= 1
        assert torch.equal(synthetic, repeated)
    mixup_input = torch.stack((torch.zeros(3, 16, 16), torch.ones(3, 16, 16)))
    mixed = apply_synthetic_anomaly(mixup_input, "mixup", seed=42)
    assert torch.all((mixed > 0) & (mixed < 1))
    assert not torch.equal(mixed, mixup_input)
    weak_mix = apply_synthetic_anomaly(
        mixup_input, "mixup", seed=42, mixup_alpha_range=(0.10, 0.10)
    )
    strong_mix = apply_synthetic_anomaly(
        mixup_input, "mixup", seed=42, mixup_alpha_range=(0.45, 0.45)
    )
    assert torch.mean(torch.abs(strong_mix - mixup_input)) > torch.mean(
        torch.abs(weak_mix - mixup_input)
    )
    curve_source = torch.ones(2, 3, 32, 32)
    light_curve = apply_synthetic_anomaly(curve_source, "dark_curve", seed=42, curve_darkness=0.20)
    dark_curve = apply_synthetic_anomaly(curve_source, "dark_curve", seed=42, curve_darkness=0.90)
    assert dark_curve.mean() < light_curve.mean()
    defaults = synthetic_anomaly_defaults()
    assert "blur" not in defaults
    assert defaults["cutmix"]["cutmix_opacity_range"] == (0.06, 0.14)
    assert defaults["gray_curve"]["curve_opacity_range"] == (0.10, 0.25)
    assert defaults["white_curve"]["curve_color_range"] == (0.85, 1.00)
    marked_source = torch.full((2, 3, 64, 64), 0.4)
    internal_curve = apply_synthetic_anomaly(marked_source, "dark_curve", seed=42)
    internal_line = apply_synthetic_anomaly(marked_source, "white_line", seed=42)
    gray_curve = apply_synthetic_anomaly(marked_source, "gray_curve", seed=42)
    white_curve = apply_synthetic_anomaly(marked_source, "white_curve", seed=42)
    for marked in (internal_curve, internal_line, gray_curve, white_curve):
        assert torch.equal(marked[:, :, 0, :], marked_source[:, :, 0, :])
        assert torch.equal(marked[:, :, -1, :], marked_source[:, :, -1, :])
        assert torch.equal(marked[:, :, :, 0], marked_source[:, :, :, 0])
        assert torch.equal(marked[:, :, :, -1], marked_source[:, :, :, -1])
    normal_head_features = torch.tensor(
        [[0.0, 0.1, 0.0, 0.1, 0.0, 0.1], [0.1, 0.0, 0.1, 0.0, 0.1, 0.0]]
    )
    synthetic_head_features = torch.tensor(
        [[1.0, 0.9, 1.0, 0.9, 1.0, 0.9], [0.9, 1.0, 0.9, 1.0, 0.9, 1.0]]
    )
    evidence_head = fit_positive_evidence_head(
        normal_head_features,
        synthetic_head_features,
        seed=42,
    )
    evidence = positive_evidence_scores(
        torch.cat((normal_head_features, synthetic_head_features)), evidence_head
    )
    assert torch.all(evidence >= 0)
    assert evidence[2:].mean() > evidence[:2].mean()
    weighted_head = fit_positive_evidence_head(
        normal_head_features,
        synthetic_head_features,
        seed=42,
        synthetic_weights=torch.tensor([0.9, 0.1]),
    )
    assert torch.isfinite(weighted_head["coefficient"]).all()


def test_changed_patch_mask_preserves_thin_edits_and_dilates_context():
    original = torch.zeros(1, 3, 8, 8)
    transformed = original.clone()
    transformed[:, :, 1, 1] = 0.25
    native = changed_patch_mask(
        original,
        transformed,
        grid_size=(4, 4),
        difference_threshold=0.10,
        dilation=0,
    )
    assert native.shape == (1, 16)
    assert native.sum().item() == 1
    assert native.reshape(1, 4, 4)[0, 0, 0]

    dilated = changed_patch_mask(
        original,
        transformed,
        grid_size=(4, 4),
        difference_threshold=0.10,
        dilation=1,
    )
    assert dilated.sum().item() == 4
    assert not changed_patch_mask(
        original,
        transformed,
        grid_size=(4, 4),
        difference_threshold=0.30,
        dilation=0,
    ).any()


def test_fixed_count_rank_fusion_preserves_count_and_changes_order():
    base = [0.1, 0.2, 0.9, 0.8]
    patch = [0.9, 0.8, 0.1, 0.2]
    base_labels, base_ranks = fixed_count_rank_fusion(
        base,
        patch,
        anomaly_count=2,
        patch_weight=0.0,
    )
    patch_labels, patch_ranks = fixed_count_rank_fusion(
        base,
        patch,
        anomaly_count=2,
        patch_weight=1.0,
    )
    assert base_labels.tolist() == [0, 0, 1, 1]
    assert patch_labels.tolist() == [1, 1, 0, 0]
    assert base_labels.sum() == patch_labels.sum() == 2
    assert base_ranks.argmax() == 2
    assert patch_ranks.argmax() == 0
    _, tied_ranks = fixed_count_rank_fusion(
        base,
        [0.0, 0.0, 0.0, 1.0],
        anomaly_count=2,
        patch_weight=1.0,
    )
    assert tied_ranks[:3].tolist() == [0.5, 0.5, 0.5]


def test_anomalydino_extractor_and_cosine_tail_scoring():
    extractor = DinoV2PatchFeatureExtractor(
        "vit_small_patch14_dinov2.lvd142m",
        image_size=56,
        pretrained_allowed=False,
    ).eval()
    with torch.inference_mode():
        tokens = extractor(torch.rand(1, 3, 56, 56))
    assert tokens.shape == (1, 16, 384)
    assert torch.allclose(tokens.norm(dim=-1), torch.ones(1, 16), atol=1e-5)

    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    statistics = patch_memory_features(
        embeddings,
        bank,
        top_fraction=0.25,
        distance_metric="cosine",
    )
    assert statistics.shape == (1, 6)
    assert torch.allclose(statistics[:, -1], torch.tensor([1.0]), atol=1e-6)


def test_multidegree_patch_aggregation_and_distance_fusion():
    """Local context must preserve the grid and support patch-level score fusion."""
    tokens = torch.zeros(1, 9, 2)
    tokens[0, 4] = torch.tensor([1.0, 0.0])
    degree_one = aggregate_patch_neighborhoods(tokens, kernel_size=1, grid_size=(3, 3))
    degree_three = aggregate_patch_neighborhoods(tokens, kernel_size=3, grid_size=(3, 3))
    assert degree_one.shape == degree_three.shape == tokens.shape
    assert torch.count_nonzero(degree_three[..., 0]) == 9
    bank = torch.tensor([[0.0, 1.0]])
    native_distances = patch_memory_distances(
        degree_one,
        bank,
        distance_metric="cosine",
    )
    context_distances = patch_memory_distances(
        degree_three,
        bank,
        distance_metric="cosine",
    )
    fused = summarize_patch_distances(
        (native_distances + context_distances) / 2,
        top_fraction=1 / 9,
    )
    assert native_distances.shape == context_distances.shape == (1, 9)
    assert fused.shape == (1, 6)
    assert fused[0, -1] >= 0


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
