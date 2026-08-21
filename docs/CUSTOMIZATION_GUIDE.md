# Drag-and-plug customization cookbook

This page is intentionally syntax-first. Copy a block into the configuration/model section of the
nearest notebook, then change one experimental axis at a time.

## Model, optimizer, scheduler, and loss

```python
# Any timm name returned by list_supported_classifiers("*keyword*")
MODEL_NAME = "convnext_tiny"
IMAGE_SIZE = 224
BATCH_SIZE = 32

training = TrainerConfig(
    epochs=12,
    optimizer="adamw",  # adamw | adam | sgd | adafactor
    learning_rate=3e-4,
    weight_decay=1e-4,
    scheduler="cosine",  # cosine | onecycle | plateau | none
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    mixed_precision="auto",  # BF16 when supported, otherwise FP16; off on CPU
    patience=3,
)

criterion = classification_loss(label_smoothing=0.05)
# Imbalanced single-label: classification_loss(weights=class_weights(labels, mapping))
# Multi-label: classification_loss(multilabel=True, weights=positive_class_weights)
```

## Test-time augmentation

```python
from olp_ai_26.cv.tta import build_classification_tta

TTA_NAMES = ("identity", "hflip")
logits = predict_logits(
    model,
    test_loader,
    device="auto",
    transforms=build_classification_tta(TTA_NAMES),
)
```

Available classification names: `identity`, `hflip`, `vflip`, `rot90`. Do not use vertical flips
or rotations when label semantics depend on orientation. TTA costs approximately one inference
pass per transform and must improve the unchanged validation split before final use.

Segmentation uses inverse transforms to realign masks:

```python
logits = predict_segmentation_tta(model, test_loader, device="auto", names=("identity", "hflip"))
masks = logits.sigmoid() >= 0.5
```

## Semantic segmentation swap

```python
model = build_segmentation_model(
    architecture="deeplabv3plus",
    encoder_name="resnet50",
    classes=1,  # use K for multiclass logits
    pretrained_allowed=False,
)
criterion = DiceBCELoss(dice_weight=0.5)
```

For mutually exclusive K-class masks, replace binary BCE/Dice with cross entropy and decode with
`logits.argmax(dim=1)`. Tune the binary threshold on validation predictions, never the test set.

## Classification with auxiliary masks

```python
model = ClassificationSegmentationModel(
    "convnext_tiny",
    num_classes=NUM_CLASSES,
    mask_classes=1,
    decoder_channels=128,
    pretrained_allowed=False,
)
criterion = ClassificationSegmentationLoss(class_weight=1.0, mask_weight=0.4)
outputs = model(images)
loss, parts = criterion(outputs, class_labels, masks)
class_predictions = outputs["class_logits"].argmax(1)
```

The auxiliary mask weight is an experiment. Compare class validation score at 0, 0.2, and 0.5;
no gain means remove the head for the final run.

## Detection and instance segmentation

```python
loader = DataLoader(dataset, batch_size=4, collate_fn=detection_collate)
model = build_detection_model(
    num_classes=1 + foreground_class_count,
    architecture="fasterrcnn_mobilenet_v3_large_fpn",
    pretrained_allowed=False,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
loss = train_detection_epoch(model, loader, optimizer, device="auto")
```

Swap to instances with `InstanceMaskDataset` and `build_instance_segmentation_model`. Box and
instance models consume lists of images rather than one stacked image tensor.

## Freeze and unfreeze

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad = False

# Later, unfreeze with a smaller learning rate.
for parameter in model.encoder.parameters():
    parameter.requires_grad = True
```

Freezing a randomly initialized encoder is normally a mistake. It is useful only with legal,
meaningful pretrained/local weights.

## Ensemble only compatible outputs

```python
from olp_ai_26.core.inference import weighted_ensemble

ensemble_logits = weighted_ensemble([fold1_logits, fold2_logits], weights=[0.6, 0.4])
```

All arrays must have identical row/class ordering. Choose weights from validation predictions;
never infer weights from public leaderboard noise.

## Colab OOM recovery order

1. Reduce batch size.
2. Increase gradient accumulation to preserve effective batch size.
3. Reduce image size or sequence length if task signal permits.
4. Choose a smaller backbone.
5. Enable gradient checkpointing when the model supports it.

Do not reinstall or replace Colab's CUDA-matched PyTorch stack to solve an OOM.
