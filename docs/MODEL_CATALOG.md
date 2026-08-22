# Model discovery and selection

The examples in notebooks are starting points, **not the complete supported set**.

## Normal-only image anomaly detection

The task-specific notebook's reproducible 0.577 preset uses `wide_resnet50_2` for all categories.
Its category-specialized presets also use `resnet50` and `convnext_tiny`. Use `resnet18` for a
pipeline smoke test. The extractor accepts timm models that support `features_only=True`; ordinary
classifier discovery is necessary but not sufficient. Confirm a candidate with a no-pretrained
constructor/forward smoke test before committing a Colab run. Model weights are frozen, while
category-specific patch memories and thresholds are learned from official normal images.

## Image classification

`build_image_classifier()` accepts any identifier returned by the installed timm version:

```python
from olp_ai_26.cv.model_catalog import list_supported_classifiers, model_preset_rows

print(list_supported_classifiers("*convnext*"))
print(list_supported_classifiers("*swin*"))
print(list_supported_classifiers("vit*", pretrained=False))
```

Curated Colab starting points:

| Model | Family | Input | Approx. starting batch | Use |
|---|---:|---:|---:|---|
| `mobilenetv3_small_100` | CNN | 224 | 128 | pipeline smoke test |
| `resnet18` | CNN | 224 | 64 | debugging baseline |
| `resnet50` | CNN | 224 | 48 | stronger classic CNN |
| `efficientnet_b0` | CNN | 224 | 64 | speed/accuracy tradeoff |
| `efficientnet_b3` | CNN | 300 | 24 | higher resolution |
| `convnext_tiny` | modern CNN | 224 | 32 | strong default |
| `vit_small_patch16_224` | transformer | 224 | 48 | compact ViT |
| `swin_tiny_patch4_window7_224` | transformer | 224 | 32 | local-attention transformer |
| `maxvit_tiny_tf_224` | hybrid | 224 | 24 | strong, memory-heavier |

Batch sizes are rough starting values, not promises. Check `gpu_report()`, then reduce batch size
or image size after OOM. Preserve image resolution when small objects or fine textures determine
the label.

```python
MODEL_NAME = "convnext_tiny"
model = build_image_classifier(
    MODEL_NAME,
    num_classes=len(label_to_index),
    pretrained_allowed=False,  # change only if the rules permit those exact weights
    dropout=0.1,
)
```

## Semantic segmentation

Supported SMP architecture strings are:

```python
from olp_ai_26.cv.segmentation import SEGMENTATION_ARCHITECTURES

print(SEGMENTATION_ARCHITECTURES)

import segmentation_models_pytorch as smp

print(smp.encoders.get_encoder_names())  # complete list in the installed version
```

Use U-Net as the correctness baseline, FPN for multi-scale objects, and DeepLabV3+ for stronger
context modeling. Encoder examples include `resnet18`, `resnet50`, `efficientnet-b0`, and timm/
transformer encoders exposed by the installed SMP version.

## Detection and instance segmentation

Box detection accepts:

- `fasterrcnn_resnet50_fpn_v2`
- `fasterrcnn_resnet50_fpn`
- `fasterrcnn_mobilenet_v3_large_fpn`

Instance segmentation accepts `maskrcnn_resnet50_fpn_v2` and `maskrcnn_resnet50_fpn`.
Torchvision labels reserve 0 for background; real classes begin at 1.

## NLP model paths

Hugging Face helpers accept a local model directory or permitted Hub identifier. They default to
`local_files_only=True`; downloading weights is a policy decision, never an accidental side
effect. Reasonable model families depend on language and task: BERT/RoBERTa/DeBERTa for sequence
classification, token-classification variants for NER, and T5/BART-style models for seq2seq.
Verify Vietnamese tokenizer coverage before committing to a large run.
