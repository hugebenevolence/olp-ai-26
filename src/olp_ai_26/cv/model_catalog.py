"""Discoverable timm classifier presets for fast competition model swapping.

The curated presets are safe starting points, not a hard allowlist. Any model name returned by
``timm.list_models()`` can be passed to :func:`build_image_classifier`.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm


@dataclass(frozen=True)
class ModelPreset:
    """Practical defaults and memory guidance for one classifier backbone."""

    name: str
    family: str
    image_size: int
    colab_batch_size: int
    notes: str


MODEL_PRESETS: dict[str, ModelPreset] = {
    item.name: item
    for item in (
        ModelPreset("mobilenetv3_small_100", "CNN", 224, 128, "Fast smoke-test baseline."),
        ModelPreset("resnet18", "CNN", 224, 64, "Reliable debugging baseline."),
        ModelPreset("resnet50", "CNN", 224, 48, "Stronger residual baseline."),
        ModelPreset("efficientnet_b0", "CNN", 224, 64, "Good accuracy/speed trade-off."),
        ModelPreset("efficientnet_b3", "CNN", 300, 24, "Higher-resolution EfficientNet."),
        ModelPreset("convnext_tiny", "modern CNN", 224, 32, "Strong general-purpose baseline."),
        ModelPreset("vit_small_patch16_224", "transformer", 224, 48, "Compact ViT."),
        ModelPreset(
            "swin_tiny_patch4_window7_224", "transformer", 224, 32, "Hierarchical transformer."
        ),
        ModelPreset("maxvit_tiny_tf_224", "hybrid", 224, 24, "Strong but memory-heavier."),
    )
}


def get_model_preset(name: str) -> ModelPreset:
    """Return a curated preset or raise an error that explains model discovery syntax."""
    try:
        return MODEL_PRESETS[name]
    except KeyError as error:
        raise KeyError(
            f"No curated preset for {name!r}. The model can still be valid; "
            "use list_supported_classifiers('*keyword*') and set image/batch size manually."
        ) from error


def list_supported_classifiers(pattern: str = "*", *, pretrained: bool = False) -> list[str]:
    """List timm model identifiers accepted by the classifier factory.

    Args:
        pattern: Shell-style model pattern such as ``"*convnext*"`` or ``"resnet*"``.
        pretrained: When true, only list models with published pretrained weights. This only
            describes availability; it does not grant permission to download those weights.
    """
    return timm.list_models(pattern, pretrained=pretrained)


def model_preset_rows() -> list[dict[str, str | int]]:
    """Return the curated catalog as table-ready dictionaries for a notebook display cell."""
    return [
        {
            "name": item.name,
            "family": item.family,
            "image_size": item.image_size,
            "colab_batch_size": item.colab_batch_size,
            "notes": item.notes,
        }
        for item in MODEL_PRESETS.values()
    ]
