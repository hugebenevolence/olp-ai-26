"""Computer-vision datasets, model factories, losses, and inference helpers."""

from olp_ai_26.cv.anomaly_detection import AnomalyImageDataset, TimmPatchFeatureExtractor
from olp_ai_26.cv.classification import ImageTableDataset, build_image_classifier
from olp_ai_26.cv.model_catalog import MODEL_PRESETS, list_supported_classifiers
from olp_ai_26.cv.video_classification import TemporalPoolingClassifier, VideoTableDataset

__all__ = [
    "AnomalyImageDataset",
    "ImageTableDataset",
    "MODEL_PRESETS",
    "TemporalPoolingClassifier",
    "TimmPatchFeatureExtractor",
    "VideoTableDataset",
    "build_image_classifier",
    "list_supported_classifiers",
]
