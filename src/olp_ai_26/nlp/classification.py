"""CPU TF-IDF and local-first Hugging Face text-classification baselines."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from olp_ai_26.core.artifacts import load_pickle, save_pickle
from olp_ai_26.nlp.text_cleaning import normalize_text


class TfidfTextClassifier:
    """Strong CPU-first baseline combining word and character n-grams."""

    def __init__(
        self,
        *,
        max_features: int = 200_000,
        class_weight: str | dict[Any, float] | None = "balanced",
        random_state: int = 42,
        C: float = 4.0,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
    ) -> None:
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=word_ngram_range,
                        min_df=min_df,
                        max_features=max_features // 2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=char_ngram_range,
                        min_df=min_df,
                        max_features=max_features // 2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.pipeline = Pipeline([("features", features), ("classifier", classifier)])

    @staticmethod
    def _clean(texts: Iterable[object]) -> list[str]:
        return [normalize_text(text, lowercase=True, replace_urls=True) for text in texts]

    def fit(self, texts: Iterable[object], labels: Iterable[object]) -> TfidfTextClassifier:
        """Normalize texts and fit the sparse feature/classifier pipeline."""
        self.pipeline.fit(self._clean(texts), list(labels))
        return self

    def predict(self, texts: Iterable[object]) -> np.ndarray:
        """Predict original label values for normalized input texts."""
        return self.pipeline.predict(self._clean(texts))

    def predict_proba(self, texts: Iterable[object]) -> np.ndarray:
        """Predict class probabilities in the fitted classifier's class order."""
        return self.pipeline.predict_proba(self._clean(texts))

    def save(self, path: Path | str) -> Path:
        """Save the fitted vectorizers, classifier, and normalization wrapper atomically."""
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: Path | str) -> TfidfTextClassifier:
        """Reload a trusted saved text pipeline and check its type."""
        value = load_pickle(path)
        if not isinstance(value, cls):
            raise TypeError("Saved artifact is not a TfidfTextClassifier")
        return value


def build_hf_text_classifier(
    model_name_or_path: Path | str,
    *,
    num_labels: int,
    pretrained_allowed: bool = False,
    local_files_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Build a sequence classifier from legal local/config weights without implicit downloads."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    source = str(model_name_or_path)
    if pretrained_allowed:
        return AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=num_labels,
            local_files_only=local_files_only,
            **kwargs,
        )
    config = AutoConfig.from_pretrained(source, local_files_only=local_files_only)
    config.num_labels = num_labels
    return AutoModelForSequenceClassification.from_config(config, **kwargs)
