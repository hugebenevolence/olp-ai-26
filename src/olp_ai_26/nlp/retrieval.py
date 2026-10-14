"""Sparse TF-IDF document retrieval and mean reciprocal-rank evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from olp_ai_26.core.artifacts import load_pickle, save_pickle
from olp_ai_26.nlp.text_cleaning import normalize_text


class TfidfRetriever:
    """Fit a sparse word n-gram index and return cosine-ranked document positions."""

    def __init__(
        self, *, ngram_range: tuple[int, int] = (1, 2), max_features: int = 200_000
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, max_features=max_features, sublinear_tf=True
        )
        self.documents: list[str] = []
        self.matrix: Any = None

    def fit(self, documents: Iterable[object]) -> TfidfRetriever:
        """Normalize documents and fit/store their sparse TF-IDF matrix."""
        self.documents = [normalize_text(value, lowercase=True) for value in documents]
        self.matrix = self.vectorizer.fit_transform(self.documents)
        return self

    def search(
        self, queries: Iterable[object], top_k: int = 5, *, query_batch_size: int = 32
    ) -> list[list[tuple[int, float]]]:
        """Return top document positions and cosine scores for every query."""
        if self.matrix is None:
            raise RuntimeError("Call fit before search")
        if top_k < 1 or query_batch_size < 1:
            raise ValueError("top_k and query_batch_size must be positive")
        cleaned = [normalize_text(value, lowercase=True) for value in queries]
        results = []
        for start in range(0, len(cleaned), query_batch_size):
            query_matrix = self.vectorizer.transform(cleaned[start : start + query_batch_size])
            scores = cosine_similarity(query_matrix, self.matrix)
            for row in scores:
                indices = np.argsort(-row, kind="stable")[:top_k]
                results.append([(int(index), float(row[index])) for index in indices])
        return results

    def save(self, path: Path | str) -> Path:
        """Save the fitted vocabulary, document order, and sparse corpus index."""
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: Path | str) -> TfidfRetriever:
        """Reload a trusted local retrieval index."""
        value = load_pickle(path)
        if not isinstance(value, cls) or value.matrix is None:
            raise TypeError("Artifact must contain a fitted TfidfRetriever")
        return value


def reciprocal_rank(relevant: Sequence[set[int]], ranked: Sequence[Sequence[int]]) -> float:
    """Compute mean reciprocal rank from relevant-item sets and ranked item sequences."""
    values = []
    for expected, predictions in zip(relevant, ranked, strict=True):
        rank = next(
            (index for index, item in enumerate(predictions, start=1) if item in expected), None
        )
        values.append(0.0 if rank is None else 1.0 / rank)
    return float(np.mean(values)) if values else 0.0
