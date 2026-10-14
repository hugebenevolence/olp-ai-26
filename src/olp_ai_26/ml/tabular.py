"""Small classification/regression pipelines with train-only preprocessing and cold reload."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from olp_ai_26.core.artifacts import load_pickle, save_pickle


@dataclass(slots=True)
class TabularConfig:
    """Explicit task, model family, random seed, and bounded boosting budget."""

    task: Literal["classification", "regression"] = "classification"
    model: Literal["dummy", "linear", "hist_gradient_boosting"] = "linear"
    seed: int = 42
    max_iter: int = 200
    learning_rate: float = 0.1
    max_leaf_nodes: int = 15

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if self.model not in {"dummy", "linear", "hist_gradient_boosting"}:
            raise ValueError("Unsupported tabular model")
        if self.max_iter < 1 or self.learning_rate <= 0 or self.max_leaf_nodes < 2:
            raise ValueError("Invalid tabular training budget")


class TabularBaseline:
    """Fit preprocessing and estimator together; retain original feature and class order."""

    def __init__(self, config: TabularConfig | None = None) -> None:
        self.config = config or TabularConfig()
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.pipeline: Pipeline | None = None

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        result = frame[self.feature_columns].copy()
        for column in self.numeric_columns:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
            if np.isinf(result[column]).any():
                raise ValueError(f"Infinite values in numeric feature: {column}")
        for column in self.categorical_columns:
            result[column] = (
                result[column]
                .map(lambda value: np.nan if pd.isna(value) else str(value))
                .astype(object)
            )
        return result

    def fit(self, frame: pd.DataFrame, labels: Any) -> TabularBaseline:
        """Fit on the training fold; explicitly exclude ID, target, group, and time features."""
        if frame.empty or frame.columns.duplicated().any():
            raise ValueError("Training features must be nonempty with unique column names")
        self.feature_columns = frame.columns.tolist()
        self.numeric_columns = frame.select_dtypes(include="number").columns.tolist()
        self.categorical_columns = [
            column for column in self.feature_columns if column not in self.numeric_columns
        ]
        targets = np.asarray(labels)
        if len(targets) != len(frame) or pd.isna(targets).any():
            raise ValueError("Labels must be nonmissing and match the training rows")
        config = self.config
        boosting = config.model == "hist_gradient_boosting"
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", "passthrough" if boosting else StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant", fill_value="__missing__", keep_empty_features=True
                    ),
                ),
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1, max_categories=255
                    )
                    if boosting
                    else OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric, self.numeric_columns),
                ("categorical", categorical, self.categorical_columns),
            ]
        )
        classification = config.task == "classification"
        if config.model == "dummy":
            estimator = DummyClassifier(strategy="prior") if classification else DummyRegressor()
        elif config.model == "linear":
            estimator = (
                LogisticRegression(max_iter=1000, random_state=config.seed)
                if classification
                else Ridge(solver="lsqr")
            )
        else:
            factory = (
                HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
            )
            estimator = factory(
                max_iter=config.max_iter,
                learning_rate=config.learning_rate,
                max_leaf_nodes=config.max_leaf_nodes,
                min_samples_leaf=10,
                categorical_features=list(
                    range(len(self.numeric_columns), len(self.feature_columns))
                ),
                # Internal row-random validation may violate the externally chosen group/time split.
                early_stopping=False,
                random_state=config.seed,
            )
        self.pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
        self.pipeline.fit(self._features(frame), targets)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict using the fitted preprocessing and original training feature order."""
        if self.pipeline is None:
            raise RuntimeError("Call fit before predict")
        return self.pipeline.predict(self._features(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return classification probabilities in ``classes_`` order."""
        if self.pipeline is None or self.config.task != "classification":
            raise RuntimeError("Fit a classification pipeline before predict_proba")
        return self.pipeline.predict_proba(self._features(frame))

    @property
    def classes_(self) -> np.ndarray:
        """Return fitted class values in probability-column order."""
        if self.pipeline is None or self.config.task != "classification":
            raise RuntimeError("Classes are available only after classification fitting")
        return self.pipeline.named_steps["model"].classes_

    def save(self, path: Path | str) -> Path:
        """Atomically save fitted preprocessing, feature order, and estimator."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted pipeline")
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: Path | str) -> TabularBaseline:
        """Reload a trusted locally produced tabular pipeline."""
        value = load_pickle(path)
        if not isinstance(value, cls) or value.pipeline is None:
            raise TypeError("Artifact must contain a fitted TabularBaseline")
        return value
