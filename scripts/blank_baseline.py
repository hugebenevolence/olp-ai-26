"""Study answer key: rebuild these cells with the repository closed during timed drills.

Standalone single-label text/tabular classification; no olp_ai_26 imports or downloads.
The intentionally small contract is id + label, with a text column for text tasks.
This is educational reconstruction code, not another competition training framework.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# %% 1. Read and inspect the contract; keep literal IDs such as 001 and NA.
def read_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the practice schema, rejecting ambiguous identities and missing labels."""
    frames = [
        pd.read_csv(data_dir / name, converters={"id": str, "label": str})
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    ]
    for frame in frames:
        if "id" not in frame or frame["id"].eq("").any() or not frame["id"].is_unique:
            raise ValueError("Every manifest needs nonempty unique IDs")
    train, test, sample = frames
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("This answer key supports exactly id,label output")
    if "label" not in train or train["label"].eq("").any() or train["label"].nunique() < 2:
        raise ValueError("Need at least two classes and a label for every training row")
    if "label" in test or set(train["id"]) & set(test["id"]):
        raise ValueError("Expected disjoint labeled train and unlabeled test manifests")
    if set(test["id"]) != set(sample["id"]):
        raise ValueError("Test and sample IDs differ")
    return train, test, sample


# %% 2. Split before fitting anything. Choose the mode from the data-generating process.
def split_rows(
    train: pd.DataFrame, mode: str, column: str | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed train/validation row positions, using groups or chronology when needed."""
    indices = np.arange(len(train))
    if mode == "stratified":
        return train_test_split(indices, test_size=0.2, stratify=train["label"], random_state=seed)
    if column is None or train[column].isna().any():
        raise ValueError("Group/time split needs a complete split column")
    if mode == "group":
        a, b = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                train, groups=train[column]
            )
        )
        assert not set(train.iloc[a][column]) & set(train.iloc[b][column])
        return a, b
    if mode == "time":
        times = pd.to_datetime(train[column], errors="raise", utc=True)
        cutoff = times.sort_values().iloc[min(len(train) - 1, int(len(train) * 0.8))]
        # Equal timestamps stay together. Add a purge gap if the task requires one.
        return indices[(times < cutoff).to_numpy()], indices[(times >= cutoff).to_numpy()]
    raise ValueError("Choose stratified, group, or time")


# %% 3. Keep the estimator replaceable; fitted preprocessing stays inside the pipeline.
def make_model(task: str, features: pd.DataFrame | pd.Series, *, seed: int = 42) -> Pipeline:
    """Build a small CPU classifier with no pretrained assets or optional libraries."""
    classifier = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    if task == "text":
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1), classifier)
    if task != "tabular":
        raise ValueError("Choose text or tabular classification")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in features if name not in numeric]
    preprocessing = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )
    return make_pipeline(preprocessing, classifier)


# %% 4. Score all validation predictions together, using the specified label universe.
def score_labels(truth: ArrayLike, predictions: ArrayLike, metric: str, labels: list[str]) -> float:
    """Support two explicit practice metrics; replace this for a different official scorer."""
    if metric == "accuracy":
        return float(accuracy_score(truth, predictions))
    if metric == "macro_f1":
        return float(f1_score(truth, predictions, labels=labels, average="macro", zero_division=0))
    raise ValueError("Unsupported metric: implement the exact official scorer first")


# %% 5. Preserve a reloadable model, then attach predictions to IDs rather than row numbers.
def deliver(
    sample: pd.DataFrame, test_ids: pd.Series, predictions: ArrayLike, output: Path
) -> Path:
    """Align predictions and verify the exact UTF-8 CSV that will be submitted."""
    values = np.asarray(predictions)
    if values.shape != (len(test_ids),) or not test_ids.is_unique:
        raise ValueError("Expected one prediction per unique test ID")
    if set(test_ids) != set(sample["id"]):
        raise ValueError("Prediction IDs must match sample IDs exactly")
    submission = sample.copy()
    submission["label"] = pd.Series(values, index=test_ids).reindex(sample["id"]).to_numpy()
    if submission["label"].isna().any() or submission["label"].eq("").any():
        raise ValueError("Missing predictions")
    destination = output / "submission.csv"
    temporary = output / "submission.csv.tmp"
    submission.to_csv(temporary, index=False, encoding="utf-8")
    restored = pd.read_csv(temporary, converters={"id": str, "label": str})
    pd.testing.assert_frame_equal(restored, submission.astype(str), check_dtype=False)
    temporary.replace(destination)
    return destination


# %% 6. Wire the cells together; the plumbing is deliberately longer than the model choice.
def run(
    data_dir: Path,
    output: Path,
    *,
    task: str,
    metric: str,
    split: str = "stratified",
    split_column: str | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Fit one candidate, record its validation evidence, reload, and produce a checked CSV."""
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    train, test, sample = read_tables(data_dir)
    a, b = split_rows(train, split, split_column, seed)
    if not len(a) or not len(b) or set(train.iloc[a]["label"]) != set(train["label"]):
        raise ValueError("Empty split or a class absent from training; inspect the split")
    if task == "text":
        features = train["text"].fillna("")
        test_features = test["text"].fillna("")
    else:
        columns = [name for name in train if name not in {"id", "label", split_column}]
        features, test_features = train[columns], test[columns]
    model = make_model(task, features, seed=seed)
    model.fit(features.iloc[a], train.iloc[a]["label"])
    labels = sorted(train["label"].unique().tolist())
    validation_predictions = model.predict(features.iloc[b])
    value = score_labels(train.iloc[b]["label"], validation_predictions, metric, labels)
    temporary = output / "pipeline.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(model, handle)
    temporary.replace(output / "pipeline.pkl")
    # Load only your own trusted artifacts. This saves inference, not exact training continuation.
    with (output / "pipeline.pkl").open("rb") as handle:
        restored = pickle.load(handle)
    np.testing.assert_array_equal(restored.predict(features.iloc[b]), validation_predictions)
    predictions = restored.predict(test_features)
    if not set(predictions) <= set(labels):
        raise ValueError("Predictions contain an unknown label")
    deliver(sample, test["id"], predictions, output)
    report = {
        "task": task,
        "metric": metric,
        "validation_score": value,
        "seed": seed,
        "split": split,
        "split_column": split_column,
        "train_ids": train.iloc[a]["id"].tolist(),
        "validation_ids": train.iloc[b]["id"].tolist(),
        "labels": labels,
        "feature_columns": ["text"] if task == "text" else features.columns.tolist(),
        "model": str(model),
        "refit_all_train": False,
        "pretrained_allowed": False,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{metric}={value:.4f}; delivered to {output}")
    return report


def main() -> None:
    """Run the answer key locally; timed learners recreate the cells instead of invoking it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path, help="A new run directory")
    parser.add_argument("--task", choices=["text", "tabular"], required=True)
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], required=True)
    parser.add_argument("--split", choices=["stratified", "group", "time"], default="stratified")
    parser.add_argument("--split-column")
    arguments = parser.parse_args()
    run(
        arguments.data_dir,
        arguments.output,
        task=arguments.task,
        metric=arguments.metric,
        split=arguments.split,
        split_column=arguments.split_column,
    )


if __name__ == "__main__":
    main()
