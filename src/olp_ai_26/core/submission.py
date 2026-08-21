from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class SubmissionValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Invalid submission: " + "; ".join(self.errors))


def build_submission(
    sample: pd.DataFrame,
    predictions: Mapping[str, Sequence[object]],
) -> pd.DataFrame:
    submission = sample.copy()
    for column, values in predictions.items():
        if column not in submission.columns:
            raise KeyError(f"Prediction column not present in sample submission: {column}")
        if len(values) != len(submission):
            raise ValueError(
                f"Column {column} has {len(values)} values; expected {len(submission)}"
            )
        submission[column] = values
    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    id_columns: str | Iterable[str] | None = None,
    allowed_labels: Mapping[str, Iterable[object]] | None = None,
) -> SubmissionValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if list(submission.columns) != list(sample.columns):
        errors.append(
            f"Columns/order differ: got {list(submission.columns)}, expected {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        errors.append(f"Row count differs: got {len(submission)}, expected {len(sample)}")
    if submission.isna().any().any():
        columns = submission.columns[submission.isna().any()].tolist()
        errors.append(f"Missing values in columns: {columns}")
    ids = [id_columns] if isinstance(id_columns, str) else list(id_columns or [])
    for column in ids:
        if column not in submission or column not in sample:
            errors.append(f"Missing ID column: {column}")
        elif len(submission) == len(sample) and not submission[column].equals(sample[column]):
            errors.append(f"ID values/order changed in column: {column}")
        elif submission[column].duplicated().any():
            warnings.append(f"Duplicate IDs in column: {column}")
    for column, labels in (allowed_labels or {}).items():
        if column in submission:
            invalid = set(submission[column].dropna().unique()) - set(labels)
            if invalid:
                errors.append(f"Unexpected labels in {column}: {sorted(map(str, invalid))[:10]}")
    return SubmissionValidation(valid=not errors, errors=errors, warnings=warnings)


def write_submission(
    submission: pd.DataFrame,
    path: Path | str,
    *,
    sample: pd.DataFrame | None = None,
    id_columns: str | Iterable[str] | None = None,
) -> Path:
    if sample is not None:
        validation = validate_submission(submission, sample, id_columns=id_columns)
        validation.raise_for_errors()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False, encoding="utf-8")
    return destination
