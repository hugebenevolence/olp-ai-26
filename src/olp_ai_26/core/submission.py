"""Build, validate, and atomically write competition submission tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SubmissionValidation:
    """Validation errors and non-fatal warnings for a candidate submission."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        """Raise one ``ValueError`` containing every validation error."""
        if self.errors:
            raise ValueError("Invalid submission: " + "; ".join(self.errors))


def build_submission(
    sample: pd.DataFrame,
    predictions: Mapping[str, Sequence[object]],
    *,
    prediction_ids: pd.DataFrame | Sequence[object] | None = None,
    id_columns: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Copy the sample and optionally align predictions using unique (possibly composite) IDs.

    Without IDs, predictions must already be in sample order. Prefer explicit IDs for inference.
    Alignment rejects missing, duplicate, extra, or absent keys rather than guessing row order.
    """
    submission = sample.copy()
    ids = [id_columns] if isinstance(id_columns, str) else list(id_columns or [])
    order = None
    if prediction_ids is not None:
        if not ids:
            raise ValueError("id_columns is required with prediction_ids")
        if isinstance(prediction_ids, pd.DataFrame):
            keys = prediction_ids[ids].reset_index(drop=True)
        elif len(ids) == 1:
            keys = pd.DataFrame({ids[0]: list(prediction_ids)})
        else:
            raise ValueError("Composite prediction IDs must be a DataFrame")
        for name, frame in (("sample", sample), ("predictions", keys)):
            if (
                frame[ids].isna().any().any()
                or frame[ids].eq("").any().any()
                or frame.duplicated(ids).any()
            ):
                raise ValueError(f"{name} IDs must be nonmissing and unique across {ids}")
        expected = pd.MultiIndex.from_frame(sample[ids])
        actual = pd.MultiIndex.from_frame(keys)
        order = actual.get_indexer(expected)
        if len(actual) != len(expected) or (order < 0).any():
            raise ValueError("Prediction IDs must match the sample IDs exactly")
    for column, values in predictions.items():
        if column not in submission.columns:
            raise KeyError(f"Prediction column not present in sample submission: {column}")
        if len(values) != len(submission):
            raise ValueError(
                f"Column {column} has {len(values)} values; expected {len(submission)}"
            )
        if column in ids:
            raise ValueError(f"Cannot overwrite ID column: {column}")
        values = list(values)
        submission[column] = values if order is None else [values[index] for index in order]
    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    id_columns: str | Iterable[str] | None = None,
    allowed_labels: Mapping[str, Iterable[object]] | None = None,
) -> SubmissionValidation:
    """Check schema, row count, ID order, missing values, and optional allowed labels."""
    errors: list[str] = []
    warnings: list[str] = []
    if submission.columns.duplicated().any() or sample.columns.duplicated().any():
        errors.append("Submission and sample column names must be unique")
        return SubmissionValidation(valid=False, errors=errors)
    if list(submission.columns) != list(sample.columns):
        errors.append(
            f"Columns/order differ: got {list(submission.columns)}, expected {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        errors.append(f"Row count differs: got {len(submission)}, expected {len(sample)}")
    if submission.isna().any().any():
        columns = submission.columns[submission.isna().any()].tolist()
        errors.append(f"Missing values in columns: {columns}")
    numeric = submission.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all():
        errors.append("Non-finite numeric values are not allowed")
    ids = [id_columns] if isinstance(id_columns, str) else list(id_columns or [])
    for column in ids:
        if column not in submission or column not in sample:
            errors.append(f"Missing ID column: {column}")
        elif submission[column].isna().any() or submission[column].eq("").any():
            errors.append(f"Missing ID values in column: {column}")
        elif len(submission) == len(sample) and not submission[column].equals(sample[column]):
            errors.append(f"ID values/order changed in column: {column}")
    if ids and all(column in submission for column in ids) and submission.duplicated(ids).any():
        warnings.append(f"Duplicate ID keys across columns: {ids}")
    for column, labels in (allowed_labels or {}).items():
        if column not in submission:
            errors.append(f"Missing allowed-label column: {column}")
        else:
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
    allowed_labels: Mapping[str, Iterable[object]] | None = None,
) -> Path:
    """Validate then atomically write a UTF-8 CSV submission."""
    if sample is not None:
        validation = validate_submission(
            submission, sample, id_columns=id_columns, allowed_labels=allowed_labels
        )
        validation.raise_for_errors()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        submission.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
