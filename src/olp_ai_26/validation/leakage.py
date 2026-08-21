"""Text and file-identity checks for duplicates and cross-split leakage."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def normalize_text(value: object) -> str:
    """Normalize Unicode, whitespace, and case for duplicate comparison only."""
    text = unicodedata.normalize("NFC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def duplicate_text_groups(frame: pd.DataFrame, column: str) -> list[list[int]]:
    """Return index groups whose normalized text occurs more than once."""
    if column not in frame:
        raise KeyError(column)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in frame[column].items():
        digest = hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()
        groups[digest].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def cross_split_overlap(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: str | Iterable[str],
) -> pd.DataFrame:
    """Return validation rows whose normalized selected-column key also occurs in training."""
    selected = [columns] if isinstance(columns, str) else list(columns)
    for column in selected:
        if column not in train or column not in valid:
            raise KeyError(column)
    train_keys = train[selected].astype(str).agg("|".join, axis=1).map(normalize_text)
    valid_keys = valid[selected].astype(str).agg("|".join, axis=1).map(normalize_text)
    overlap = set(train_keys) & set(valid_keys)
    return valid.loc[valid_keys.isin(overlap)].copy()


def file_hashes(paths: Iterable[Path | str]) -> dict[str, list[str]]:
    """Group input paths by byte-level SHA-256 digest."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item in paths:
        path = Path(item)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        groups[digest.hexdigest()].append(str(path))
    return dict(groups)
