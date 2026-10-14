"""Conservative, configurable Unicode-aware text normalization."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def normalize_text(
    value: object,
    *,
    lowercase: bool = False,
    replace_urls: bool = False,
) -> str:
    """Conservative Unicode cleaning that preserves Vietnamese diacritics."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFC", html.unescape(str(value)))
    if replace_urls:
        text = _URL.sub("<URL>", text)
    text = _SPACE.sub(" ", text).strip()
    return text.casefold() if lowercase else text


def normalize_batch(values: list[object], **kwargs: object) -> list[str]:
    """Apply :func:`normalize_text` with shared options to a list of values."""
    return [normalize_text(value, **kwargs) for value in values]
