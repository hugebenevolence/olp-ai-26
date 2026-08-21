"""Conservative, configurable Unicode-aware text normalization."""

from __future__ import annotations

import html
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def normalize_text(
    value: object,
    *,
    lowercase: bool = False,
    replace_urls: bool = False,
) -> str:
    """Conservative Unicode cleaning that preserves Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", html.unescape(str(value)))
    if replace_urls:
        text = _URL.sub("<URL>", text)
    text = _SPACE.sub(" ", text).strip()
    return text.casefold() if lowercase else text


def normalize_batch(values: list[object], **kwargs: object) -> list[str]:
    """Apply :func:`normalize_text` with shared options to a list of values."""
    return [normalize_text(value, **kwargs) for value in values]
