"""Local-first token-classifier construction and word-to-subword label alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_token_classifier(
    model_name_or_path: Path | str,
    *,
    num_labels: int,
    pretrained_allowed: bool = False,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Build a fast tokenizer and token classifier without implicit weight downloads."""
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    source = str(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        source, use_fast=True, local_files_only=local_files_only
    )
    if pretrained_allowed:
        model = AutoModelForTokenClassification.from_pretrained(
            source, num_labels=num_labels, local_files_only=local_files_only
        )
    else:
        config = AutoConfig.from_pretrained(source, local_files_only=local_files_only)
        config.num_labels = num_labels
        model = AutoModelForTokenClassification.from_config(config)
    return tokenizer, model


def tokenize_and_align_labels(
    batch: dict[str, list[Any]],
    tokenizer: Any,
    *,
    token_column: str = "tokens",
    label_column: str = "ner_tags",
    label_all_tokens: bool = False,
) -> dict[str, Any]:
    """Tokenize split words and align word labels with generated subword positions."""
    encoded = tokenizer(batch[token_column], truncation=True, is_split_into_words=True)
    aligned: list[list[int]] = []
    for batch_index, labels in enumerate(batch[label_column]):
        word_ids = encoded.word_ids(batch_index=batch_index)
        previous = None
        row: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                row.append(-100)
            elif word_id != previous:
                row.append(labels[word_id])
            else:
                row.append(labels[word_id] if label_all_tokens else -100)
            previous = word_id
        aligned.append(row)
    encoded["labels"] = aligned
    return encoded
