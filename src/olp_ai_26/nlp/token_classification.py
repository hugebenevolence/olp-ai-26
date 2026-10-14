"""Local-first token-classifier construction and word-to-subword label alignment."""

from __future__ import annotations

from collections.abc import Sequence
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
    max_length: int = 256,
) -> dict[str, Any]:
    """Tokenize split words and align word labels with generated subword positions."""
    encoded = tokenizer(
        batch[token_column], truncation=True, is_split_into_words=True, max_length=max_length
    )
    aligned: list[list[int]] = []
    for batch_index, labels in enumerate(batch[label_column]):
        word_ids = encoded.word_ids(batch_index=batch_index)
        covered = {index for index in word_ids if index is not None}
        if len(covered) != len(labels):
            raise ValueError(
                "Training tokens were truncated; increase max_length or chunk labeled sentences"
            )
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


def predict_word_tags(
    model: Any,
    tokenizer: Any,
    tokens: list[list[str]],
    *,
    device: str = "cpu",
    max_length: int = 256,
    stride: int = 32,
    batch_size: int = 16,
    label_names: Sequence[str] | None = None,
) -> list[list[Any]]:
    """Predict one tag per original word using fast-tokenizer overflow windows.

    Take the first complete first-subword prediction for each word. Fail if any word cannot be
    aligned; do not silently replace truncated words with the outside label.
    """
    import torch

    if not tokens:
        return []
    if batch_size < 1 or stride < 0 or stride >= max_length - 2:
        raise ValueError("Invalid inference batch size or overflow stride")
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    sources = encoded.pop("overflow_to_sample_mapping").tolist()
    offsets = encoded.pop("offset_mapping")
    word_ids = [encoded.word_ids(batch_index=i) for i in range(len(sources))]
    result: list[list[Any]] = [[None] * len(row) for row in tokens]
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(sources), batch_size):
            batch = {
                key: value[start : start + batch_size].to(device) for key, value in encoded.items()
            }
            predictions = model(**batch).logits.argmax(-1).cpu().tolist()
            for offset, row in enumerate(predictions):
                window = start + offset
                source = sources[window]
                for position, label in enumerate(row):
                    word = word_ids[window][position]
                    if (
                        word is not None
                        and offsets[window, position, 0].item() == 0
                        and result[source][word] is None
                    ):
                        result[source][word] = label_names[label] if label_names else label
    if any(label is None for row in result for label in row):
        raise ValueError(
            "Tokenizer did not align every word; check fast-tokenizer support and empty words"
        )
    return result
