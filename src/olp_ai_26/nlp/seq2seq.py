"""Local-first sequence-to-sequence model construction, tokenization, and decoding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_seq2seq_model(
    model_name_or_path: Path | str,
    *,
    pretrained_allowed: bool = False,
    local_files_only: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a tokenizer and either permitted weights or a random model from local config."""
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    source = str(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
    if pretrained_allowed:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            source, local_files_only=local_files_only, **kwargs
        )
    else:
        config = AutoConfig.from_pretrained(source, local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_config(config, **kwargs)
    return tokenizer, model


def tokenize_seq2seq_batch(
    batch: Mapping[str, list[str]],
    tokenizer: Any,
    *,
    source_column: str,
    target_column: str,
    source_prefix: str = "",
    max_source_length: int = 256,
    max_target_length: int = 256,
) -> dict[str, Any]:
    """Tokenize source and target text columns with independent maximum lengths."""
    inputs = [source_prefix + str(value) for value in batch[source_column]]
    targets = [str(value) for value in batch[target_column]]
    encoded = tokenizer(inputs, max_length=max_source_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    encoded["labels"] = labels["input_ids"]
    return encoded


def decode_generated(tokenizer: Any, token_ids: Any) -> list[str]:
    """Decode generated token IDs while removing special tokens and spacing artifacts."""
    return tokenizer.batch_decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def generate_texts(
    model: Any,
    tokenizer: Any,
    sources: list[str],
    *,
    device: str,
    source_prefix: str = "",
    max_source_length: int = 256,
    max_target_length: int = 128,
    num_beams: int = 1,
    batch_size: int = 8,
) -> list[str]:
    """Generate unlabeled text directly, avoiding trainer forward passes that require labels.

    Match validation's prefix, truncation, beam count, and total target-length limit exactly.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.to(device).eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(sources), batch_size):
            encoded = tokenizer(
                [source_prefix + str(source) for source in sources[start : start + batch_size]],
                truncation=True,
                padding=True,
                max_length=max_source_length,
                return_tensors="pt",
            )
            generated = model.generate(
                **{key: value.to(device) for key, value in encoded.items()},
                max_length=max_target_length,
                max_new_tokens=None,
                num_beams=num_beams,
            )
            result.extend(decode_generated(tokenizer, generated))
    return result
