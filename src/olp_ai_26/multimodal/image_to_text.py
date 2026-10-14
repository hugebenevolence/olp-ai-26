"""Local image-captioning dataset, model factory, and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CaptionProcessor:
    """Pair a standalone image processor with the decoder tokenizer for local caption models."""

    image_processor: Any
    tokenizer: Any

    def __call__(self, **kwargs: Any) -> Any:
        """Delegate pixel preprocessing to the image processor."""
        return self.image_processor(**kwargs)

    def save_pretrained(self, directory: Path | str) -> None:
        """Save both pixel preprocessing and text tokenizer files for cold reload."""
        self.image_processor.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)


class ImageCaptionDataset(Dataset[dict[str, torch.Tensor]]):
    """Preprocess image/caption table rows for a Hugging Face vision-language model."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        processor: Any,
        image_column: str,
        caption_column: str,
        root: Path | str = ".",
        max_length: int = 64,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.image_column = image_column
        self.caption_column = caption_column
        self.root = Path(root)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(self.root / str(row[self.image_column])) as source:
            image = source.convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        labels = tokenizer(
            str(row[self.caption_column]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels = labels.masked_fill(labels == pad_token_id, -100)
        return {"pixel_values": pixel_values, "labels": labels}


def build_image_captioner(
    model_name_or_path: Path | str,
    *,
    pretrained_allowed: bool = False,
    local_files_only: bool = True,
) -> tuple[Any, Any]:
    """Build a local Hugging Face image-to-text model under an explicit weight policy."""
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
    )

    source = str(model_name_or_path)
    processor = CaptionProcessor(
        AutoImageProcessor.from_pretrained(source, local_files_only=local_files_only),
        AutoTokenizer.from_pretrained(source, local_files_only=local_files_only),
    )
    if pretrained_allowed:
        model = VisionEncoderDecoderModel.from_pretrained(source, local_files_only=local_files_only)
    else:
        config = VisionEncoderDecoderConfig.from_pretrained(
            source, local_files_only=local_files_only
        )
        model = VisionEncoderDecoderModel(config=config)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Caption decoder requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    start = model.config.decoder_start_token_id
    if start is None:
        start = (
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        )
    if start is None:
        raise ValueError(
            "Set the caption model's decoder_start_token_id in its local configuration"
        )
    model.config.decoder_start_token_id = start
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = start
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return processor, model


@torch.inference_mode()
def generate_captions(
    model: Any,
    processor: Any,
    images: list[Any],
    *,
    device: str,
    max_new_tokens: int = 64,
    num_beams: int = 4,
    max_length: int | None = None,
) -> list[str]:
    """Generate and decode captions for batches of preprocessed pixel values."""
    model.to(device).eval()
    encoded = processor(images=images, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)
    length_options = (
        {"max_length": max_length, "max_new_tokens": None}
        if max_length is not None
        else {"max_new_tokens": max_new_tokens}
    )
    generated = model.generate(pixel_values=pixel_values, num_beams=num_beams, **length_options)
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)
