from __future__ import annotations

import pandas as pd
import torch
from PIL import Image

from olp_ai_26.multimodal.image_to_text import ImageCaptionDataset


class DummyProcessor:
    pad_token_id = 0

    def __call__(self, value=None, *, images=None, return_tensors=None, **kwargs):
        del return_tensors, kwargs
        if images is not None:
            return {"pixel_values": torch.ones(1, 3, 8, 8)}
        assert value == "một con mèo"
        return {"input_ids": torch.tensor([[2, 3, 0, 0]])}


def test_image_caption_dataset_masks_padding(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")
    frame = pd.DataFrame({"image": ["image.png"], "caption": ["một con mèo"]})
    dataset = ImageCaptionDataset(
        frame,
        processor=DummyProcessor(),
        image_column="image",
        caption_column="caption",
        root=tmp_path,
    )
    item = dataset[0]
    assert item["pixel_values"].shape == (3, 8, 8)
    assert item["labels"].tolist() == [2, 3, -100, -100]
