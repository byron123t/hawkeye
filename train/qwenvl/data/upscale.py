"""
Bicubic upscaling of training images, matching Hawkeye's inference contract.

install_upscale(factor) wraps data_processor._build_messages so every image of the selected rows
is resized by `factor` per side before the processor (which still caps it at max_pixels). It
applies to the eval dataset too, so eval_loss is measured on the same inputs. Rows are selected
by their `source_dataset` field (prepare_data.py writes "res_video" / "res_photo").
"""
from __future__ import annotations

from PIL import Image

from qwenvl.data import data_processor as dp


def install_upscale(factor: float, sources: set[str]) -> None:
    if factor == 1.0 or not sources:
        return
    orig = dp._build_messages

    def wrapped(item, base_path):
        msgs = orig(item, base_path)
        if item.get("source_dataset") not in sources:
            return msgs
        for m in msgs:
            for c in m["content"]:
                if c.get("type") == "image" and isinstance(c.get("image"), str):
                    im = Image.open(c["image"]).convert("RGB")
                    c["image"] = im.resize((max(1, round(im.size[0] * factor)),
                                            max(1, round(im.size[1] * factor))), Image.BICUBIC)
        return msgs

    dp._build_messages = wrapped
