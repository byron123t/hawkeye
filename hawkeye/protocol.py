"""
The evaluation protocol shared by every model in the paper (local and API).

SYSTEM_PROMPT  the fixed transcription-only system prompt (id "ocr_transcription_only_v1"),
               sent as a system turn to every model.
encode_image   the image every model receives: RGB, longest side scaled down to <= 2048 px
               (LANCZOS) if larger, re-encoded as JPEG (quality 92). API models get these bytes;
               local models decode them back to RGB, then (Hawkeye) upscale x3.
"""
from __future__ import annotations

import io

from PIL import Image

SYSTEM_PROMPT_ID = "ocr_transcription_only_v1"
SYSTEM_PROMPT = (
    "You are an OCR engine. Transcribe the text visible in the image exactly as it "
    "appears.\n"
    "Output rules:\n"
    "- Output ONLY the transcribed text. No preamble, no explanation, no commentary, "
    "no closing remarks.\n"
    "- Do not describe the image. Do not say what the text is or where it is.\n"
    "- Preserve the original line breaks: one visual line of text per output line.\n"
    "- Do not wrap the output in quotes or code fences, and do not add Markdown.\n"
    "- Do not correct spelling or expand abbreviations; reproduce what is printed.\n"
    "- If no text is legible, output nothing."
)
MAX_SIDE = 2048
JPEG_QUALITY = 92


def encode_image(image, max_side: int = MAX_SIDE) -> bytes:
    """JPEG bytes of `image` (a path or PIL image): RGB, downscaled only if a side exceeds
    max_side (most BSTRB crops don't), quality 92."""
    im = image if isinstance(image, Image.Image) else Image.open(image)
    im = im.convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def protocol_image(image) -> Image.Image:
    """The decoded RGB image a local model sees before any upscaling."""
    return Image.open(io.BytesIO(encode_image(image))).convert("RGB")


__all__ = ["SYSTEM_PROMPT", "SYSTEM_PROMPT_ID", "encode_image", "protocol_image", "MAX_SIDE", "JPEG_QUALITY"]
