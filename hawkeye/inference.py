"""
Inference with Hawkeye (and, for comparison, its Qwen3-VL base).

The input contract used for every Hawkeye number in the paper:

  1. upscale the image x3 per side with bicubic interpolation (plain interpolation, no
     super-resolution model),
  2. let the processor bring it within [65,536, 1,003,520] pixels (min_pixels / max_pixels),
  3. one user turn (image, then the prompt), no system prompt,
  4. greedy decoding, up to 2,048 new tokens.

Base models are evaluated with the same pixel limits, either on the plain image (`upscale=1`)
or at the same x3 ("base x3").

    from hawkeye import Hawkeye
    reader = Hawkeye("byron123t/Hawkeye-4B")
    print(reader.read("photo.jpg"))
"""
from __future__ import annotations

from typing import Optional, Union

from PIL import Image

UPSCALE = 3.0
MIN_PIXELS = 65_536
MAX_PIXELS = 1_003_520
MAX_NEW_TOKENS = 2048
DEFAULT_PROMPT = "Provide only the text content of this image, line by line as it appears."
MODELS = {"4b": "byron123t/Hawkeye-4B", "8b": "byron123t/Hawkeye-8B"}


def upscale(image: Image.Image, factor: float = UPSCALE) -> Image.Image:
    """Bicubic resize by `factor` per side (no-op for 1.0)."""
    if factor == 1.0:
        return image
    w, h = image.size
    return image.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BICUBIC)


def pin_pixels(processor, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> None:
    """Set the processor's pixel budget in place. Attribute mutation (rather than a `size=`
    kwarg) is what both Qwen2.5-VL and Qwen3-VL processors honour."""
    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
    if isinstance(getattr(ip, "size", None), dict):
        ip.size["shortest_edge"] = min_pixels
        ip.size["longest_edge"] = max_pixels


def load(model_id: str = MODELS["4b"], *, device: str = "cuda", dtype: str = "bfloat16",
         attn_implementation: str = "sdpa", min_pixels: int = MIN_PIXELS,
         max_pixels: int = MAX_PIXELS):
    """Load a Hawkeye or Qwen3-VL checkpoint (HF id or local dir) and its pinned processor."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    arch = AutoConfig.from_pretrained(model_id).architectures[0]
    cls = getattr(transformers, arch)
    model = cls.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                attn_implementation=attn_implementation, device_map=device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    pin_pixels(processor, min_pixels, max_pixels)
    return model, processor


def transcribe(model, processor, image: Union[str, Image.Image], prompt: str = DEFAULT_PROMPT, *,
               upscale_factor: float = UPSCALE, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """One image -> one transcription (greedy)."""
    import torch

    if isinstance(image, str):
        image = Image.open(image)
    image = upscale(image.convert("RGB"), upscale_factor)
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    return processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()


class Hawkeye:
    """Convenience wrapper: `Hawkeye("byron123t/Hawkeye-4B").read("img.jpg")`."""

    def __init__(self, model_id: str = MODELS["4b"], upscale_factor: float = UPSCALE, **load_kw):
        self.model, self.processor = load(MODELS.get(model_id.lower(), model_id), **load_kw)
        self.upscale_factor = upscale_factor

    def read(self, image: Union[str, Image.Image], prompt: str = DEFAULT_PROMPT,
             max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        return transcribe(self.model, self.processor, image, prompt,
                          upscale_factor=self.upscale_factor, max_new_tokens=max_new_tokens)


__all__ = ["Hawkeye", "load", "transcribe", "upscale", "pin_pixels", "UPSCALE", "MIN_PIXELS",
           "MAX_PIXELS", "MAX_NEW_TOKENS", "DEFAULT_PROMPT", "MODELS"]
