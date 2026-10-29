"""
Inference with Hawkeye (and, for comparison, its Qwen3-VL base).

The protocol behind every held-out BSTRB number in the paper, identical for Hawkeye and base
models:

  1. image: RGB, longest side scaled down to <= 2,048 px (LANCZOS) if larger, re-encoded as JPEG
     quality 92 and decoded back (hawkeye.protocol.encode_image; API models get the same bytes);
  2. Hawkeye only: upscale x3 per side with bicubic interpolation (plain interpolation, no
     super-resolution model). Base models: x1 in the paper's table ("base x3" also reported);
  3. the processor keeps the image within [65,536, 1,003,520] pixels (min_pixels / max_pixels);
  4. messages: a system turn with the fixed transcription-only prompt
     (hawkeye.protocol.SYSTEM_PROMPT), then a user turn with the instruction TEXT FIRST and the
     IMAGE SECOND;
  5. greedy decoding (do_sample=False), up to 4,096 new tokens; sdpa attention, bfloat16.

`system_prompt=None` (CLI `--no_system_prompt`) sends the user turn only.

    from hawkeye import Hawkeye
    reader = Hawkeye("byron123t/Hawkeye-4B")
    print(reader.read("photo.jpg"))
"""
from __future__ import annotations

from typing import Optional, Union

from PIL import Image

from .protocol import SYSTEM_PROMPT, protocol_image

UPSCALE = 3.0
MIN_PIXELS = 65_536
MAX_PIXELS = 1_003_520
MAX_NEW_TOKENS = 4096
DEFAULT_PROMPT = "Provide only the text content of this image, line by line as it appears."
MODELS = {"4b": "byron123t/Hawkeye-4B", "8b": "byron123t/Hawkeye-8B"}


def upscale(image: Image.Image, factor: float = UPSCALE) -> Image.Image:
    """Bicubic resize by `factor` per side (no-op for 1.0)."""
    if factor == 1.0:
        return image
    w, h = image.size
    return image.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BICUBIC)


def pin_pixels(processor, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> None:
    """Set the processor's pixel budget in place (attribute mutation, which both Qwen2.5-VL and
    Qwen3-VL processors honour; a `size=` kwarg is ignored by Qwen2.5-VL)."""
    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
    if isinstance(getattr(ip, "size", None), dict):
        ip.size["shortest_edge"] = min_pixels
        ip.size["longest_edge"] = max_pixels


def load(model_id: str = MODELS["4b"], *, device: str = "auto", dtype: str = "bfloat16",
         attn_implementation: str = "sdpa", processor_id: Optional[str] = None,
         min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS):
    """Load a Hawkeye or Qwen3-VL checkpoint (HF id or local dir) and its pinned processor."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    cls = getattr(transformers, AutoConfig.from_pretrained(model_id).architectures[0])
    model = cls.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                attn_implementation=attn_implementation, device_map=device).eval()
    processor = AutoProcessor.from_pretrained(processor_id or model_id)
    pin_pixels(processor, min_pixels, max_pixels)
    return model, processor


def build_messages(image: Image.Image, prompt: str, system_prompt: Optional[str] = SYSTEM_PROMPT) -> list:
    """System turn (unless None), then the user turn: instruction text first, then the image."""
    user = {"role": "user", "content": [{"type": "text", "text": prompt},
                                        {"type": "image", "image": image}]}
    if system_prompt is None:
        return [user]
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, user]


def prepare_image(image: Union[str, Image.Image], upscale_factor: float = UPSCALE) -> Image.Image:
    """Protocol steps 1-2: JPEG re-encode (longest side <= 2048), then the bicubic upscale."""
    return upscale(protocol_image(image), upscale_factor)


def transcribe(model, processor, image: Union[str, Image.Image], prompt: str = DEFAULT_PROMPT, *,
               system_prompt: Optional[str] = SYSTEM_PROMPT, upscale_factor: float = UPSCALE,
               max_new_tokens: int = MAX_NEW_TOKENS, return_info: bool = False):
    """One image -> one transcription (greedy). With return_info, also returns
    {"output_tokens", "finish_reason"} ("stop" if <|im_end|> was produced, else "length")."""
    import torch

    prompt = prompt.replace("<image>", "").strip()
    img = prepare_image(image, upscale_factor)
    text = processor.apply_chat_template(build_messages(img, prompt, system_prompt),
                                         tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    gen = out[0, inputs["input_ids"].shape[1]:]
    pred = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
    if not return_info:
        return pred
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    return pred, {"output_tokens": int(gen.shape[0]),
                  "finish_reason": "stop" if bool((gen == eos).any()) else "length"}


class Hawkeye:
    """Convenience wrapper: `Hawkeye("byron123t/Hawkeye-4B").read("img.jpg")`."""

    def __init__(self, model_id: str = MODELS["4b"], upscale_factor: float = UPSCALE,
                 system_prompt: Optional[str] = SYSTEM_PROMPT, **load_kw):
        self.model, self.processor = load(MODELS.get(model_id.lower(), model_id), **load_kw)
        self.upscale_factor = upscale_factor
        self.system_prompt = system_prompt

    def read(self, image: Union[str, Image.Image], prompt: str = DEFAULT_PROMPT,
             max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        return transcribe(self.model, self.processor, image, prompt, system_prompt=self.system_prompt,
                          upscale_factor=self.upscale_factor, max_new_tokens=max_new_tokens)


__all__ = ["Hawkeye", "load", "transcribe", "build_messages", "prepare_image", "upscale", "pin_pixels",
           "SYSTEM_PROMPT", "UPSCALE", "MIN_PIXELS", "MAX_PIXELS", "MAX_NEW_TOKENS", "DEFAULT_PROMPT", "MODELS"]
