"""Command line: transcribe one or more images.

    hawkeye-read photo.jpg                       # Hawkeye 4B, x3 upscale
    hawkeye-read a.jpg b.jpg --model 8b          # Hawkeye 8B
    hawkeye-read photo.jpg --model Qwen/Qwen3-VL-4B-Instruct --upscale 1   # plain base

Defaults follow the paper's evaluation protocol (see hawkeye/inference.py): JPEG re-encode, x3
bicubic upscale, transcription-only system prompt, instruction text before the image, greedy
decoding up to 4,096 tokens.
"""
from __future__ import annotations

import argparse
import sys

from .inference import DEFAULT_PROMPT, MAX_NEW_TOKENS, MODELS, UPSCALE
from .protocol import SYSTEM_PROMPT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hawkeye-read", description="Transcribe text in images with Hawkeye.")
    ap.add_argument("images", nargs="+", help="image file(s)")
    ap.add_argument("--model", default="4b",
                    help="'4b', '8b', a Hugging Face id, or a local checkpoint (default: 4b)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--upscale", type=float, default=UPSCALE,
                    help="bicubic per-side factor before the processor (default 3; Hawkeye was trained at 3)")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--no_system_prompt", action="store_true",
                    help="send the user turn only (default: the paper's transcription-only system prompt)")
    ap.add_argument("--device", default="auto", help="device_map for loading (default: auto)")
    ap.add_argument("--attn", default="sdpa", help="attention implementation (sdpa | flash_attention_2 | eager)")
    a = ap.parse_args(argv)

    from .inference import load, transcribe
    model, processor = load(MODELS.get(a.model.lower(), a.model), device=a.device, attn_implementation=a.attn)
    for path in a.images:
        text = transcribe(model, processor, path, a.prompt, upscale_factor=a.upscale,
                          max_new_tokens=a.max_new_tokens,
                          system_prompt=None if a.no_system_prompt else SYSTEM_PROMPT)
        if len(a.images) > 1:
            print(f"==> {path}")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
