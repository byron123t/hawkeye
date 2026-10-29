#!/usr/bin/env python3
"""
Per-generated-token attention maps for one model ("arm") on a set of images.

For every token the model generates (greedy), the attention from that token's position to the
image-token positions is taken from the last 4 decoder layers, averaged over heads and layers,
and reshaped onto the image's merged patch grid (one cell per visual token). The sequence is
recomputed at every step with eager attention so each map is that token's own attention row.

Both arms see the same input: the image upscaled x3 (bicubic) and capped at 1,003,520 pixels,
i.e. Hawkeye's operating point, so base and Hawkeye are compared at matched resolution.

Output layout (consumed by the other analysis scripts):
    <out>/<sample_id>/<arm>/metadata.json          sample, prediction, per-token records
    <out>/<sample_id>/<arm>/heatmaps/step_0000.npy  raw (H', W') attention map per token

    python analysis/extract_attention.py --model Qwen/Qwen3-VL-4B-Instruct --arm base \
        --split test --kinds video_frame --sample 160 --seed 13 --out attn/4b
    python analysis/extract_attention.py --model byron123t/Hawkeye-4B --arm hawkeye \
        --split test --kinds video_frame --sample 160 --seed 13 --out attn/4b
    python analysis/extract_attention.py --model byron123t/Hawkeye-4B --arm hawkeye \
        --images photo.jpg --prompt "What text is written in the image that you can read?" --out attn/photos

Needs a GPU; eager attention is slow (about 2 min per image for 256 tokens on a 48 GB card).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkeye.inference import MAX_PIXELS, MIN_PIXELS, UPSCALE, pin_pixels, upscale  # noqa: E402

LAST_LAYERS = 4


def image_positions(input_ids, model, tokenizer) -> list[int]:
    tid = getattr(model.config, "image_token_id", None)
    if tid is None:
        tid = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return [i for i, t in enumerate(input_ids) if t == tid]


def token_map(attentions, query_pos: int, img_pos: list[int], grid_thw, merge: int) -> np.ndarray:
    """Mean over heads, then over the last LAST_LAYERS layers -> (H', W') grid."""
    vec = None
    for layer in attentions[-LAST_LAYERS:]:
        row = layer[0].float().mean(0)[query_pos, img_pos]          # heads averaged
        vec = row if vec is None else vec + row
    vec = (vec / LAST_LAYERS).cpu().numpy()
    t, h, w = (int(x) for x in grid_thw[0].tolist())
    h, w = h // merge, w // merge
    if t * h * w != vec.shape[0]:                                   # defensive; should not happen
        fixed = np.zeros(t * h * w, np.float32)
        fixed[:min(len(vec), len(fixed))] = vec[:len(fixed)]
        vec = fixed
    return vec.reshape(t, h, w).mean(0).astype(np.float32)


def run_one(model, processor, image_path: str, prompt: str, out_dir: Path, max_new_tokens: int,
            upscale_factor: float, sample_meta: dict) -> dict:
    import torch
    image = upscale(Image.open(image_path).convert("RGB"), upscale_factor)
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = dict(processor(text=[rendered], images=[image], padding=True, return_tensors="pt"))
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    img_pos = image_positions(inputs["input_ids"][0].tolist(), model, processor.tokenizer)
    merge = int(getattr(processor.image_processor, "merge_size", 2) or 2)
    eos = processor.tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    hm_dir = out_dir / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)

    tokens, generated = [], []
    with torch.no_grad():
        nxt = model(**inputs, use_cache=False).logits[:, -1].argmax(-1)
        for step in range(max_new_tokens):
            tok = int(nxt.item())
            if tok in eos:
                break
            # append the token and recompute; its own position is the query
            inputs["input_ids"] = torch.cat([inputs["input_ids"], nxt.view(1, 1)], 1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 1))], 1)
            for k in ("mm_token_type_ids", "token_type_ids"):
                if k in inputs:
                    inputs[k] = torch.cat([inputs[k], inputs[k].new_zeros((1, 1))], 1)
            out = model(**inputs, output_attentions=True, use_cache=False)
            q = inputs["input_ids"].shape[1] - 1
            hm = token_map(out.attentions, q, img_pos, inputs["image_grid_thw"], merge)
            path = hm_dir / f"step_{step:04d}.npy"
            np.save(path, hm)
            generated.append(tok)
            tokens.append({"step": step, "token_id": tok,
                           "token_text": processor.tokenizer.decode([tok], skip_special_tokens=False),
                           "heatmap": str(path.resolve()), "grid": list(hm.shape)})
            nxt = out.logits[:, -1].argmax(-1)
            del out
            torch.cuda.empty_cache()
    md = {"sample": sample_meta, "prompt": prompt, "upscale": upscale_factor,
          "prediction": processor.tokenizer.decode(generated, skip_special_tokens=True).strip(),
          "layers": f"last{LAST_LAYERS}", "heads": "mean", "tokens": tokens}
    (out_dir / "metadata.json").write_text(json.dumps(md, ensure_ascii=False, indent=1))
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, help="label for this model, e.g. base or hawkeye")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="BSTRB root (with --split)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--kinds", default="video_frame")
    ap.add_argument("--sample", type=int, default=0, help="random subset of this size (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--images", nargs="*", default=None, help="image files instead of BSTRB")
    ap.add_argument("--prompt", default="What text is written in the image that you can read?",
                    help="prompt for --images (BSTRB rows use their own)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--upscale", type=float, default=UPSCALE)
    a = ap.parse_args()

    if a.images:
        items = [{"id": Path(p).stem, "image": str(Path(p).resolve()), "prompt": a.prompt} for p in a.images]
    else:
        from hawkeye.data import load_split, resolve_root
        items = load_split(resolve_root(a.data), a.split, set(a.kinds.split(",")))
        if a.sample:
            items = random.Random(a.seed).sample(items, min(a.sample, len(items)))

    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    cls = getattr(transformers, AutoConfig.from_pretrained(a.model).architectures[0])
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager",
                                device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(a.model)
    pin_pixels(processor, MIN_PIXELS, MAX_PIXELS)

    for k, it in enumerate(items, 1):
        od = Path(a.out) / it["id"] / a.arm
        if (od / "metadata.json").exists():
            continue
        meta = {k2: it.get(k2) for k2 in ("id", "image", "kind", "video_id", "text_height_px")}
        meta["ground_truth"] = it.get("transcription")
        md = run_one(model, processor, it["image"], it["prompt"], od, a.max_new_tokens, a.upscale, meta)
        print(f"[{a.arm}] {k}/{len(items)} {it['id']}: {len(md['tokens'])} tokens", flush=True)


if __name__ == "__main__":
    main()
