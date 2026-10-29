#!/usr/bin/env python3
"""
Run a local model (Hawkeye or a Qwen3-VL base) over a BSTRB split and write its predictions.

One process, one GPU. For several GPUs use run_sharded.py, which launches this script once
per GPU with --shard k --num_shards n and merges the outputs.

Output: <out>/predictions_<name>.jsonl (or .shard<k>.jsonl), one row per image:
    id, split, kind, unit, text_height_px, prompt, reference, prediction, model
The file is appended as it goes, so an interrupted run resumes where it stopped.

    python eval/run_local.py --model byron123t/Hawkeye-4B --name hawkeye-4b --out results/
    python eval/run_local.py --model Qwen/Qwen3-VL-4B-Instruct --name qwen3-vl-4b --upscale 1 --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkeye.data import load_split, resolve_root, unit_of  # noqa: E402
from hawkeye.inference import MAX_NEW_TOKENS, UPSCALE, load, transcribe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint")
    ap.add_argument("--name", required=True, help="short name used in output files")
    ap.add_argument("--data", default=None, help="BSTRB root (default: $BSTRB_ROOT or download)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--kinds", default="video_frame,photo", help="comma list of row kinds")
    ap.add_argument("--upscale", type=float, default=UPSCALE,
                    help="3 for Hawkeye (and 'base x3'); 1 for the plain base")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--attn", default="flash_attention_2", help="sdpa if flash-attn is not installed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load_split(resolve_root(a.data), a.split, set(a.kinds.split(",")))
    rows = rows[a.shard::a.num_shards]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{a.shard}" if a.num_shards > 1 else ""
    path = out / f"predictions_{a.name}{suffix}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path) if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[run_local] {a.name}: {len(todo)} to do ({len(done)} already in {path})", flush=True)
    if not todo:
        return

    model, processor = load(a.model, attn_implementation=a.attn)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for k, r in enumerate(todo, 1):
            try:
                pred = transcribe(model, processor, r["image"], r["prompt"],
                                  upscale_factor=a.upscale, max_new_tokens=a.max_new_tokens)
            except Exception as e:  # keep going; scored as an error row
                pred = f"[ERROR] {type(e).__name__}: {e}"
            f.write(json.dumps({"id": r["id"], "split": r["split"], "kind": r["kind"], "unit": unit_of(r),
                                "text_height_px": r.get("text_height_px"), "prompt": r["prompt"],
                                "reference": r["transcription"], "prediction": pred,
                                "model": a.name}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(todo)}  {(time.time() - t0) / k:.1f}s/img", flush=True)


if __name__ == "__main__":
    main()
