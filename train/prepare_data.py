#!/usr/bin/env python3
"""
Convert BSTRB (Hugging Face layout) into the LLaVA-style JSON that qwen-vl-finetune and the GRPO
script read:

    {"image": "/abs/path.jpg",
     "conversations": [{"from": "human", "value": "<image>\\n<prompt>"},
                       {"from": "gpt", "value": "<transcription>"}],
     "source_dataset": "res_video" | "res_photo", "id": ..., "video_id": ...}

Hawkeye 4B/8B were trained on the video frames only (6,734 frames, labels = GPT-5
transcriptions) and selected on validation loss over the validation split (human labels).

    python train/prepare_data.py --data /path/to/BSTRB --out data/
      -> data/bstrb_train_video.json, data/bstrb_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkeye.data import load_split, resolve_root  # noqa: E402

SOURCE = {"video_frame": "res_video", "photo": "res_photo"}


def convert(rows):
    return [{"image": r["image"],
             "conversations": [{"from": "human", "value": "<image>\n" + r["prompt"]},
                               {"from": "gpt", "value": r["transcription"]}],
             "source_dataset": SOURCE[r["kind"]], "id": r["id"], "video_id": r.get("video_id")}
            for r in rows if r["transcription"].strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="BSTRB root (default: $BSTRB_ROOT or download)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train_kinds", default="video_frame",
                    help="'video_frame' reproduces the paper; 'video_frame,photo' adds the photos")
    a = ap.parse_args()
    root = resolve_root(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = convert(load_split(root, "train", set(a.train_kinds.split(","))))
    val = convert(load_split(root, "val"))
    name = "bstrb_train_video.json" if a.train_kinds == "video_frame" else "bstrb_train.json"
    (out / name).write_text(json.dumps(train, ensure_ascii=False))
    (out / "bstrb_val.json").write_text(json.dumps(val, ensure_ascii=False))
    print(f"train: {len(train)} rows -> {out / name}\nval:   {len(val)} rows -> {out / 'bstrb_val.json'}")


if __name__ == "__main__":
    main()
