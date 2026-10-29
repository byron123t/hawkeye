#!/usr/bin/env python3
"""
Text regions for the reading-trace metrics: EasyOCR's CRAFT detector only (no recognition), run
on the image as the model sees it (bicubic x3), CPU. Boxes are returned in original-image pixels.
On two hand-annotated captures, 99% and 90% of the detected boxes fall inside the hand regions.

    python analysis/text_regions.py --runs attn/4b --out text_regions.json
    python analysis/text_regions.py --images a.jpg b.jpg --out text_regions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import ARMS

UP = 3


def images_from_runs(roots) -> list[str]:
    out = []
    for r in roots:
        for md in Path(r).glob(f"*/{ARMS[0]}/metadata.json"):
            out.append(json.load(open(md))["sample"]["image"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="extract_attention.py output dirs")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    imgs = sorted(set(images_from_runs(a.runs) + a.images))
    out = Path(a.out)
    done = json.load(open(out)) if out.exists() else {}
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, recognizer=False, verbose=False)
    for k, p in enumerate(imgs):
        if p in done:
            continue
        im = Image.open(p).convert("RGB")
        W, H = im.size
        big = np.asarray(im.resize((W * UP, H * UP), Image.BICUBIC))
        horiz, free = reader.detect(big, canvas_size=2560, mag_ratio=1.0)
        boxes = [[x0 / UP, y0 / UP, x1 / UP, y1 / UP] for x0, x1, y0, y1 in horiz[0]]
        for poly in free[0]:                      # rotated text: axis-aligned bound
            xs, ys = [q[0] for q in poly], [q[1] for q in poly]
            boxes.append([min(xs) / UP, min(ys) / UP, max(xs) / UP, max(ys) / UP])
        boxes = [[max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])] for b in boxes
                 if b[2] > b[0] and b[3] > b[1]]
        done[p] = {"size": [W, H], "boxes": [[round(v, 1) for v in b] for b in boxes]}
        if k % 20 == 0:
            out.write_text(json.dumps(done))
            print(f"{k + 1}/{len(imgs)}", flush=True)
    out.write_text(json.dumps(done))
    print(f"{len(done)} images -> {out}")


if __name__ == "__main__":
    main()
