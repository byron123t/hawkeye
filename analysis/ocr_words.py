#!/usr/bin/env python3
"""
Word boxes with recognised text (EasyOCR detect + recognise), for word-level grounding: does the
attention peak at the step that emits a word fall on that word? Boxes in original-image pixels.
Uses the GPU if available, with a per-image CPU fallback.

    python analysis/ocr_words.py --runs attn/photos --out ocr_words.json [--gpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from text_regions import images_from_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    imgs = sorted(set(images_from_runs(a.runs) + a.images))
    out = Path(a.out)
    done = json.load(open(out)) if out.exists() else {}
    import easyocr
    readers = {"main": easyocr.Reader(["en"], gpu=a.gpu, verbose=False)}
    for k, p in enumerate(imgs):
        if p in done:
            continue
        im = np.asarray(Image.open(p).convert("RGB"))
        try:
            res = readers["main"].readtext(im, paragraph=False, width_ths=0.0)     # word-level boxes
        except Exception:
            readers.setdefault("cpu", easyocr.Reader(["en"], gpu=False, verbose=False))
            res = readers["cpu"].readtext(im, paragraph=False, width_ths=0.0)
        words = []
        for poly, text, conf in res:
            xs, ys = [q[0] for q in poly], [q[1] for q in poly]
            words.append({"box": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                          "text": text, "conf": round(float(conf), 3)})
        done[p] = {"size": [im.shape[1], im.shape[0]], "words": words}
        if k % 10 == 0:
            out.write_text(json.dumps(done))
            print(f"{k + 1}/{len(imgs)}", flush=True)
    out.write_text(json.dumps(done))
    print(f"{len(done)} images -> {out}")


if __name__ == "__main__":
    main()
