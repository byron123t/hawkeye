#!/usr/bin/env python3
"""
Does Hawkeye look at more of the text than its base? (paper appendix, "Attention on detected
text, at scale").

Text regions come from text_regions.py (detector boxes, no hand annotation). Attention sinks are
excluded. A patch is "text" if it overlaps any detected box. Per image and arm:
  text_mass      mean over generated tokens of the share of non-sink attention on text patches
  peaks_on_text  share of tokens whose peak patch is a text patch
  coverage       share of text boxes touched by at least one token's peak patch
  coverage_k     the same over the first k tokens, k = the shorter of the two arms' outputs
                 (removes the advantage of simply generating more tokens)
  unique_peaks   distinct peak patches per 100 tokens
Hawkeye - base is paired over images, bootstrapped over videos (BSTRB frames) or images, with
Holm across every test in the run.

    python analysis/reading_metrics.py --regions text_regions.json --run bstrb_4b:attn/4b \
        --run bstrb_8b:attn/8b --out reading_metrics.json [--dilate 1] [--no_sink_mask]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_maps, paired_samples, sink_mask  # noqa: E402
from hawkeye.stats import apply_holm, paired_test  # noqa: E402

KEYS = ("text_mass", "peaks_on_text", "coverage", "coverage_k", "unique_peaks")


def geometry(grid, size, boxes, dilate=0):
    """Text-patch mask (gy, gx) and, per box, the set of patch indices it overlaps."""
    gy, gx = grid
    W, H = size
    pw, ph = W / gx, H / gy
    text = np.zeros(grid, bool)
    box_patches = []
    for x0, y0, x1, y1 in boxes:
        ix0, ix1 = int(x0 // pw), min(gx - 1, int(np.ceil(x1 / pw)) - 1)
        iy0, iy1 = int(y0 // ph), min(gy - 1, int(np.ceil(y1 / ph)) - 1)
        ix0, iy0 = max(0, ix0 - dilate), max(0, iy0 - dilate)
        ix1, iy1 = min(gx - 1, ix1 + dilate), min(gy - 1, iy1 + dilate)
        text[iy0:iy1 + 1, ix0:ix1 + 1] = True
        box_patches.append({iy * gx + ix for iy in range(iy0, iy1 + 1) for ix in range(ix0, ix1 + 1)})
    return text, box_patches


def arm_metrics(Hm, text, box_patches, k, use_sink_mask=True):
    keep = ~sink_mask(Hm) if use_sink_mask else np.ones(Hm.shape[1:], bool)
    tflat, kflat = text.ravel(), keep.ravel()
    mass, on, peaks = [], [], []
    for h in Hm:
        q = h.ravel() * kflat
        if q.sum() <= 0:
            continue
        mass.append(float(q[tflat].sum() / q.sum()))
        p = int(q.argmax())
        peaks.append(p)
        on.append(bool(tflat[p]))
    if not peaks:
        return None

    def cov(ps):
        return float(np.mean([bool(bp & ps) for bp in box_patches]))

    return {"text_mass": float(np.mean(mass)), "peaks_on_text": float(np.mean(on)),
            "coverage": cov(set(peaks)), "coverage_k": cov(set(peaks[:k])),
            "unique_peaks": 100 * len(set(peaks)) / len(peaks), "tokens": len(peaks),
            "text_area": float(text.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True)
    ap.add_argument("--run", action="append", required=True, help="name:dir")
    ap.add_argument("--dilate", type=int, default=0, help="grow boxes by N patches (robustness)")
    ap.add_argument("--no_sink_mask", action="store_true", help="keep sink patches (robustness)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(a.regions))
    rng = np.random.default_rng(0)
    res, per, tests = {}, {}, []
    for spec in a.run:
        name, root = spec.split(":", 1)
        rows = []
        for sdir in paired_samples(Path(root)):
            Hb, mdb = load_maps(sdir / "base")
            Hf, _ = load_maps(sdir / "hawkeye")
            s = mdb["sample"]
            reg = R.get(s["image"])
            if Hb is None or Hf is None or not reg or not reg["boxes"] or Hb.shape[1:] != Hf.shape[1:]:
                continue
            text, bp = geometry(Hb.shape[1:], reg["size"], reg["boxes"], a.dilate)
            k = min(len(Hb), len(Hf))
            mb = arm_metrics(Hb, text, bp, k, not a.no_sink_mask)
            mf = arm_metrics(Hf, text, bp, k, not a.no_sink_mask)
            if mb and mf:
                rows.append({"id": s.get("id"), "image": s["image"], "unit": s.get("video_id") or s.get("id") or s["image"],
                             "text_height_px": s.get("text_height_px"), "boxes": len(reg["boxes"]),
                             "base": mb, "hawkeye": mf})
        per[name] = rows
        out = {"n_images": len(rows), "n_units": len({r["unit"] for r in rows}), "measures": {}}
        for key in KEYS:
            r = paired_test([x["hawkeye"][key] for x in rows], [x["base"][key] for x in rows],
                            [x["unit"] for x in rows], rng=rng)
            out["measures"][key] = r
            tests.append(r)
        res[name] = out
    apply_holm(tests)
    for name, o in res.items():
        print(f"== {name}: {o['n_images']} images ({o['n_units']} units)")
        for key, r in o["measures"].items():
            print(f"  {key:14s} base {r.mean_b:.3f} -> hawkeye {r.mean_a:.3f}  {r.diff:+.3f} "
                  f"[{r.lo:+.3f}, {r.hi:+.3f}]  p_holm {r.p_holm:.4f}  {r.verdict}")
        o["measures"] = {k: v.to_dict() for k, v in o["measures"].items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).with_name(Path(a.out).stem + "_per_image.json").write_text(json.dumps(per))


if __name__ == "__main__":
    main()
