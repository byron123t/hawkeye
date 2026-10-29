#!/usr/bin/env python3
"""
How fine-tuning changes the spread of attention over the image (paper appendix, "Attention").

Per image and arm, with attention-sink patches reported separately and excluded elsewhere:
  sink     share of attention on sink patches
  entropy  normalized entropy of the non-sink attention (1 = uniform)
  top1     mass in the top 1% of non-sink patches;  top5: top 5%
  move     mean distance between consecutive tokens' peak patches (fraction of the grid diagonal)
Each per-token value is averaged over the image's tokens; Hawkeye - base is paired over images
with a 95% bootstrap CI and the standardized effect d_z.

    python analysis/attention_stats.py attn/4b --json attn/4b_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_maps, paired_samples, sink_mask

KEYS = ("sink", "entropy", "top1", "top5", "move")


def per_image(arm_dir: Path) -> dict | None:
    H, _ = load_maps(arm_dir)
    if H is None:
        return None
    grid = H.shape[1:]
    sink = sink_mask(H)
    keep = ~sink.ravel()
    n = int(keep.sum())
    ent, t1, t5, sinkm, peaks = [], [], [], [], []
    for h in H:
        p = h.ravel()
        if p.sum() <= 0:
            continue
        sinkm.append(float(p[~keep].sum() / p.sum()))
        q = p * keep
        if q.sum() <= 0:
            continue
        q = q / q.sum()
        nz = q[q > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / np.log(n)) if n > 1 else 0.0)
        srt = np.sort(q[keep])[::-1]
        t1.append(float(srt[:max(1, int(round(0.01 * n)))].sum()))
        t5.append(float(srt[:max(1, int(round(0.05 * n)))].sum()))
        peaks.append(np.unravel_index(int(q.argmax()), grid))
    if not ent:
        return None
    diag = float(np.hypot(*grid))
    mv = [np.hypot(a[0] - b[0], a[1] - b[1]) / diag for a, b in zip(peaks, peaks[1:])]
    return {"entropy": np.mean(ent), "top1": np.mean(t1), "top5": np.mean(t5),
            "move": float(np.mean(mv)) if mv else np.nan, "sink": float(np.mean(sinkm)),
            "n_sink_patches": int(sink.sum()), "tokens": len(ent), "grid": list(grid)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="extract_attention.py output with base/ and hawkeye/ arms")
    ap.add_argument("--json", default=None)
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    rows = []
    for sdir in paired_samples(Path(a.root)):
        b, f = per_image(sdir / "base"), per_image(sdir / "hawkeye")
        if b and f and b["grid"] == f["grid"]:
            rows.append((sdir.name, b, f))
    rng = np.random.default_rng(0)
    res = {"n_images": len(rows), "measures": {}}
    for k in KEYS:
        bv = np.array([b[k] for _, b, _ in rows], float)
        fv = np.array([f[k] for _, _, f in rows], float)
        ok = np.isfinite(bv) & np.isfinite(fv)
        d = fv[ok] - bv[ok]
        boot = d[rng.integers(0, len(d), size=(a.B, len(d)))].mean(1)
        res["measures"][k] = {"base": round(float(bv[ok].mean()), 4), "hawkeye": round(float(fv[ok].mean()), 4),
                              "diff": round(float(d.mean()), 4),
                              "lo": round(float(np.percentile(boot, 2.5)), 4),
                              "hi": round(float(np.percentile(boot, 97.5)), 4),
                              "d_z": round(float(d.mean() / d.std(ddof=1)), 2) if len(d) > 1 and d.std() > 0 else None,
                              "n": int(ok.sum())}
    print(f"{len(rows)} images")
    for k, m in res["measures"].items():
        print(f"  {k:8s} base {m['base']:.3f} -> hawkeye {m['hawkeye']:.3f}  "
              f"diff {m['diff']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]  d_z {m['d_z']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
