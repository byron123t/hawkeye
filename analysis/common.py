"""Shared helpers for the attention analyses (layout written by extract_attention.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARMS = ("base", "hawkeye")


def sink_mask(maps: np.ndarray) -> np.ndarray:
    """Attention-sink patches from a (tokens, H, W) stack: patches whose mean attention exceeds
    50x the median patch, capped at the top 1% of patches. A handful of content-free patches hold
    30-40% of the attention and are the argmax for nearly every token, so peak and top-k measures
    computed on raw maps describe the sink, not reading. Every measure excludes them."""
    m = maps.mean(0)
    cand = m > 50 * np.median(m)
    cap = max(1, int(0.01 * m.size))
    if cand.sum() > cap:
        thr = np.sort(m.ravel())[::-1][cap - 1]
        cand = m >= thr
    return cand


def load_maps(arm_dir: Path):
    """(maps (T, H, W) or None, metadata) for one arm of one sample."""
    md = json.load(open(arm_dir / "metadata.json"))
    maps = [np.clip(np.load(t["heatmap"]).astype(np.float64), 0, None) for t in md["tokens"] if t.get("heatmap")]
    return (np.stack(maps) if maps else None), md


def paired_samples(root: Path, arms=ARMS):
    """Sample directories under `root` that have every arm."""
    return [d for d in sorted(Path(root).iterdir())
            if d.is_dir() and all((d / a / "metadata.json").exists() for a in arms)]


def peak_points(maps: np.ndarray):
    """Per-token argmax patch (sinks excluded) as normalized (x, y) image coordinates."""
    keep = ~sink_mask(maps)
    gy, gx = maps.shape[1:]
    pts = []
    for h in maps:
        iy, ix = np.unravel_index(int((h * keep).argmax()), h.shape)
        pts.append(((ix + 0.5) / gx, (iy + 0.5) / gy))
    return pts
