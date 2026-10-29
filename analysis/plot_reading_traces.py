#!/usr/bin/env python3
"""
Reading trace of base vs Hawkeye on one image: the attention peak of every generated token
(sinks excluded), coloured by decoding step, over the photo with its text regions -- green if a
peak lands in the region, pink if never ("blind spot"). Panel titles give the share of peaks on
text and the number of regions read.

Regions: a JSON list of {"x0","y0","x1","y1"} in image pixels (--rois), or the detector boxes for
that image from text_regions.py (--regions).

    python analysis/plot_reading_traces.py --run attn/photos_4b --sample <sample id> \
        --regions text_regions.json --label 4B --out figs/trace_4b.jpg [--panel_w 1.6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import plotstyle
from common import ARMS, load_maps, peak_points


def render(photo_path, arms, rois, out_path, plt, panel_w=None):
    """arms: list of (label, colour, arm_dir), drawn side by side. Returns per-panel numbers."""
    from matplotlib.patches import Patch, Rectangle
    photo = Image.open(photo_path).convert("RGB")
    W, Hh = photo.size
    native = panel_w is not None                 # drawn at print size
    panel_w = panel_w or (3.4 if W > Hh else 1.9)
    k = min(1.0, panel_w / 3.4)
    fs = 5.6 if native else 6.4
    fig, axes = plt.subplots(1, len(arms), squeeze=False,
                             figsize=(panel_w * len(arms) + (0.35 if native else 0),
                                      panel_w * Hh / W + (0.62 if native else 0.75)))
    res, sc = {}, None
    for ax, (label, col, arm) in zip(axes[0], arms):
        H, _ = load_maps(arm)
        xy = [(x * W, y * Hh) for x, y in peak_points(H)]
        inside = [[r["x0"] <= x <= r["x1"] and r["y0"] <= y <= r["y1"] for r in rois] for x, y in xy]
        on_text = float(np.mean([any(v) for v in inside])) if inside else 0.0
        read = [any(v[j] for v in inside) for j in range(len(rois))]
        ax.imshow(photo, alpha=0.55)
        for r, ok in zip(rois, read):
            ax.add_patch(Rectangle((r["x0"], r["y0"]), r["x1"] - r["x0"], r["y1"] - r["y0"],
                                   fill=not ok, facecolor="#e8a0b0", alpha=0.55 if not ok else 1,
                                   edgecolor="#1aa260" if ok else "#d0607a", lw=1.0 * max(k, 0.6)))
        if xy:
            xs, ys = zip(*xy)
            ax.plot(xs, ys, color="#555555", lw=0.4 * max(k, 0.6), alpha=0.6, zorder=3)
            sc = ax.scatter(xs, ys, c=np.arange(len(xs)), cmap="turbo", s=8 * max(k, 0.45), zorder=4,
                            edgecolors="white", linewidths=0.3 * max(k, 0.6))
        ax.set_title(f"{label}\n{100 * on_text:.0f}% of peaks on text\n{sum(read)}/{len(rois)} regions · {len(xy)} tokens",
                     fontsize=fs, color=col, fontweight="bold", linespacing=1.1, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        res[label] = {"tokens": len(xy), "peaks_on_text": round(on_text, 3), "regions_read": int(sum(read)),
                      "regions": len(rois)}
    if sc is not None:
        cb = fig.colorbar(sc, ax=list(axes[0]), fraction=0.02, pad=0.01)
        cb.set_label("decoding step", fontsize=fs - 0.4, labelpad=1)
        cb.ax.tick_params(labelsize=fs - 1, length=1.5, pad=1)
    fig.legend(handles=[Patch(facecolor="none", edgecolor="#1aa260", label="text region the model peaks in"),
                        Patch(facecolor="#e8a0b0", edgecolor="#d0607a", label="region never peaked in (blind spot)")],
               loc="upper center", ncol=2, fontsize=fs, frameon=False, handlelength=1.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, min(ax.get_position().y0 for ax in axes[0]) - 0.01))
    fig.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="extract_attention.py output dir")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--label", default="4B", help="model size label for the panel titles")
    ap.add_argument("--rois", default=None, help='JSON list of {"x0","y0","x1","y1"}')
    ap.add_argument("--regions", default=None, help="text_regions.py output (used if --rois is absent)")
    ap.add_argument("--panel_w", type=float, default=None, help="panel width in inches (print size)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sdir = Path(a.run) / a.sample
    img = json.load(open(sdir / ARMS[0] / "metadata.json"))["sample"]["image"]
    if a.rois:
        rois = json.load(open(a.rois))
    else:
        boxes = json.load(open(a.regions))[img]["boxes"]
        rois = [dict(x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for b in boxes]
    plt = plotstyle.apply()
    lab = a.label
    arms = [(f"Base Qwen3-VL-{lab}", plotstyle.BASE_COLOR, sdir / "base"),
            (f"Hawkeye {lab}", plotstyle.HAWKEYE_COLOR.get(lab.lower(), "#008300"), sdir / "hawkeye")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(img, arms, rois, a.out, plt, a.panel_w), indent=1))


if __name__ == "__main__":
    main()
