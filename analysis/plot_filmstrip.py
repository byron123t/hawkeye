#!/usr/bin/env python3
"""
Per-decoding-step attention "filmstrip" for one image: one row per (run, arm); each panel is the
attention map (sinks excluded) at the step that emits the word in its title, zoomed to the word,
with the word's OCR box (green: the peak is on the word, pink: not) and the peak (cross).

Words are picked by rule: content words (>= 4 letters, no stopwords) of the row's own output
that OCR also finds, preferring those whose peak lands on the word, spread evenly over decoding.

    python analysis/plot_filmstrip.py --runs 4B:attn/photos_4b 8B:attn/photos_8b \
        --sample <sample id> --ocr ocr_words.json --out figs/ [--n 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import plotstyle
from common import ARMS, load_maps, sink_mask
from grounding import MINLEN, norm, words_with_steps

STOP = {"the", "and", "that", "this", "with", "what", "they", "their", "there", "from", "have", "had",
        "into", "our", "for", "are", "was", "were", "you", "your", "can", "could", "would", "anyone"}
ZOOM = 0.30          # each panel: a 2:1 window, ZOOM x image width, centred on the emitted word


def pick_words(arm: Path, ocr, n: int):
    H, md = load_maps(arm)
    toks = [t for t in md["tokens"] if t.get("heatmap")]
    keep = ~sink_mask(H)
    gy, gx = H.shape[1:]
    W, Hh = ocr["size"]
    pw, ph = W / gx, Hh / gy
    step_i = {t["step"]: i for i, t in enumerate(toks)}
    index = {}
    for w in ocr["words"]:
        k = norm(w["text"])
        if len(k) >= MINLEN and w["conf"] >= 0.3:
            index.setdefault(k, []).append(w["box"])
    cands = []
    for word, steps in words_with_steps(toks):
        k = norm(word)
        if len(k) < 4 or k in STOP or k.isdigit():
            continue
        boxes = index.get(k, [])
        best = None
        for s in steps:
            if s not in step_i:
                continue
            h = H[step_i[s]] * keep
            iy, ix = np.unravel_index(int(h.argmax()), h.shape)
            cx, cy = (ix + 0.5) * pw, (iy + 0.5) * ph
            hit = any(x0 - pw / 2 <= cx <= x1 + pw / 2 and y0 - ph / 2 <= cy <= y1 + ph / 2
                      for x0, y0, x1, y1 in boxes)
            if best is None or (hit and not best[3]):
                best = (s, word, (cx, cy), hit)
        if best:
            cands.append({"step": best[0], "word": best[1], "peak": best[2], "hit": best[3],
                          "boxes": boxes, "map": H[step_i[best[0]]] * keep})
    hits = [c for c in cands if c["hit"]]
    pool = hits if len(hits) >= n else hits + [c for c in cands if not c["hit"] and c["boxes"]] + \
        [c for c in cands if not c["boxes"]]
    pool.sort(key=lambda c: c["step"])
    if len(pool) > n:
        pool = [pool[int(round(i * (len(pool) - 1) / (n - 1)))] for i in range(n)]
    return pool, len(toks), sum(c["hit"] for c in cands), sum(bool(c["boxes"]) for c in cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label:dir, e.g. 4B:attn/photos_4b")
    ap.add_argument("--sample", required=True, help="sample directory name inside each run")
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    O = json.load(open(a.ocr))
    rows = []
    for spec in a.runs:
        label, root = spec.split(":", 1)
        for arm in ARMS:
            name = f"{'Base' if arm == 'base' else 'Hawkeye'} {label}"
            color = plotstyle.BASE_COLOR if arm == "base" else plotstyle.HAWKEYE_COLOR.get(label.lower(), "#008300")
            rows.append((Path(root) / a.sample / arm, name, color))
    img = json.load(open(rows[0][0] / "metadata.json"))["sample"]["image"]
    ocr = O[img]
    photo = Image.open(img).convert("RGB")
    W, Hh = photo.size
    grey = np.asarray(photo.convert("L").convert("RGB")).astype(float) / 255 * 0.65
    plt = plotstyle.apply()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    hot = LinearSegmentedColormap.from_list(
        "hot_alpha", [(1, .45, 0, 0), (1, .45, 0, .55), (1, .85, .1, .85), (1, 1, .6, .95)])
    n, aspect = a.n, 0.5
    pw_in = (plotstyle.FULL_WIDTH / 1.1 - 0.35) / n
    fig, axes = plt.subplots(len(rows), n, figsize=(plotstyle.FULL_WIDTH / 1.1, len(rows) * (pw_in * aspect + 0.2) + 0.1),
                             squeeze=False)
    meta = []
    for row, (arm_dir, label, col) in zip(axes, rows):
        picks, ntok, nhit, nmatch = pick_words(arm_dir, ocr, n)
        meta.append({"row": label, "tokens": ntok, "ocr_matched_words": nmatch, "grounded_words": nhit,
                     "shown": [{"step": c["step"], "word": c["word"], "hit": c["hit"]} for c in picks]})
        for k, ax in enumerate(row):
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(1.0)
            if k >= len(picks):
                ax.axis("off")
                continue
            c = picks[k]
            ax.imshow(grey)
            ax.imshow(c["map"], extent=(0, W, Hh, 0), cmap=hot, interpolation="bilinear",
                      vmin=0, vmax=np.percentile(c["map"], 99.7))
            for x0, y0, x1, y1 in c["boxes"]:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.0,
                                       ec="#1aa260" if c["hit"] else "#d0607a"))
            ax.plot(*c["peak"], marker="+", ms=7, mew=1.1, color="#00e5ff")
            if c["boxes"]:
                bx = min(c["boxes"], key=lambda b: abs((b[0] + b[2]) / 2 - c["peak"][0]) + abs((b[1] + b[3]) / 2 - c["peak"][1]))
                mx, my = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            else:
                mx, my = c["peak"]
            ww, wh = ZOOM * W, ZOOM * W * aspect
            x0, y0 = min(max(0, mx - ww / 2), W - ww), min(max(0, my - wh / 2), Hh - wh)
            ax.set_xlim(x0, x0 + ww)
            ax.set_ylim(y0 + wh, y0)
            ax.set_title(f"step {c['step']}: “{c['word'][:16]}”", fontsize=5.6, pad=1.5)
        row[0].set_ylabel(label, fontsize=6, color=col, fontweight="bold", labelpad=2)
    fig.tight_layout(pad=0.15, w_pad=0.2, h_pad=0.3)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "attention_filmstrip.jpg", dpi=260)
    plt.close(fig)
    (out / "attention_filmstrip.json").write_text(json.dumps({"image": img, "rows": meta}, indent=1))
    for m in meta:
        print(f"{m['row']}: {m['tokens']} tokens, {m['grounded_words']}/{m['ocr_matched_words']} grounded; "
              f"shown {[(s['step'], s['word'], s['hit']) for s in m['shown']]}")


if __name__ == "__main__":
    main()
