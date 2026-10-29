#!/usr/bin/env python3
"""
Word-level grounding of the per-token attention ("pointing accuracy"): for every word a model
emits that OCR also finds in the image (ocr_words.py, matched on normalized text, >= 3 letters,
OCR confidence >= 0.3), is the attention peak at one of the steps that emit it (sinks excluded)
on a box of that word? Chance = share of the image's patches covered by that word's boxes.

Also scores each arm's own output against the reference with a bag-of-words F1 on alphanumeric
lowercase words, so examples can be checked for accuracy. Hawkeye - base is paired per image,
bootstrapped over units (--units JSON mapping sample id -> unit, e.g. object; default: the
sample's video_id or id), Holm across tests.

    python analysis/grounding.py --ocr ocr_words.json --run 4b:attn/photos_4b --run 8b:attn/photos_8b \
        --out grounding.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_maps, paired_samples, sink_mask  # noqa: E402
from hawkeye.stats import apply_holm, paired_test  # noqa: E402

MINLEN = 3


def norm(w: str) -> str:
    return re.sub(r"[^0-9a-z]", "", w.lower())


def words_with_steps(tokens):
    """[(word, [steps])] from the generated tokens; a word ends at whitespace."""
    out, cur, steps = [], "", []
    for t in tokens:
        for ch in t["token_text"]:
            if ch.isspace():
                if cur:
                    out.append((cur, steps))
                    cur, steps = "", []
            else:
                cur += ch
                if not steps or steps[-1] != t["step"]:
                    steps.append(t["step"])
    if cur:
        out.append((cur, steps))
    return out


def bow_f1_simple(pred: str, gt: str) -> float:
    P, G = Counter(map(norm, pred.split())), Counter(map(norm, gt.split()))
    P.pop("", None)
    G.pop("", None)
    o = sum((P & G).values())
    return 0.0 if not o else 2 * o / (sum(P.values()) + sum(G.values()))


def word_index(ocr) -> dict:
    index = {}
    for w in ocr["words"]:
        n = norm(w["text"])
        if len(n) >= MINLEN and w["conf"] >= 0.3:
            index.setdefault(n, []).append(w["box"])
    return index


def analyse(arm: Path, ocr, gt: str) -> dict:
    H, md = load_maps(arm)
    toks = [t for t in md["tokens"] if t.get("heatmap")]
    keep = ~sink_mask(H)
    gy, gx = H.shape[1:]
    W, Hh = ocr["size"]
    pw, ph = W / gx, Hh / gy
    peak = {t["step"]: np.unravel_index(int((h * keep).argmax()), (gy, gx)) for t, h in zip(toks, H)}
    index = word_index(ocr)
    rec = []
    for word, steps in words_with_steps(toks):
        n = norm(word)
        if len(n) < MINLEN or n not in index:
            continue
        mask = np.zeros((gy, gx), bool)
        for x0, y0, x1, y1 in index[n]:
            mask[int(y0 // ph):min(gy, int(np.ceil(y1 / ph))), int(x0 // pw):min(gx, int(np.ceil(x1 / pw)))] = True
        hit = any(mask[peak[s]] for s in steps if s in peak)
        rec.append({"word": word, "steps": steps, "hit": bool(hit), "chance": float(mask.mean())})
    return {"pred": md["prediction"], "bow_f1": bow_f1_simple(md["prediction"], gt or ""), "tokens": len(toks),
            "words": rec, "acc": float(np.mean([r["hit"] for r in rec])) if rec else np.nan,
            "chance": float(np.mean([r["chance"] for r in rec])) if rec else np.nan, "n_words": len(rec)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--run", action="append", required=True, help="name:dir")
    ap.add_argument("--units", default=None, help="optional JSON {sample id: unit}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O = json.load(open(a.ocr))
    units = json.load(open(a.units)) if a.units else {}
    rng = np.random.default_rng(0)
    res, per, tests = {}, {}, []
    for spec in a.run:
        name, root = spec.split(":", 1)
        rows = []
        for sdir in paired_samples(Path(root)):
            s = json.load(open(sdir / "base" / "metadata.json"))["sample"]
            if s["image"] not in O:
                continue
            b = analyse(sdir / "base", O[s["image"]], s.get("ground_truth"))
            f = analyse(sdir / "hawkeye", O[s["image"]], s.get("ground_truth"))
            rows.append({"id": s.get("id"), "image": s["image"], "dir": str(sdir),
                         "unit": units.get(s.get("id")) or s.get("video_id") or s.get("id") or s["image"],
                         "base": b, "hawkeye": f})
        per[name] = rows
        out = {"n_images": len(rows), "chance": float(np.nanmean([r["hawkeye"]["chance"] for r in rows]))}
        for key in ("acc", "bow_f1", "n_words"):
            ok = [r for r in rows if np.isfinite(r["base"][key]) and np.isfinite(r["hawkeye"][key])]
            t = paired_test([r["hawkeye"][key] for r in ok], [r["base"][key] for r in ok],
                            [r["unit"] for r in ok], rng=rng)
            out[key] = t
            tests.append(t)
        res[name] = out
    apply_holm(tests)
    for name, o in res.items():
        print(f"== {name}: {o['n_images']} images, chance {o['chance']:.3f}")
        for key in ("acc", "bow_f1", "n_words"):
            t = o[key]
            print(f"  {key:8s} base {t.mean_b:.3f} -> hawkeye {t.mean_a:.3f}  {t.diff:+.3f} "
                  f"[{t.lo:+.3f}, {t.hi:+.3f}]  p_holm {t.p_holm:.4f}  (n={t.n})")
            o[key] = t.to_dict()
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).with_name(Path(a.out).stem + "_per_image.json").write_text(json.dumps(per))


if __name__ == "__main__":
    main()
