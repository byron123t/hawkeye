#!/usr/bin/env python3
"""
Paired comparisons between models scored on the same items (run score.py first).

Each pair A:B is tested on every subset (video frames, <= 20 px, > 20 px, photos) with a paired
bootstrap of the mean difference that resamples videos (frames share a transcript) and photos;
Holm correction is applied across all tests printed. Verdict: "better"/"worse" when the 95% CI
excludes zero and the Holm-adjusted p < 0.05, otherwise "n.s.".

    python eval/compare.py results/ --pairs hawkeye-4b:qwen3-vl-4b hawkeye-8b:qwen3-vl-8b
    python eval/compare.py results/ --pairs hawkeye-4b:qwen3-vl-4b --metric wir_rate --json out.json

With the paper's predictions this reproduces the headline gains (BoW F1, held-out test):
video frames +10.1 pp (4B) and +6.9 pp (8B); photos +6.0 pp and +5.3 pp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkeye.stats import apply_holm, paired_test  # noqa: E402
from score import METRICS, SUBSETS  # noqa: E402


def load(res: Path, name: str) -> dict:
    p = res / f"scores_{name}.jsonl"
    if not p.exists():
        sys.exit(f"missing {p}; run eval/score.py {res} first")
    return {r["id"]: r for r in map(json.loads, filter(str.strip, open(p, encoding="utf-8")))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B, tested as A - B")
    ap.add_argument("--metric", default="bow_f1", choices=METRICS)
    ap.add_argument("--subsets", default="video,video_le20px,video_gt20px,photo")
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    rng = np.random.default_rng(0)
    tests = []
    for pair in a.pairs:
        A, Bm = pair.split(":")
        sa, sb = load(res, A), load(res, Bm)
        common = [i for i in sa if i in sb]
        for sub in a.subsets.split(","):
            ids = [i for i in common if SUBSETS[sub](sa[i])]
            if not ids:
                continue
            r = paired_test([sa[i][a.metric] for i in ids], [sb[i][a.metric] for i in ids],
                            [sa[i]["unit"] for i in ids], B=a.B, rng=rng)
            tests.append(({"a": A, "b": Bm, "subset": sub, "metric": a.metric}, r))
    apply_holm([t[1] for t in tests])

    print(f"\n{a.metric}: A - B, paired bootstrap over videos/photos, Holm across {len(tests)} tests\n")
    print("| A | B | subset | n (units) | A | B | diff (pp) | 95% CI (pp) | p (Holm) | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for meta, r in tests:
        p = f"{r.p_holm:.4f}" + ("*" if r.p_at_floor else "")
        print(f"| {meta['a']} | {meta['b']} | {meta['subset']} | {r.n} ({r.units}) | {r.mean_a:.3f} | "
              f"{r.mean_b:.3f} | {100 * r.diff:+.1f} | [{100 * r.lo:+.1f}, {100 * r.hi:+.1f}] | {p} | {r.verdict} |")
    print(f"\n* raw bootstrap p at its floor (< 1/B = {1 / a.B:g}) before the Holm adjustment")
    if a.json:
        Path(a.json).write_text(json.dumps([meta | r.to_dict() for meta, r in tests], indent=1))


if __name__ == "__main__":
    main()
