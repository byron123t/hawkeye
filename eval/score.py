#!/usr/bin/env python3
"""
Score prediction files and print a summary table.

For every <results>/predictions_<name>.jsonl, writes <results>/scores_<name>.jsonl (per-item
metrics) and adds the model to <results>/summary.json. Means are item means ("macro", as in the
paper) with 95% CIs that resample videos / photos. Subsets: all, video frames, frames with mean
text height <= 20 px and > 20 px, photos. Rows whose prediction starts with "[ERROR]" (failed
API calls) are excluded and counted.

    python eval/score.py results/
    python eval/score.py results/ --metric wir_rate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkeye.metrics import degenerate_reason, score  # noqa: E402
from hawkeye.stats import cluster_bootstrap_mean  # noqa: E402

METRICS = ("bow_f1", "bow_precision", "bow_recall", "wir_rate", "wer", "align_f1")
SUBSETS = {
    "all": lambda r: True,
    "video": lambda r: r["kind"] == "video_frame",
    "video_le20px": lambda r: r["kind"] == "video_frame" and (r.get("text_height_px") or 1e9) <= 20,
    "video_gt20px": lambda r: r["kind"] == "video_frame" and (r.get("text_height_px") or 0) > 20,
    "photo": lambda r: r["kind"] == "photo",
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def score_file(path: Path) -> list[dict]:
    rows = []
    for r in load_jsonl(path):
        if str(r["prediction"]).startswith("[ERROR]"):
            continue
        s = score(r["reference"], r["prediction"]).to_dict()
        rows.append({k: r.get(k) for k in ("id", "split", "kind", "unit", "text_height_px", "model")}
                    | {m: s[m] for m in METRICS}
                    | {"gt_words": s["gt_words"], "hyp_words": s["hyp_words"],
                       "degenerate": degenerate_reason(r["prediction"])})
    return rows


def summarise(rows: list[dict], metric: str) -> dict:
    out = {}
    for name, keep in SUBSETS.items():
        sub = [r for r in rows if keep(r)]
        if sub:
            ci = cluster_bootstrap_mean([r[metric] for r in sub], [r["unit"] for r in sub])
            out[name] = {"n": ci.n, "units": ci.units, "mean": round(ci.mean, 4),
                         "ci95": [round(ci.lo, 4), round(ci.hi, 4)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--metric", default="bow_f1", choices=METRICS)
    a = ap.parse_args()
    res = Path(a.results)
    summary = {}
    for p in sorted(res.glob("predictions_*.jsonl")):
        if ".shard" in p.name:
            continue
        name = p.stem[len("predictions_"):]
        n_total = sum(1 for l in open(p, encoding="utf-8") if l.strip())
        rows = score_file(p)
        with open(res / f"scores_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[name] = {"rows": len(rows), "errors": n_total - len(rows),
                         "loops": sum(r["degenerate"] == "degenerate_repetition" for r in rows),
                         **{m: summarise(rows, m) for m in METRICS}}
    (res / "summary.json").write_text(json.dumps(summary, indent=1))

    cols = list(SUBSETS)
    print(f"\n{a.metric} (item mean, 95% CI resampling videos/photos)\n")
    print("| model | " + " | ".join(cols) + " | errors |")
    print("|---|" + "---|" * (len(cols) + 1))
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1][a.metric].get("all", {}).get("mean", 0)):
        cells = [f"{s[a.metric][c]['mean']:.3f} [{s[a.metric][c]['ci95'][0]:.3f}, {s[a.metric][c]['ci95'][1]:.3f}]"
                 if c in s[a.metric] else "-" for c in cols]
        print(f"| {name} | " + " | ".join(cells) + f" | {s['errors']} |")
    print(f"\nwrote {res / 'summary.json'}")


if __name__ == "__main__":
    main()
