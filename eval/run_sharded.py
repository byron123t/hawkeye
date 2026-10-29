#!/usr/bin/env python3
"""
Run run_local.py on several GPUs in parallel (one shard per GPU) and merge the shards into
<out>/predictions_<name>.jsonl. All other arguments are passed through to run_local.py.

    python eval/run_sharded.py --gpus 0,1,2,3 --model byron123t/Hawkeye-8B --name hawkeye-8b --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__.strip() + "\n\n--gpus G0,G1,...  one shard per GPU (default 0)\n\nrun_local.py options:\n", flush=True)
        subprocess.run([sys.executable, str(HERE / "run_local.py"), "--help"])
        return
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a, rest = ap.parse_known_args()
    gpus = [g for g in a.gpus.split(",") if g]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for k, g in enumerate(gpus):
        cmd = [sys.executable, str(HERE / "run_local.py"), "--name", a.name, "--out", a.out,
               "--shard", str(k), "--num_shards", str(len(gpus)), *rest]
        log = open(out / f"run_{a.name}.shard{k}.log", "a")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": g}))
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"[run_sharded] shard exit codes {codes}; see {out}/run_{a.name}.shard*.log")
    rows = {}
    for k in range(len(gpus)):
        f = out / f"predictions_{a.name}.shard{k}.jsonl"
        if f.exists():
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
    merged = out / f"predictions_{a.name}.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in sorted(rows.values(), key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[run_sharded] {len(rows)} rows -> {merged}")


if __name__ == "__main__":
    main()
