#!/usr/bin/env python3
"""
Run API models (any OpenAI-compatible chat-completions endpoint) over a BSTRB split.

Credentials come from the environment only:
    OPENAI_BASE_URL   e.g. https://api.openai.com/v1  or a LiteLLM / vLLM proxy URL ending in /v1
    OPENAI_API_KEY

Protocol used for the paper's API baselines: the fixed transcription-only system prompt below,
then one user turn with the item's prompt and the image (JPEG, longest side <= 2048 px),
temperature 0 where the endpoint accepts it, and a large output cap so reasoning models are not
truncated (check `finish_reason == "length"` afterwards). Responses are appended to
<out>/raw/<name>.jsonl as they arrive (resumable; failed rows are retried on the next run), and
<out>/predictions_<name>.jsonl is written in the same schema as run_local.py.

    export OPENAI_BASE_URL=... OPENAI_API_KEY=...
    python eval/run_api.py --model gpt-4o --out results/ --workers 8
    python eval/run_api.py --model my-proxy-name --name qwen3-vl-235b --no_temperature --out results/
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkeye.data import load_split, resolve_root, unit_of  # noqa: E402

SYSTEM_PROMPT = (
    "You are an OCR engine. Transcribe the text visible in the image exactly as it "
    "appears.\n"
    "Output rules:\n"
    "- Output ONLY the transcribed text. No preamble, no explanation, no commentary, "
    "no closing remarks.\n"
    "- Do not describe the image. Do not say what the text is or where it is.\n"
    "- Preserve the original line breaks: one visual line of text per output line.\n"
    "- Do not wrap the output in quotes or code fences, and do not add Markdown.\n"
    "- Do not correct spelling or expand abbreviations; reproduce what is printed.\n"
    "- If no text is legible, output nothing."
)
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529, None}
_lock = threading.Lock()


def encode_image(path: str, max_side: int = 2048) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def call(base: str, key: str, model: str, image: str, prompt: str, *, max_tokens: int,
         temperature, system: bool, timeout: float = 600.0) -> dict:
    msgs = ([{"role": "system", "content": SYSTEM_PROMPT}] if system else []) + [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}]}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        ch = (d.get("choices") or [{}])[0]
        return {"ok": True, "text": (ch.get("message") or {}).get("content") or "",
                "finish_reason": ch.get("finish_reason"), "usage": d.get("usage") or {},
                "latency_s": time.perf_counter() - t0}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.read().decode(errors="replace")[:600],
                "latency_s": time.perf_counter() - t0}
    except Exception as e:
        return {"ok": False, "status": None, "error": f"{type(e).__name__}: {e}"[:600],
                "latency_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name at the endpoint")
    ap.add_argument("--name", default=None, help="short name for output files (default: --model)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--kinds", default="video_frame,photo")
    ap.add_argument("--max_tokens", type=int, default=16000,
                    help="raise to 32k+ for reasoning models; check finish_reason afterwards")
    ap.add_argument("--no_temperature", action="store_true", help="for endpoints that reject temperature")
    ap.add_argument("--no_system_prompt", action="store_true", help="user turn only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, key = os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        sys.exit("set OPENAI_BASE_URL and OPENAI_API_KEY")
    name = a.name or a.model.replace("/", "_")

    rows = load_split(resolve_root(a.data), a.split, set(a.kinds.split(",")))
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    raw = out / "raw" / f"{name}.jsonl"
    ok = set()
    if raw.exists():
        ok = {r["id"] for r in map(json.loads, filter(str.strip, open(raw, encoding="utf-8"))) if r["ok"]}
    todo = [r for r in rows if r["id"] not in ok]
    print(f"[run_api] {name}: {len(todo)} to do, {len(ok)} done")

    def one(r):
        res = None
        for k in range(1, a.attempts + 1):
            res = call(base, key, a.model, r["image"], r["prompt"], max_tokens=a.max_tokens,
                       temperature=None if a.no_temperature else 0.0, system=not a.no_system_prompt)
            if res["ok"] or res.get("status") not in RETRYABLE:
                break
            time.sleep(min(60, 2 ** k + random.random()))
        return {"id": r["id"], "attempts": k, **res}

    with cf.ThreadPoolExecutor(a.workers) as ex, open(raw, "a", encoding="utf-8") as fh:
        for k, res in enumerate(ex.map(one, todo), 1):
            with _lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
            if k % 100 == 0:
                print(f"  {k}/{len(todo)}", flush=True)

    latest = {}
    for r in map(json.loads, filter(str.strip, open(raw, encoding="utf-8"))):
        if r["id"] not in latest or not latest[r["id"]]["ok"] or r["ok"]:
            latest[r["id"]] = r
    n_err = 0
    with open(out / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            res = latest.get(r["id"])
            if res is None:
                continue
            n_err += not res["ok"]
            f.write(json.dumps({"id": r["id"], "split": r["split"], "kind": r["kind"], "unit": unit_of(r),
                                "text_height_px": r.get("text_height_px"), "prompt": r["prompt"],
                                "reference": r["transcription"],
                                "prediction": res["text"].strip() if res["ok"] else "[ERROR]",
                                "finish_reason": res.get("finish_reason"), "model": name},
                               ensure_ascii=False) + "\n")
    print(f"[run_api] wrote predictions_{name}.jsonl ({n_err} errors)")


if __name__ == "__main__":
    main()
