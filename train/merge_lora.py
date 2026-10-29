#!/usr/bin/env python3
"""
Merge a LoRA SFT run (train_sft.py --lora_enable True) into full weights.

Picks the checkpoint with the lowest validation loss (or --step), folds in modules_to_save (e.g.
a ViT trained in full) with merge_and_unload, and copies the processor files. Runs on CPU; needs
RAM for the model. Not needed for the Hawkeye 4B/8B recipe (full SFT; the GRPO script writes
merged weights itself).

  python train/merge_lora.py --base Qwen/Qwen3-VL-4B-Instruct --run runs/lora_sft --out runs/lora_sft_merged
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
import transformers
from transformers import AutoConfig, AutoProcessor

PROC_FILES = ("preprocessor_config.json", "video_preprocessor_config.json",
              "chat_template.json", "chat_template.jinja", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
              "merges.txt", "added_tokens.json")


def best_adapter(run: Path) -> tuple[Path, dict]:
    st = json.load(open(run / "trainer_state.json"))
    ev = {x["step"]: x["eval_loss"] for x in st["log_history"] if "eval_loss" in x}
    cands = {int(p.name.split("-")[1]): p for p in run.glob("checkpoint-*")
             if (p / "adapter_config.json").exists()}
    cands = {s: p for s, p in cands.items() if s in ev}
    if st.get("global_step") in ev and (run / "adapter_config.json").exists():
        cands.setdefault(st["global_step"], run)
    step = min(cands, key=lambda s: ev[s])
    return cands[step], {"selected_step": step, "selected_eval_loss": ev[step],
                         "candidates": {s: ev[s] for s in sorted(cands)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, help="merge this checkpoint instead of the best-val one")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    if (out / "config.json").exists():
        print(f"[merge] {out} already merged"); return
    adapter, sel = best_adapter(run)
    if a.step:
        adapter = run / f"checkpoint-{a.step}"
        st = json.load(open(run / "trainer_state.json"))
        ev = {x["step"]: x["eval_loss"] for x in st["log_history"] if "eval_loss" in x}
        sel = {"selected_step": a.step, "selected_eval_loss": ev.get(a.step), "forced": True}
    print(f"[merge] {run.name}: adapter {adapter.name} {sel}")

    # peft probes for autoawq on every LoRA dispatch; some autoawq versions fail to import with
    # recent transformers. These models contain no AWQ layers, so report AWQ as absent.
    import peft.tuners.lora.awq as _peft_awq
    _peft_awq.is_auto_awq_available = lambda: False
    arch = AutoConfig.from_pretrained(a.base).architectures[0]
    cls = getattr(transformers, arch)
    model = cls.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for f in PROC_FILES:  # processor/tokenizer from the run (saved at the end), else base
        src = run / f if (run / f).exists() else Path(a.base) / f
        if src.exists():
            shutil.copy2(src, out / f)
    AutoProcessor.from_pretrained(out)  # fail fast if anything needed for eval is missing
    (out / "merge_selection.json").write_text(json.dumps(
        {"run": str(run), "adapter": str(adapter), "base": a.base, **sel}, indent=2))
    print(f"[merge] wrote {out}")


if __name__ == "__main__":
    main()
