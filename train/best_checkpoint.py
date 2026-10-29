#!/usr/bin/env python3
"""Print the saved checkpoint of an SFT run with the lowest validation loss.

The SFT stage saves every 100 steps and evaluates on the validation split; GRPO starts from the
best one (checkpoint-400 for both Hawkeye 4B and 8B).

    python train/best_checkpoint.py runs/hawkeye-4b-sft
"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
state = json.load(open(run / "trainer_state.json"))
ev = {x["step"]: x["eval_loss"] for x in state["log_history"] if "eval_loss" in x}
saved = [s for s in ev if (run / f"checkpoint-{s}").exists()]
if not saved:
    sys.exit(f"no saved checkpoint with an eval loss in {run}")
print(run / f"checkpoint-{min(saved, key=ev.get)}")
