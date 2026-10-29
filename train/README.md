# Training Hawkeye

Run everything from this directory. `scripts/train_4b.sh` and `scripts/train_8b.sh` reproduce the
released models end to end; the pieces are:

| file | what it does |
|---|---|
| `prepare_data.py` | BSTRB (Hugging Face layout) -> `data/bstrb_train_video.json`, `data/bstrb_val.json` |
| `scripts/sft.sh` -> `qwenvl/train/train_sft.py` | stage 1: supervised fine-tuning (DeepSpeed ZeRO-3) |
| `best_checkpoint.py` | the SFT checkpoint with the lowest validation loss |
| `scripts/grpo.sh` -> `qwenvl/train/grpo.py` | stage 2: GRPO with a BoW-F1 reward (LoRA, plain torch.distributed); writes merged weights |
| `merge_lora.py` | merge a LoRA SFT run into full weights (not used by the default recipe) |
| `scripts/zero3.json`, `scripts/zero3_offload.json` | DeepSpeed configs (4B, and 8B with CPU optimizer offload) |

All launcher settings are environment variables with the recipe as default, e.g.

```bash
MODEL=Qwen/Qwen3-VL-4B-Instruct OUTPUT_DIR=runs/my-sft NPROC_PER_NODE=2 bash scripts/sft.sh
INIT=runs/my-sft/checkpoint-400 PROCESSOR=Qwen/Qwen3-VL-4B-Instruct OUTPUT_DIR=runs/my-grpo bash scripts/grpo.sh
```

The effective SFT batch stays 32 whatever the GPU count (gradient accumulation adjusts).
`TRAIN_UPSCALE=3` applies the x3 bicubic upscale used at inference to every BSTRB image during
training and validation. `REWARD_BETA` < 1 turns the GRPO reward into F-beta (more weight on
precision). Training JSON locations can be overridden with `BSTRB_TRAIN_JSON` / `BSTRB_VAL_JSON`,
and `DATASETS` / `EVAL_DATASETS` also accept JSON paths directly.

`qwenvl/` is derived from the Qwen team's qwen-vl-finetune (Apache-2.0); see `../NOTICE`.
