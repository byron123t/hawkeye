#!/bin/bash
# Reproduce Hawkeye 4B end to end on 4 GPUs (80 GB class; the paper used 4 x 48 GB).
#   1. BSTRB -> training/validation JSON          (train/prepare_data.py)
#   2. SFT: Qwen3-VL-4B-Instruct, video frames, 420 steps, effective batch 32, x3 upscale
#   3. pick the SFT checkpoint with the lowest validation loss (checkpoint-400 in the paper)
#   4. GRPO from it: BoW-F1 reward, LoRA r64 on q/k/v/o, 300 steps, x3 upscale
# Run from train/:  BSTRB_ROOT=/path/to/BSTRB bash scripts/train_4b.sh
set -euo pipefail
BASE=${BASE:-Qwen/Qwen3-VL-4B-Instruct}
RUNS=${RUNS:-runs}
python prepare_data.py --data "${BSTRB_ROOT:-byron123t/BSTRB}" --out data
MODEL=$BASE OUTPUT_DIR=$RUNS/hawkeye-4b-sft bash scripts/sft.sh
INIT=$(python best_checkpoint.py "$RUNS/hawkeye-4b-sft")
echo "GRPO init: $INIT"
INIT=$INIT PROCESSOR=$BASE OUTPUT_DIR=$RUNS/hawkeye-4b bash scripts/grpo.sh
echo "Hawkeye 4B written to $RUNS/hawkeye-4b"
