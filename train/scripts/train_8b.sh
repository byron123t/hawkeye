#!/bin/bash
# Reproduce Hawkeye 8B: the 4B recipe on Qwen3-VL-8B-Instruct. The only change is memory:
# full SFT runs under ZeRO-3 with the optimizer offloaded to CPU (DeepSpeed CPU Adam, keeping the
# three learning-rate groups). GRPO still fits one model copy per GPU.
# Run from train/:  BSTRB_ROOT=/path/to/BSTRB bash scripts/train_8b.sh
set -euo pipefail
BASE=${BASE:-Qwen/Qwen3-VL-8B-Instruct}
RUNS=${RUNS:-runs}
python prepare_data.py --data "${BSTRB_ROOT:-byron123t/BSTRB}" --out data
MODEL=$BASE OUTPUT_DIR=$RUNS/hawkeye-8b-sft DEEPSPEED=./scripts/zero3_offload.json \
  DS_CPU_ADAM=1 SAVE_LIMIT=2 bash scripts/sft.sh
INIT=$(python best_checkpoint.py "$RUNS/hawkeye-8b-sft")
echo "GRPO init: $INIT"
INIT=$INIT PROCESSOR=$BASE OUTPUT_DIR=$RUNS/hawkeye-8b bash scripts/grpo.sh
echo "Hawkeye 8B written to $RUNS/hawkeye-8b"
