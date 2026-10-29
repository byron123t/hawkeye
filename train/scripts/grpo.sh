#!/bin/bash
# Stage 2 (GRPO) launcher. Defaults are the Hawkeye recipe.
# Run from train/:  INIT=<best SFT checkpoint> PROCESSOR=<base model> OUTPUT_DIR=runs/hawkeye-4b bash scripts/grpo.sh
set -euo pipefail
INIT=${INIT:?set INIT (the SFT checkpoint to start from)}
PROCESSOR=${PROCESSOR:?set PROCESSOR (the base model: SFT checkpoints carry no processor)}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
DATA=${DATA:-data/bstrb_train_video.json}
[ -e "$OUTPUT_DIR" ] && { echo "REFUSING: $OUTPUT_DIR already exists"; exit 1; }
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=${SEED:-1234}

# 4 ranks x 2 prompts x 8 samples per step, 300 steps: 2,400 prompts in total.
torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" --master_port="${MASTER_PORT:-28500}" \
  qwenvl/train/grpo.py --model "$INIT" --processor "$PROCESSOR" --data "$DATA" --output_dir "$OUTPUT_DIR" \
  --steps "${MAX_STEPS:-300}" --seed "${SEED:-1234}" --lr "${LR:-2e-5}" --beta "${BETA:-0.02}" \
  --group "${GROUP:-8}" --prompts_per_rank "${PROMPTS:-2}" --max_new_tokens "${MAX_NEW:-512}" \
  --upscale "${TRAIN_UPSCALE:-3}" --reward_beta "${REWARD_BETA:-1.0}" --gpus_per_rank "${GPUS_PER_RANK:-1}"
# The merged model is written to OUTPUT_DIR (config.json + safetensors + processor);
# the LoRA adapter to OUTPUT_DIR/adapter.
