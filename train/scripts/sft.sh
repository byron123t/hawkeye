#!/bin/bash
# Stage 1 (SFT) launcher. Every knob is an environment variable; defaults are the Hawkeye recipe.
# Run from train/:  MODEL=Qwen/Qwen3-VL-4B-Instruct OUTPUT_DIR=runs/hawkeye-4b-sft bash scripts/sft.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL (base model id or path)}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
DATASETS=${DATASETS:-bstrb_train_video}          # see qwenvl/data/__init__.py
EVAL_DATASETS=${EVAL_DATASETS:-bstrb_val}
MAX_STEPS=${MAX_STEPS:-420}
SEED=${SEED:-1234}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PER_DEV_BS=${PER_DEV_BS:-1}
EFF_BATCH=${EFF_BATCH:-32}                       # effective batch held at 32 whatever the GPU count
GRAD_ACCUM=$(( EFF_BATCH / (NPROC_PER_NODE * PER_DEV_BS) ))
[ $(( GRAD_ACCUM * NPROC_PER_NODE * PER_DEV_BS )) -eq "$EFF_BATCH" ] || { echo "EFF_BATCH not divisible"; exit 1; }

# learning rates: LLM 1e-6, projector 1e-5, vision tower 1e-6 (ViT unfrozen).
# vision_tower_lr only takes effect when mm_projector_lr is also set (trainer.py).
LR=${LR:-1e-6}; PROJ_LR=${PROJ_LR:-1e-5}; VISION_LR=${VISION_LR:-1e-6}
TUNE_VISION=${TUNE_VISION:-True}
TRAIN_UPSCALE=${TRAIN_UPSCALE:-3}                # bicubic x3, as at inference
MIN_PIXELS=${MIN_PIXELS:-65536}; MAX_PIXELS=${MAX_PIXELS:-1003520}
DEEPSPEED=${DEEPSPEED:-./scripts/zero3.json}

# train_sft.py resumes from any checkpoint-* in OUTPUT_DIR; saved checkpoints hold no optimizer
# state (save_only_model), so refuse to reuse a directory.
[ -e "$OUTPUT_DIR" ] && { echo "REFUSING: $OUTPUT_DIR already exists"; exit 1; }
mkdir -p "$(dirname "$OUTPUT_DIR")"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=$SEED

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29501}" \
  qwenvl/train/train_sft.py \
  --deepspeed "$DEEPSPEED" \
  --model_name_or_path "$MODEL" \
  --dataset_use "$DATASETS" --eval_dataset_use "$EVAL_DATASETS" \
  --data_flatten True \
  --tune_mm_vision "$TUNE_VISION" --tune_mm_mlp True --tune_mm_llm True \
  --train_upscale "$TRAIN_UPSCALE" \
  --bf16 \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEV_BS" --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --min_pixels "$MIN_PIXELS" --max_pixels "$MAX_PIXELS" \
  --learning_rate "$LR" --mm_projector_lr "$PROJ_LR" --vision_tower_lr "$VISION_LR" \
  --optim adamw_torch --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --eval_strategy steps --eval_steps 100 --eval_on_start True \
  --save_strategy steps --save_steps 100 --save_total_limit "${SAVE_LIMIT:-11}" --save_only_model True \
  --logging_steps 10 \
  --model_max_length 16384 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --seed "$SEED" --data_seed "$SEED" \
  --report_to "${REPORT_TO:-none}"
