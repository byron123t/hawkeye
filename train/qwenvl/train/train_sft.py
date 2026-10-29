"""
Stage 1 of the Hawkeye recipe: supervised fine-tuning of Qwen3-VL on BSTRB video frames.

Built on qwen-vl-finetune's train_qwen.py (Qwen team, Apache-2.0). Differences from it:

  1. The random seed is set BEFORE the data module is built. LazySupervisedDataset shuffles the
     data when it is constructed; upstream, each DDP rank built a different permutation while the
     DistributedSampler assumed they agree.
  2. --eval_dataset_use builds a validation set (upstream hardcodes eval_dataset=None), so the
     best checkpoint can be selected on validation loss (train/best_checkpoint.py).
  3. --train_upscale bicubic-upscales training and validation images by the inference factor (x3).
  4. The architecture is read from config.json (upstream infers MoE from the path string).
  5. DS_CPU_ADAM=1 rebuilds the optimizer as DeepSpeed's CPU Adam while keeping the separate
     LLM / projector / vision-tower learning rates (used for the 8B run with ZeRO-3 offload).
  6. Optional LoRA on the LLM attention (--lora_enable); not used for Hawkeye 4B/8B.

Launch with train/scripts/sft.sh.
"""

import json
import logging
import os
import pathlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class  # noqa: E402
from transformers import AutoProcessor, Trainer  # noqa: E402

from qwenvl.data.data_processor import LazySupervisedDataset, make_supervised_data_module  # noqa: E402
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments  # noqa: E402
from qwenvl.train.train_qwen import safe_save_model_for_hf_trainer, set_model  # noqa: E402

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class SFTDataArguments(DataArguments):
    eval_dataset_use: str = field(default="")
    train_upscale: float = field(default=1.0)
    train_upscale_sources: str = field(default="res_video,res_photo")


class SFTTrainer(Trainer):
    def create_optimizer(self):
        opt = super().create_optimizer()
        if os.environ.get("DS_CPU_ADAM") == "1" and hasattr(opt, "param_groups"):
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            if not isinstance(opt, DeepSpeedCPUAdam):
                # ZeRO-Offload needs DeepSpeed's CPU Adam. Rebuild it from the trainer's groups so
                # the LLM / projector / vision learning rates survive (a DeepSpeed-config optimizer
                # would collapse them into one). Convert once: HF calls create_optimizer again
                # after the scheduler has set every group's lr to its warmup value.
                groups = [{k: g[k] for k in ("params", "lr", "weight_decay") if k in g}
                          for g in opt.param_groups if g.get("params")]
                self.optimizer = opt = DeepSpeedCPUAdam(
                    groups, lr=self.args.learning_rate, betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay, adamw_mode=True)
        if self.args.local_rank in (-1, 0) and hasattr(opt, "param_groups"):
            print(f"[sft] optimizer groups (lr, n_tensors): "
                  f"{[(g.get('lr'), len(g.get('params', []))) for g in opt.param_groups]}")
        return opt


def train(attn_implementation="flash_attention_2"):
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, SFTDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # (1) seed before the data module shuffles
    transformers.set_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    # (3) x3 upscale, as at inference
    if data_args.train_upscale != 1.0:
        from qwenvl.data.upscale import install_upscale
        srcs = set(filter(None, data_args.train_upscale_sources.split(",")))
        install_upscale(data_args.train_upscale, srcs)
        rank0_print(f"[sft] bicubic upscale x{data_args.train_upscale} on {sorted(srcs)}")

    # (4) architecture from config.json
    cfg = json.load(open(Path(model_args.model_name_or_path) / "config.json")) \
        if (Path(model_args.model_name_or_path) / "config.json").exists() \
        else transformers.AutoConfig.from_pretrained(model_args.model_name_or_path).to_dict()
    arch = (cfg.get("architectures") or [""])[0]
    cls = getattr(transformers, arch)
    model = cls.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,
                                attn_implementation=attn_implementation,
                                dtype=(torch.bfloat16 if training_args.bf16 else None))
    a = arch.lower()
    data_args.model_type = "qwen3vl" if "qwen3vl" in a else "qwen2.5vl" if "qwen2_5" in a else "qwen2vl"
    rank0_print(f"[sft] {model_args.model_name_or_path}: {arch}")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if training_args.lora_enable:
        # (6) LoRA on the LLM attention; an unfrozen ViT/merger is trained in full via modules_to_save
        from peft import LoraConfig, TaskType, get_peft_model
        for p in model.parameters():
            p.requires_grad = False
        full = ["visual"] if model_args.tune_mm_vision else (["merger"] if model_args.tune_mm_mlp else [])
        model = get_peft_model(model, LoraConfig(
            r=training_args.lora_r or 64, lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=full or None, bias="none", task_type=TaskType.CAUSAL_LM))
    else:
        set_model(model_args, model)

    data_module = make_supervised_data_module(processor, data_args=data_args)
    # (2) validation set for checkpoint selection
    if data_args.eval_dataset_use:
        import copy
        eval_args = copy.deepcopy(data_args)
        eval_args.dataset_use = data_args.eval_dataset_use
        data_module["eval_dataset"] = LazySupervisedDataset(processor, data_args=eval_args)
        rank0_print(f"[sft] eval dataset: {len(data_module['eval_dataset'])} samples")

    trainer = SFTTrainer(model=model, processing_class=processor.tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
