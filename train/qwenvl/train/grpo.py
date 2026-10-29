"""
Stage 2 of the Hawkeye recipe: GRPO on BSTRB video frames with a bag-of-words F1 reward.

SFT copies its targets, and the BSTRB training labels (GPT-5 transcriptions) run short, so SFT
alone teaches the model to say less (precision up, recall down). Here the model is rewarded for
what it reads instead: for each training frame, G readings are sampled, each is scored by
bag-of-words F1 against the frame's label (hawkeye.metrics.score, the benchmark's own
scorer), and the policy moves toward the readings that score above their group's mean.
F1 punishes both missed words and hallucinated ones, so neither shorter nor longer output
is rewarded for its own sake.

  * LoRA on the LLM attention (q/k/v/o); ViT and merger frozen. The reference policy for
    the KL term is the same model with the adapter switched off, so no second copy.
  * Plain torch.distributed: each rank samples and backprops its own groups, LoRA grads
    are all-reduced before the step. No DeepSpeed (LoRA state is tiny).
  * Loss: Dr. GRPO form. Advantage = reward - group mean, divided by the group std;
    token log-ratio terms summed and divided by a constant (max_new_tokens), so long and
    short completions are not reweighted by their length. Groups where every sample got
    the same reward carry no signal and are skipped.
  * Output: adapters every --save_steps, and at the end the merged full model in
    --output_dir (config.json + processor), evaluable like any fine-tune.

Launch with train/scripts/grpo.sh, or:
torchrun --nproc_per_node 4 qwenvl/train/grpo.py --model <SFT checkpoint> --processor <base model> \
    --data data/bstrb_train_video.json --output_dir runs/grpo --upscale 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for the hawkeye package
from hawkeye.metrics import score as score_sample  # noqa: E402

IM_END = 151645


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default=None,
                    help="processor/tokenizer source; defaults to --model (SFT checkpoints lack one)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompts_per_rank", type=int, default=2)
    ap.add_argument("--micro", type=int, default=4, help="completions per backward pass")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--min_pixels", type=int, default=65536)
    ap.add_argument("--max_pixels", type=int, default=1003520)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reward_beta", type=float, default=1.0,
                    help="F-beta reward; <1 weights precision more (penalises guessing on illegible frames)")
    ap.add_argument("--gpus_per_rank", type=int, default=1,
                    help=">1 shards each rank's model over that many GPUs (for models too large for one GPU)")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="bicubic per-side factor before the processor, as eval --upscale")
    return ap.parse_args()


def main():
    a = args_()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", rank))
    first = lr_ * a.gpus_per_rank          # this rank owns GPUs first .. first+gpus_per_rank-1
    torch.cuda.set_device(first)
    dev = torch.device("cuda", first)
    random.seed(a.seed + rank)
    torch.manual_seed(a.seed + rank)
    out = Path(a.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    from peft import LoraConfig, get_peft_model
    import peft.tuners.lora.awq as _peft_awq
    import transformers
    from transformers import AutoConfig, AutoProcessor
    _peft_awq.is_auto_awq_available = lambda: False  # no AWQ layers; avoids an autoawq/transformers import clash

    proc = AutoProcessor.from_pretrained(a.processor or a.model)
    ip = proc.image_processor
    ip.min_pixels, ip.max_pixels = a.min_pixels, a.max_pixels
    ip.size["shortest_edge"], ip.size["longest_edge"] = a.min_pixels, a.max_pixels

    cls = getattr(transformers, AutoConfig.from_pretrained(a.model).architectures[0])  # dense or MoE
    if a.gpus_per_rank > 1:
        mem = {g: "40GiB" for g in range(first, first + a.gpus_per_rank)}
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                    device_map="auto", max_memory=mem)
    else:
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))  # LLM only; ViT uses qkv/proj
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"[grpo] trainable {sum(p.numel() for p in params)/1e6:.1f}M params, world={world}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / a.steps))))

    rows = [r for r in json.load(open(a.data)) if r["conversations"][-1]["value"].strip()]
    random.Random(a.seed).shuffle(rows)  # same order on every rank, then strided
    rows = rows[rank::world]
    cursor = 0
    hist = []

    def logp_of(inp, comp, mask):
        """Per-token log-probs of `comp` (B, T) given the prompt; logits only for the tail."""
        T = comp.shape[1]
        o = model(**inp, logits_to_keep=T + 1, use_cache=False)
        lg = o.logits[:, :-1].float()
        return torch.gather(F.log_softmax(lg, -1), 2, comp[..., None]).squeeze(-1) * mask

    for step in range(a.steps):
        t0 = time.time()
        stats = dict(reward=[], length=[], kl=[], skipped=0)
        opt.zero_grad(set_to_none=True)
        norm = a.max_new_tokens * a.prompts_per_rank * a.group
        for _ in range(a.prompts_per_rank):
            r = rows[cursor % len(rows)]
            cursor += 1
            prompt = r["conversations"][0]["value"].replace("<image>", "").strip()
            label = r["conversations"][-1]["value"]
            img = r["image"]
            if a.upscale != 1.0:
                from PIL import Image
                im = Image.open(img).convert("RGB")
                img = im.resize((max(1, round(im.size[0] * a.upscale)),
                                 max(1, round(im.size[1] * a.upscale))), Image.BICUBIC)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": prompt}]}]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(dev)
            L = inp["input_ids"].shape[1]
            model.eval()
            with torch.no_grad():
                gen = model.generate(**inp, do_sample=True, temperature=a.temperature, top_p=a.top_p,
                                     top_k=a.top_k, max_new_tokens=a.max_new_tokens,
                                     num_return_sequences=a.group, use_cache=True)
            model.train()
            comp = gen[:, L:]
            # mask up to and including the first <|im_end|>; pad after it
            is_end = (comp == IM_END).int()
            after = (torch.cumsum(is_end, 1) - is_end) > 0
            mask = (~after).float()
            texts = proc.batch_decode(comp.masked_fill(after, proc.tokenizer.pad_token_id),
                                      skip_special_tokens=True)
            def fbeta(sc, b=a.reward_beta):
                p, r = sc.bow_precision, sc.bow_recall
                return 0.0 if p + r == 0 else (1 + b * b) * p * r / (b * b * p + r)
            rew = torch.tensor([fbeta(score_sample(label, t)) for t in texts], device=dev)
            stats["reward"] += rew.tolist()
            stats["length"] += mask.sum(1).tolist()
            if rew.std() < 1e-6:
                stats["skipped"] += 1
                continue
            adv = (rew - rew.mean()) / (rew.std() + 1e-4)
            for i in range(0, a.group, a.micro):
                c, m, ad = comp[i:i + a.micro], mask[i:i + a.micro], adv[i:i + a.micro]
                n = c.shape[0]
                full = dict(
                    input_ids=torch.cat([inp["input_ids"].expand(n, -1), c], 1),
                    attention_mask=torch.cat([inp["attention_mask"].expand(n, -1), m.long()], 1),
                    pixel_values=inp["pixel_values"].repeat(n, 1),
                    image_grid_thw=inp["image_grid_thw"].repeat(n, 1))
                with torch.no_grad(), model.disable_adapter():
                    ref = logp_of(full, c, m)
                lp = logp_of(full, c, m)
                d = ref - lp
                kl = (torch.exp(d) - d - 1) * m
                pg = -ad[:, None] * torch.exp(lp - lp.detach()) * m
                loss = (pg + a.beta * kl).sum() / norm
                loss.backward()
                stats["kl"].append(float((kl.sum() / m.sum().clamp(min=1)).detach()))
        # average LoRA grads over ranks (ranks with skipped groups add zeros). A sharded model
        # keeps grads on several GPUs while NCCL is bound to this rank's first one, so the
        # grads are flattened onto it, reduced once, and copied back.
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = torch.cat([p.grad.reshape(-1).to(dev, torch.float32) for p in params])
        dist.all_reduce(flat)
        flat /= world
        o = 0
        for p in params:
            n = p.numel()
            p.grad.copy_(flat[o:o + n].view_as(p.grad).to(p.grad.device, p.grad.dtype))
            o += n
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        opt.step()
        sched.step()

        agg = torch.tensor([sum(stats["reward"]), len(stats["reward"]), sum(stats["length"]),
                            sum(stats["kl"]), len(stats["kl"]), stats["skipped"]], device=dev)
        dist.all_reduce(agg)
        rec = dict(step=step + 1, reward=round(float(agg[0] / agg[1]), 4),
                   length=round(float(agg[2] / agg[1]), 1),
                   kl=round(float(agg[3] / max(agg[4], 1)), 5), skipped_groups=int(agg[5]),
                   grad_norm=round(float(gn), 4), lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1))
        hist.append(rec)
        log(f"[grpo] {json.dumps(rec)}")
        if rank == 0:
            (out / "grpo_log.json").write_text(json.dumps(hist, indent=0))
            if (step + 1) % a.save_steps == 0 and step + 1 < a.steps:
                model.save_pretrained(out / f"checkpoint-{step + 1}")
        dist.barrier()

    if rank == 0:
        model.save_pretrained(out / "adapter")
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
        proc.save_pretrained(out)
        log(f"[grpo] merged model written to {out}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
