# Hawkeye: Reading Illegible Text with Vision Language Models

**Brian Tang, Kang G. Shin** — University of Michigan

[Project page](https://byron123t.github.io/hawkeye/) ·
Paper (coming soon) ·
[Hawkeye-4B](https://huggingface.co/byron123t/Hawkeye-4B) ·
[Hawkeye-8B](https://huggingface.co/byron123t/Hawkeye-8B) ·
[BSTRB dataset](https://huggingface.co/datasets/byron123t/BSTRB)

Vision-language models read clean scene text well but fail on text that is small, blurred or far
away, which is the normal case for smart glasses and other egocentric cameras. This repository
contains:

- **BSTRB** (Blurry Scene Text Recognition Benchmark) tools: loading, scoring and the statistics
  used in the paper;
- **Hawkeye 4B and 8B**: Qwen3-VL-4B/8B-Instruct fine-tuned on BSTRB video frames (supervised
  fine-tuning, then GRPO with a bag-of-words F1 reward), with inference code and the full training
  recipe;
- the **attention analyses** from the paper's appendix (reading traces, word-level grounding).

On the held-out BSTRB test set (BoW F1, paired bootstrap over videos, Holm-corrected):

| | Qwen3-VL base | Hawkeye | gain |
|---|---|---|---|
| 4B, video frames (1,331 frames, 224 videos) | .627 | .728 | **+10.1 pp** |
| 8B, video frames | .675 | .744 | **+6.9 pp** |
| 4B, photos (241) | .556 | .616 | **+6.0 pp** |
| 8B, photos | .602 | .655 | **+5.3 pp** |

The gains grow as text gets smaller, and neither size regresses significantly in BoW F1 on any of
10 general OCR benchmarks.

## Install

```bash
git clone https://github.com/byron123t/hawkeye && cd hawkeye
pip install -e ".[infer]"          # inference + evaluation
pip install -r train/requirements.txt      # training (adds peft, deepspeed, flash-attn)
pip install -r analysis/requirements.txt   # attention analysis and figures
```

Tested with Python 3.10, torch 2.6.0 (CUDA 12.4) and transformers 4.57.0.

## Quickstart

```python
from hawkeye import Hawkeye

reader = Hawkeye("byron123t/Hawkeye-4B")      # or "byron123t/Hawkeye-8B"
print(reader.read("photo.jpg"))
```

or from the shell: `hawkeye-read photo.jpg --model 8b`.

**Input contract.** Hawkeye was trained and evaluated on images upscaled **x3 with bicubic
interpolation**, then capped by the processor at 1,003,520 pixels (min 65,536); one user turn
(image, then prompt), no system prompt, greedy decoding, up to 2,048 new tokens. `Hawkeye.read`
and `hawkeye.transcribe` do all of this; if you call `transformers` directly, upscale the image
yourself first (`hawkeye.upscale(image)`).

## The BSTRB dataset

Gated on Hugging Face ([byron123t/BSTRB](https://huggingface.co/datasets/byron123t/BSTRB)): accept
the data-use agreement on the dataset page, then `huggingface-cli login`. Download a local copy and
point the tools at it:

```bash
huggingface-cli download byron123t/BSTRB --repo-type dataset --local-dir BSTRB
export BSTRB_ROOT=$PWD/BSTRB
```

| split | images | video frames (videos) | photos | labels |
|---|---|---|---|---|
| train | 7,434 | 6,734 (1,133) | 700 | GPT-5 transcriptions |
| val | 369 | 333 (56) | 36 | human-verified |
| test | 1,572 | 1,331 (224) | 241 | human-verified |

Each video contributes 5–6 frames sampled as the camera moves away from the object, so text
shrinks from frame to frame; all frames of a video share its transcript. No video appears in more
than one split. `hawkeye.data.load_split(root, "test")` returns the rows with absolute image paths.

## Reproduce the evaluation

```bash
# predictions (one shard per GPU), then scores, then paired comparisons
python eval/run_sharded.py --gpus 0,1,2,3 --model byron123t/Hawkeye-4B --name hawkeye-4b --out results/
python eval/run_sharded.py --gpus 0,1,2,3 --model Qwen/Qwen3-VL-4B-Instruct --name qwen3-vl-4b \
    --upscale 1 --out results/                           # the plain base, as in the paper's table
python eval/score.py results/                            # table with 95% CIs per subset
python eval/compare.py results/ --pairs hawkeye-4b:qwen3-vl-4b
```

`--upscale 3` on a base model gives "base x3", which isolates what training adds beyond the
upscaling (+6.8 pp for 4B on video frames). API models go through any OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=...
python eval/run_api.py --model gpt-4o --out results/
```

API runs use a fixed transcription-only system prompt (see `eval/run_api.py`); local models get no
system prompt. Scoring normalizes case, punctuation and whitespace and is identical for all models
(`hawkeye/metrics.py`).

## Reproduce training

```bash
cd train
BSTRB_ROOT=/path/to/BSTRB bash scripts/train_4b.sh     # ~4 h on 4 x 48 GB GPUs (SFT 1.4 h, GRPO 2.7 h)
BSTRB_ROOT=/path/to/BSTRB bash scripts/train_8b.sh     # ~9 h (SFT 5 h with ZeRO-3 CPU offload, GRPO 3.5 h)
```

Each script runs the two stages of the recipe:

1. **SFT** (`scripts/sft.sh`, `qwenvl/train/train_sft.py`): full fine-tuning of the LLM, projector
   and vision tower on the 6,734 BSTRB training frames, images upscaled x3; 420 steps, effective
   batch 32, learning rates 1e-6 (LLM) / 1e-5 (projector) / 1e-6 (vision tower), cosine schedule,
   3% warm-up. The checkpoint with the lowest loss on the validation split is kept
   (`best_checkpoint.py`; step 400 in the paper).
2. **GRPO** (`scripts/grpo.sh`, `qwenvl/train/grpo.py`): from that checkpoint, LoRA (r 64, alpha 128)
   on the LLM's q/k/v/o projections, 8 samples per prompt, reward = BoW F1 of the sample against the
   label, KL weight 0.02 to the SFT model, learning rate 2e-5, 300 steps x 8 prompts. The script
   writes the merged model.

The released checkpoints were trained with seed 1234. A rerun will not be bit-identical (data
order follows the released file order and ids). For reference, three 4B runs of this recipe
(different seeds) were within 0.4 pp of each other in video BoW F1 gain over the plain base.
The 8B SFT stage uses DeepSpeed ZeRO-3 with CPU optimizer offload (`scripts/zero3_offload.json`,
`DS_CPU_ADAM=1`), which needs ample CPU RAM; if DeepSpeed's CPU Adam fails to build because the
system `nvcc` differs from torch's CUDA version, set `DS_SKIP_CUDA_CHECK=1`.

## Attention analysis

```bash
# per-token attention maps for both arms on the same 160 held-out frames (GPU, eager attention)
python analysis/extract_attention.py --model Qwen/Qwen3-VL-4B-Instruct --arm base \
    --split test --kinds video_frame --sample 160 --seed 13 --out attn/4b
python analysis/extract_attention.py --model byron123t/Hawkeye-4B --arm hawkeye \
    --split test --kinds video_frame --sample 160 --seed 13 --out attn/4b

python analysis/attention_stats.py attn/4b                       # sink mass, entropy, top-k mass
python analysis/text_regions.py --runs attn/4b --out text_regions.json
python analysis/reading_metrics.py --regions text_regions.json --run 4b:attn/4b --out reading.json
```

`analysis/ocr_words.py` + `analysis/grounding.py` measure word-level grounding (is the attention
peak on the word being written?); `plot_reading_traces.py` and `plot_filmstrip.py` draw the
reading-trace and per-step figures. All maps exclude attention-sink patches (`analysis/common.py`).

## Repository map

```
hawkeye/            pip-installable package
  inference.py        load / transcribe / Hawkeye class (x3 upscale contract)
  metrics.py          WIR, WER, BoW P/R/F1, aligned F1, CER; degenerate-output filters
  stats.py            unit-clustered paired bootstrap, Holm correction
  data.py             BSTRB loader (Hugging Face layout)
  cli.py              hawkeye-read
eval/               run_local.py, run_sharded.py, run_api.py, score.py, compare.py
train/              prepare_data.py, best_checkpoint.py, merge_lora.py,
                    qwenvl/ (fine-tuning framework, SFT and GRPO), scripts/ (launchers, DeepSpeed configs)
analysis/           extract_attention.py, attention_stats.py, text_regions.py, ocr_words.py,
                    grounding.py, reading_metrics.py, plot_reading_traces.py, plot_filmstrip.py
tests/              known-value tests for metrics and statistics (pytest)
docs/               project page (GitHub Pages)
```

## Citation

```bibtex
@misc{tang2026hawkeye,
  title  = {Hawkeye4B: Reading Illegible Text with Vision Language Models},
  author = {Tang, Brian and Shin, Kang G.},
  year   = {2026},
  note   = {Paper link coming soon}
}
```

## License

Code and model weights: Apache-2.0 (see `LICENSE` and `NOTICE`; the fine-tuning framework under
`train/qwenvl/` is derived from the Qwen team's qwen-vl-finetune, and the weights from Qwen3-VL).
The BSTRB dataset is released separately under CC BY-NC 4.0 with gated access and a data-use
agreement; see its dataset card.

## Acknowledgements

Built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and its fine-tuning framework. Text
detection in the attention analysis uses [EasyOCR](https://github.com/JaidedAI/EasyOCR).
