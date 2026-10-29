# Evaluation

1. **Predictions.** `run_local.py` (one GPU) or `run_sharded.py --gpus 0,1,...` for local models;
   `run_api.py` for any OpenAI-compatible endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`). Both write
   `<out>/predictions_<name>.jsonl` and resume if interrupted.
2. **Scores.** `score.py <out>` scores every predictions file (`scores_<name>.jsonl`, `summary.json`)
   and prints item means with 95% CIs for all items, video frames, frames with text <= 20 px and
   > 20 px, and photos.
3. **Comparisons.** `compare.py <out> --pairs A:B ...` runs paired bootstrap tests (resampling
   videos and photos) with Holm correction across all tests printed.

Settings used in the paper (the defaults of `run_local.py` unless noted): the transcription-only
system prompt for every model (local and API; `--no_system_prompt` to disable), instruction text
before the image, images JPEG re-encoded with the longest side <= 2,048 px; Hawkeye at
`--upscale 3`, bases at `--upscale 1` (plain) and `--upscale 3` ("base x3"); greedy decoding,
`--max_new_tokens 4096`, `--attn sdpa`; test split (1,572 images).
