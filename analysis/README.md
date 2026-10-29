# Attention analysis

| step | script | output |
|---|---|---|
| per-token attention maps, one model per call (`--arm base` / `--arm hawkeye`) | `extract_attention.py` | `<out>/<sample>/<arm>/metadata.json`, `heatmaps/*.npy` |
| spread of attention: sink mass, entropy, top-1%/5% mass, peak movement | `attention_stats.py` | paired base vs Hawkeye, d_z |
| text regions (detector boxes) | `text_regions.py` | `text_regions.json` |
| attention on text, regions visited, coverage at matched token budget | `reading_metrics.py` | paired, clustered, Holm |
| word boxes with text (OCR) | `ocr_words.py` | `ocr_words.json` |
| is the peak on the word being written? | `grounding.py` | per-image accuracy vs chance |
| figures | `plot_reading_traces.py`, `plot_filmstrip.py` | `.jpg` |

Both arms of a comparison must be extracted with the same `--sample`/`--seed` (or the same
`--images`) into the same output directory. Maps are the attention of each generated token to the
image tokens, last 4 layers, mean over heads; every measure excludes attention-sink patches
(`common.sink_mask`).
