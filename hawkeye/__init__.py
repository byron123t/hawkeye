"""Hawkeye: reading degraded scene text with fine-tuned Qwen3-VL models.

Light imports only; the model code loads torch/transformers lazily.
"""
from .inference import DEFAULT_PROMPT, MODELS, UPSCALE, Hawkeye, load, transcribe, upscale
from .metrics import aggregate, score

__version__ = "1.0.0"
__all__ = ["Hawkeye", "load", "transcribe", "upscale", "score", "aggregate", "DEFAULT_PROMPT",
           "MODELS", "UPSCALE", "__version__"]
