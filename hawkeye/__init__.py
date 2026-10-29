"""Hawkeye: reading degraded scene text with fine-tuned Qwen3-VL models.

Light imports only; the model code loads torch/transformers lazily.
"""
from .inference import DEFAULT_PROMPT, MODELS, UPSCALE, Hawkeye, load, transcribe, upscale
from .metrics import aggregate, score
from .protocol import SYSTEM_PROMPT

__version__ = "1.0.0"
__all__ = ["Hawkeye", "load", "transcribe", "upscale", "score", "aggregate", "DEFAULT_PROMPT",
           "MODELS", "UPSCALE", "SYSTEM_PROMPT", "__version__"]
