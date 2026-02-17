"""Data loading and tokenization."""

from .loader import (
    create_model,
    load_model,
    load_weights_from_hf,
    create_config_from_hf,
    load_tokenizer,
    resolve_model_source,
    download_model,
    WeightLoadingError,
)
from .registry import MODEL_REGISTRY, get_quantized_dir

# VLM loading (lazy import to avoid requiring VLM-specific transformers symbols
# during LLM-only usage / tests in older environments).
def create_vlm_config_from_hf(*args, **kwargs):
    from .vlm_loader import create_vlm_config_from_hf as _impl
    return _impl(*args, **kwargs)


def build_llava_weight_mapping(*args, **kwargs):
    from .vlm_loader import build_llava_weight_mapping as _impl
    return _impl(*args, **kwargs)


def load_llava_weights(*args, **kwargs):
    from .vlm_loader import load_llava_weights as _impl
    return _impl(*args, **kwargs)


def load_llava_model(*args, **kwargs):
    from .vlm_loader import load_llava_model as _impl
    return _impl(*args, **kwargs)


def download_llava_model(*args, **kwargs):
    from .vlm_loader import download_llava_model as _impl
    return _impl(*args, **kwargs)

__all__ = [
    # LLM loading
    'create_model',
    'load_model',
    'load_weights_from_hf',
    'create_config_from_hf',
    'load_tokenizer',
    'resolve_model_source',
    'download_model',
    'WeightLoadingError',
    'MODEL_REGISTRY',
    'get_quantized_dir',
    # VLM loading
    'create_vlm_config_from_hf',
    'build_llava_weight_mapping',
    'load_llava_weights',
    'load_llava_model',
    'download_llava_model',
]
