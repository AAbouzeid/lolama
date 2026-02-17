"""Model registry for supported models."""

from __future__ import annotations

from pathlib import Path


MODEL_REGISTRY: dict[str, dict[str, str | bool]] = {
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "folder": "tinyllama-1.1b",
        "trust_remote_code": False,
        "description": "TinyLlama 1.1B Chat — small, fast, instruction-tuned",
        "params": "1.1B",
        "download_size": "2.2 GB",
    },
    "open_llama_3b": {
        "hf_name": "openlm-research/open_llama_3b_v2",
        "folder": "open-llama-3b",
        "trust_remote_code": False,
        "description": "OpenLLaMA 3B v2 — compact base model",
        "params": "3B",
        "download_size": "6.4 GB",
    },
    "open_llama_7b": {
        "hf_name": "openlm-research/open_llama_7b_v2",
        "folder": "open-llama-7b",
        "trust_remote_code": False,
        "description": "OpenLLaMA 7B v2 — full-size base model",
        "params": "7B",
        "download_size": "13.5 GB",
    },
    "llama7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "folder": "llama-7b",
        "trust_remote_code": False,
        "description": "LLaMA 2 7B — Meta's base model (gated, requires access)",
        "params": "7B",
        "download_size": "13.5 GB",
    },
    # Vision-Language Models (VLMs)
    "llava-1.5-7b": {
        "hf_name": "llava-hf/llava-1.5-7b-hf",
        "folder": "llava-1.5-7b",
        "trust_remote_code": False,
        "model_type": "vlm",
        "description": "LLaVA 1.5 7B — vision-language model for image understanding",
        "params": "7B",
        "download_size": "14 GB",
    },
    "llava-med": {
        "hf_name": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "folder": "llava-med-7b",
        "trust_remote_code": False,
        "model_type": "vlm",
        "description": "LLaVA-Med 7B — medical/radiology vision-language model (HF-compatible)",
        "params": "7B",
        "download_size": "14 GB",
    },
}


def _weights_dir() -> Path:
    return Path(__file__).parent.parent.parent / "weights"


def _canonical_folder(model_path: str) -> str:
    """Resolve any model reference to its canonical folder name.

    Priority:
      1. Registry alias  (e.g. "tinyllama" → "tinyllama-1.1b")
      2. Registry lookup by folder name (e.g. "weights/tinyllama-1.1b" → "tinyllama-1.1b")
      3. Registry lookup by HF name (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0" → "tinyllama-1.1b")
      4. Fallback: basename of local path or sanitized string
    """
    key = model_path.lower()

    # 1. Direct alias match
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["folder"]

    # 2/3. Match by folder name or HF name
    basename = Path(model_path).name if Path(model_path).exists() else model_path
    for info in MODEL_REGISTRY.values():
        if info["folder"] == basename or info["hf_name"] == model_path:
            return info["folder"]

    # 4. Fallback
    p = Path(model_path)
    return p.name if p.exists() else model_path.replace("/", "_").replace("\\", "_")


def get_quantized_dir(model_path: str, suffix: str = "int8") -> Path:
    """Single source of truth for quantized model directory naming.

    Always resolves through the registry so "tinyllama", "weights/tinyllama-1.1b",
    and "TinyLlama/TinyLlama-1.1B-Chat-v1.0" all map to the same dir.
    """
    folder = _canonical_folder(model_path)
    return _weights_dir() / f"{folder}-{suffix}"
