#!/usr/bin/env python3
"""
Model Downloader
================
Downloads models from HuggingFace and saves to weights/ directory.

Usage:
    python scripts/download.py tinyllama
    python scripts/download.py phi2
    python scripts/download.py llama7b
    python scripts/download.py --from-cache tinyllama
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

MODELS = {
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "folder": "tinyllama-1.1b",
        "trust_remote_code": False,
    },
    "phi2": {
        "hf_name": "microsoft/phi-2",
        "folder": "phi-2",
        "trust_remote_code": True,
    },
    "llama7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "folder": "llama-7b",
        "trust_remote_code": False,
    },
}


def download(model_key, from_cache=False):
    """Download or save from cache a model (weights dir first)."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)
    
    info = MODELS[model_key]
    hf_name = info["hf_name"]
    save_dir = WEIGHTS_DIR / info["folder"]
    trust = info["trust_remote_code"]
    
    print(f"Model: {hf_name}")
    print(f"Save to: {save_dir}")
    print()
    
    if save_dir.exists() and any(save_dir.iterdir()):
        print("✅ Model already saved locally.")
        print(f"   Location: {save_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust,
            local_files_only=True,
        )
    except OSError:
        if from_cache:
            print("❌ Not found in cache and --from-cache was set.")
            sys.exit(1)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust,
            local_files_only=False,
        )
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            trust_remote_code=trust,
            local_files_only=True,
        )
    except OSError:
        if from_cache:
            print("❌ Tokenizer not found in cache and --from-cache was set.")
            sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            trust_remote_code=trust,
            local_files_only=False,
        )
    
    # Save
    print(f"Saving to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Stats
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 2 / 1024**2
    
    print()
    print(f"✅ Saved!")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size: {size_mb:.1f} MB (fp16)")
    print()
    print("Files:")
    for f in sorted(save_dir.iterdir()):
        size = f.stat().st_size / 1024**2
        print(f"   {f.name}: {size:.1f} MB")


def main():
    if len(sys.argv) < 2:
        print("Model Downloader")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python scripts/download.py <model>          # Download from HuggingFace")
        print("  python scripts/download.py --from-cache <model>  # Save from local cache")
        print()
        print("Models:")
        for key, info in MODELS.items():
            print(f"  {key:10} - {info['hf_name']}")
        sys.exit(1)
    
    from_cache = "--from-cache" in sys.argv
    model_key = [arg for arg in sys.argv[1:] if not arg.startswith("--")][0]
    
    download(model_key, from_cache)


if __name__ == "__main__":
    main()
