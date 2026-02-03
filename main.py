#!/usr/bin/env python3
"""
LLaMA from Scratch
==================

Usage:
    python main.py                    # Test with random weights
    python main.py --load tinyllama   # Load pretrained TinyLlama
"""

from __future__ import annotations

import sys

import torch

from src.model import Llama, LlamaConfig


def main() -> None:
    from src.utils import resolve_device
    device = resolve_device()
    print(f"Device: {device}")
    print()
    
    if "--load" in sys.argv:
        # Load pretrained model
        idx = sys.argv.index("--load")
        model_name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "tinyllama"
        
        from src.data import load_model
        
        model_map = {
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "phi2": "microsoft/phi-2",
        }
        
        hf_name = model_map.get(model_name, model_name)
        model = load_model(hf_name, device=device)
        
    else:
        # Create small model with random weights
        print("Creating small model (random weights)...")
        config = LlamaConfig(
            vocab_size=32000,
            d_model=256,
            num_heads=4,
            num_layers=4,
            hidden_dim=688,
            max_seq_len=512
        )
        model = Llama(config).to(device)
        
        params = model.count_parameters()
        print(f"Parameters: {params['total']:,}")
    
    # Test forward pass
    print()
    print("Testing forward pass...")
    input_ids = torch.randint(0, 32000, (1, 16)).to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input:  {input_ids.shape}")
    print(f"Output: {logits.shape}")
    print()
    print("✅ Model works!")


if __name__ == "__main__":
    main()
