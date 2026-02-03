#!/usr/bin/env python3
"""
Test Model Loading
==================
Verify that weight loading works correctly.

Usage:
    python scripts/test_model.py
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer


def main():
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("Test: Loading LLaMA Weights")
    print("=" * 60)
    print()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load model
    try:
        model = load_model(MODEL_NAME, device=device)
        print("\n✅ Model loaded!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Test generation
    print()
    print("=" * 60)
    print("Testing Generation")
    print("-" * 60)
    
    tokenizer = load_tokenizer(MODEL_NAME)
    
    prompt = "The meaning of life is"
    print(f"Prompt: \"{prompt}\"")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=50)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: \"{output_text}\"")
    
    # Check if coherent
    if len(output_text) > len(prompt) + 10:
        print("\n✅ Model is generating!")
    else:
        print("\n⚠️  Output seems short")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
