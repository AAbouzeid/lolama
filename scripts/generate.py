#!/usr/bin/env python3
"""
Text Generation
===============
Generate text using a loaded model.

Usage:
    python scripts/generate.py "The meaning of life is"
    python scripts/generate.py --model weights/tinyllama-1.1b "Hello"
"""

import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer


def main():
    # Parse args
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = "The meaning of life is"
    
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        model_path = args[idx + 1]
        args = args[:idx] + args[idx+2:]
    
    if args:
        prompt = " ".join(args)
    
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load model
    model = load_model(model_path, device=device)
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    
    # Generate
    print()
    print(f"Prompt: \"{prompt}\"")
    print("-" * 50)
    
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: \"{output_text}\"")


if __name__ == "__main__":
    main()
