#!/usr/bin/env python3
"""
Quantization Test
=================
Test int8 quantization on the model.

Usage:
    python scripts/quantize_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lolama.data import load_model, load_tokenizer
from lolama.model import quantize_model_int8, get_model_size_mb, TextGenerator
from lolama.utils import resolve_device


def main() -> None:
    # Device
    device = resolve_device()
    print(f"Device: {device}")
    print()
    
    # Load model
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer(model_path)
    
    # Measure size before quantization
    size_before = get_model_size_mb(model)
    print(f"\nModel size before quantization: {size_before:.1f} MB")
    
    # Test generation before quantization
    prompt = "What is 2+2?"
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\nPrompt: \"{prompt}\"")
    print("\n--- Before Quantization (greedy) ---")

    generator = TextGenerator(model)
    with torch.no_grad():
        output_ids = generator.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,  # Greedy decoding for reproducibility
            eos_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text}")
    
    # Quantize the model
    print("\n" + "=" * 60)
    print("Quantizing model to int8...")
    print("=" * 60)
    
    # Skip lm_head to preserve output quality (common practice)
    quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])
    
    # Measure size after quantization
    size_after = get_model_size_mb(model)
    print(f"\nModel size after quantization: {size_after:.1f} MB")
    print(f"Compression ratio: {size_before / size_after:.2f}x")
    
    # Test generation after quantization
    print("\n--- After Quantization (greedy) ---")

    # Need to recreate generator since model changed
    generator = TextGenerator(model)
    with torch.no_grad():
        output_ids = generator.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,  # Greedy decoding for reproducibility
            eos_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text}")
    
    # Compare a few more prompts
    print("\n" + "=" * 60)
    print("More tests with quantized model:")
    print("=" * 60)
    
    test_prompts = [
        "The capital of France is",
        "Explain quantum computing in one sentence:",
    ]
    
    for prompt in test_prompts:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
            input_ids = input_ids.to(device)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = generator.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,  # Greedy decoding for reproducibility
                eos_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Output: {output_text}")


def test_outlier_quantization() -> None:
    """Test outlier-aware mixed-precision quantization."""
    device = resolve_device()
    print(f"Device: {device}")
    print()

    # Load a fresh model
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer(model_path)

    size_before = get_model_size_mb(model)
    print(f"\nModel size before quantization: {size_before:.1f} MB")

    # Tokenize test prompt
    prompt = "What is 2+2?"
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Quantize with outlier detection
    print("\n" + "=" * 60)
    print("Quantizing with outlier detection (threshold=6.0)...")
    print("=" * 60)

    quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'],
                        outlier_threshold=6.0)

    size_after = get_model_size_mb(model)
    print(f"\nModel size after quantization: {size_after:.1f} MB")
    print(f"Compression ratio: {size_before / size_after:.2f}x")

    # Report outlier stats
    from lolama.model.quantize import QuantizedLinear
    total_outlier_cols = 0
    layers_with_outliers = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear) and module.outlier_indices is not None:
            n_outlier = module.outlier_indices.numel()
            total_outlier_cols += n_outlier
            layers_with_outliers += 1
            print(f"  {name}: {n_outlier} outlier cols / {module.in_features}")
    print(f"\nTotal: {total_outlier_cols} outlier cols across {layers_with_outliers} layers")

    # Test generation
    print(f"\nPrompt: \"{prompt}\"")
    print("\n--- Outlier-Aware Quantization (greedy) ---")

    generator = TextGenerator(model)
    with torch.no_grad():
        output_ids = generator.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text}")

    # Verify extra_repr shows outlier_cols
    print("\n--- Layer repr sample ---")
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear) and module.outlier_indices is not None:
            print(f"  {name}: {module.extra_repr()}")
            break


if __name__ == "__main__":
    import sys

    if "--outlier" in sys.argv:
        test_outlier_quantization()
    else:
        main()
        print("\n\n" + "#" * 60)
        print("Running outlier-aware quantization test...")
        print("#" * 60 + "\n")
        test_outlier_quantization()
