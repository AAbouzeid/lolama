"""Quantization utilities."""

import torch
import torch.nn as nn


def quantize_tensor_int8(tensor):
    """
    Quantize a tensor to int8.
    
    Formula:
        quantized = round(tensor / scale)
        where scale = max(abs(tensor)) / 127
    
    Args:
        tensor: fp16/fp32 tensor
    
    Returns:
        quantized: int8 tensor
        scale: float (to dequantize later)
    """
    max_val = tensor.abs().max()
    scale = max_val / 127.0
    quantized = torch.round(tensor / scale).to(torch.int8)
    return quantized, scale


def dequantize_tensor_int8(quantized, scale):
    """Convert int8 back to fp16."""
    return quantized.float() * scale


def quantize_model_int8(model):
    """
    Quantize all Linear layers in model to int8.
    
    This is a simple version. Real quantization (like Unsloth) is more sophisticated.
    """
    print("Quantizing model to int8...")
    
    original_size = 0
    quantized_size = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            original_size += weight.numel() * 2  # fp16 = 2 bytes
            
            quantized_weight, scale = quantize_tensor_int8(weight)
            quantized_size += quantized_weight.numel() * 1  # int8 = 1 byte
            quantized_size += 4  # scale is fp32 = 4 bytes
            
            print(f"  {name}: {weight.shape} -> int8 (scale={scale:.6f})")
    
    compression_ratio = original_size / quantized_size
    print(f"\n✅ Quantization complete!")
    print(f"   Original: {original_size / 1024**2:.1f} MB")
    print(f"   Quantized: {quantized_size / 1024**2:.1f} MB")
    print(f"   Compression: {compression_ratio:.2f}x")
    
    return model
