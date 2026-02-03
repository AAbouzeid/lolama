"""Rotary Position Embeddings (RoPE)."""

import torch


def precompute_rope_frequencies(dim, max_seq_len, base=10000):
    """
    Precompute the rotation frequencies for RoPE.
    
    Args:
        dim: Head dimension
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (10000 for LLaMA 1/2, 500000 for LLaMA 3)
    
    Returns:
        cos, sin: (max_seq_len, dim/2) tensors
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """
    Apply rotary position embeddings to x.
    
    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim/2)
    
    Returns:
        Rotated tensor with same shape as x
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
