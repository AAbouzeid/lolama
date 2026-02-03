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
        cos, sin: (max_seq_len, dim) tensors (repeated for full dimension)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, dim/2)
    
    # Repeat to match full head_dim (HuggingFace style)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    """Rotate half the hidden dims of the input (HuggingFace style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """
    Apply rotary position embeddings to x (HuggingFace compatible).
    
    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim)
    
    Returns:
        Rotated tensor with same shape as x
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)
