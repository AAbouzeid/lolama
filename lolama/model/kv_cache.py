"""KV Cache for efficient autoregressive generation."""

from __future__ import annotations

import torch


class KVCache:
    """Pre-allocated KV cache for efficient generation.
    
    Instead of torch.cat every step (slow, allocates memory),
    we pre-allocate once and use slicing to update.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_seq_len: int = max_seq_len
        self.current_len: int = 0
        
        # Pre-allocate buffers: (batch, num_kv_heads, max_seq_len, head_dim)
        self.k_cache: torch.Tensor = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache: torch.Tensor = torch.zeros(
            batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
    
    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V and return full cache up to current position.
        
        Args:
            k, v: New key/value tensors of shape (batch, num_kv_heads, new_len, head_dim)
        
        Returns:
            Full K, V tensors up to current position
        """
        new_len: int = k.shape[2]
        end: int = self.current_len + new_len

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {end} tokens exceeds max_seq_len={self.max_seq_len}"
            )

        # Write new values into pre-allocated buffer
        self.k_cache[:, :, self.current_len:end] = k
        self.v_cache[:, :, self.current_len:end] = v

        self.current_len = end
        
        # Return view of cache up to current position
        return (
            self.k_cache[:, :, :self.current_len],
            self.v_cache[:, :, :self.current_len],
        )
    
    def reset(self) -> None:
        """Reset cache for new generation."""
        self.current_len = 0
    
    @property
    def seq_len(self) -> int:
        """Current sequence length in cache."""
        return self.current_len


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V heads to match the number of Q heads (for GQA).
    
    Args:
        x: (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
    
    Returns:
        (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    
    batch: int
    num_kv_heads: int
    seq_len: int
    head_dim: int
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.contiguous().view(batch, num_kv_heads * n_rep, seq_len, head_dim)
