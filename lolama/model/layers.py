"""LLaMA Neural Network Layers.

Contains only nn.Module classes:
- RMSNorm: Root Mean Square Layer Normalization
- LlamaAttention: Multi-head attention with RoPE and GQA
- SwiGLU: Gated Linear Unit with SiLU activation
- LlamaBlock: Single transformer block
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig
from .kv_cache import KVCache, repeat_kv
from ..utils.rope import apply_rope


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Computes in fp32 for numerical stability, then converts back.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype: torch.dtype = x.dtype
        x = x.to(torch.float32)
        variance: torch.Tensor = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


# Module-level causal mask cache, keyed by device to avoid cross-device bugs.
# Each entry grows monotonically as larger sequences are seen.
_causal_mask_cache: dict[torch.device, torch.Tensor] = {}


def _get_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Return a lower-triangular bool mask of at least ``size``, cached per device."""
    cached = _causal_mask_cache.get(device)
    if cached is None or cached.shape[0] < size:
        _causal_mask_cache[device] = torch.tril(
            torch.ones(size, size, device=device, dtype=torch.bool)
        )
    return _causal_mask_cache[device]


class LlamaAttention(nn.Module):
    """Multi-head attention with RoPE, GQA, and Flash Attention."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads: int = config.num_heads
        self.num_kv_heads: int = config.num_kv_heads
        self.head_dim: int = config.d_model // config.num_heads
        self.d_model: int = config.d_model
        self.n_rep: int = self.num_heads // self.num_kv_heads

        # Q projection: full size
        self.q_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # K/V projections: smaller if using GQA
        kv_dim: int = self.num_kv_heads * self.head_dim
        self.k_proj: nn.Linear = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj: nn.Linear = nn.Linear(config.d_model, kv_dim, bias=False)
        
        self.o_proj: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, d_model)
            cos, sin: RoPE frequencies from parent model
            kv_cache: KVCache for generation (updated in-place)
            attention_mask: (B, L) mask with 1=attend, 0=padding
        """
        B: int
        L: int
        B, L, _ = x.shape
        
        # Project to Q, K, V
        Q: torch.Tensor = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K: torch.Tensor = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V: torch.Tensor = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE with position offset from cache
        offset: int = kv_cache.seq_len if kv_cache is not None else 0
        Q = apply_rope(Q, cos[offset:offset+L], sin[offset:offset+L])
        K = apply_rope(K, cos[offset:offset+L], sin[offset:offset+L])
        
        # Update KV cache (if provided)
        if kv_cache is not None:
            K, V = kv_cache.update(K, V)
        
        # Expand K/V heads for GQA
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        
        # Determine masking strategy
        S: int = K.shape[2]  # Full sequence length (including cache)
        
        attn_mask: torch.Tensor | None
        is_causal: bool
        if attention_mask is not None:
            # Padding mask: (B, S) -> (B, 1, 1, S) for broadcasting
            # attention_mask has 1=attend, 0=padding
            padding_mask: torch.Tensor = attention_mask[:, None, None, :].bool()
            
            combined_mask: torch.Tensor
            if L > 1:
                # Prefill: need causal + padding mask
                causal_mask: torch.Tensor = _get_causal_mask(S, x.device)[S - L:S, :S]
                combined_mask = causal_mask.unsqueeze(0) & padding_mask
            else:
                # Generation (L=1): new token can attend to all non-padding positions
                # No causal constraint needed - just use padding mask
                combined_mask = padding_mask.expand(-1, -1, L, -1)
            
            # Convert to additive mask for SDPA (0=attend, -inf=mask)
            attn_mask = torch.where(combined_mask, 0.0, float('-inf')).to(x.dtype)
            is_causal = False  # We handle masking ourselves
        else:
            attn_mask = None
            # Use is_causal only for prefill (L > 1), not for single-token generation
            is_causal = (L > 1)
        
        # Flash Attention via PyTorch's scaled_dot_product_attention
        out: torch.Tensor = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward network."""
    
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    """A single LLaMA transformer block."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention: LlamaAttention = LlamaAttention(config)
        self.feed_forward: SwiGLU = SwiGLU(config.d_model, config.hidden_dim)
        self.attention_norm: RMSNorm = RMSNorm(config.d_model, config.eps)
        self.ffn_norm: RMSNorm = RMSNorm(config.d_model, config.eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """KV cache and attention_mask are optional."""
        attn_out: torch.Tensor = self.attention(
            self.attention_norm(x), cos, sin,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
