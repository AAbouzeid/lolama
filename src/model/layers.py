"""LLaMA Neural Network Layers.

Contains only nn.Module classes:
- RMSNorm: Root Mean Square Layer Normalization
- LlamaAttention: Multi-head attention with RoPE and GQA
- SwiGLU: Gated Linear Unit with SiLU activation
- LlamaBlock: Single transformer block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class LlamaAttention(nn.Module):
    """Multi-head attention with RoPE, GQA, and Flash Attention."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_model = config.d_model
        self.n_rep = self.num_heads // self.num_kv_heads
        
        # Q projection: full size
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # K/V projections: smaller if using GQA
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, d_model)
            cos, sin: RoPE frequencies from parent model
            kv_cache: Optional KVCache for generation (updated in-place)
            attention_mask: Optional (B, L) mask with 1=attend, 0=padding
        """
        B, L, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE with position offset from cache
        offset = kv_cache.seq_len if kv_cache is not None else 0
        Q = apply_rope(Q, cos[offset:offset+L], sin[offset:offset+L])
        K = apply_rope(K, cos[offset:offset+L], sin[offset:offset+L])
        
        # Update KV cache (if provided)
        if kv_cache is not None:
            K, V = kv_cache.update(K, V)
        
        # Expand K/V heads for GQA
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        
        # Determine masking strategy
        S = K.shape[2]  # Full sequence length (including cache)
        
        if attention_mask is not None:
            # Create combined causal + padding mask
            # Causal: position i can attend to j where j <= i
            causal_mask = torch.tril(torch.ones(L, S, device=x.device, dtype=torch.bool))
            
            # Padding mask: (B, S) -> (B, 1, 1, S) for broadcasting
            # attention_mask has 1=attend, 0=padding
            padding_mask = attention_mask[:, None, None, :].bool()
            
            # Combine: attend only if causal allows AND not padding
            # Shape: (B, 1, L, S)
            combined_mask = causal_mask.unsqueeze(0) & padding_mask
            
            # Convert to additive mask for SDPA (0=attend, -inf=mask)
            attn_mask = torch.where(combined_mask, 0.0, float('-inf')).to(x.dtype)
            is_causal = False  # We handle causal in the mask
        else:
            attn_mask = None
            # Use is_causal only for prefill (L > 1), not for single-token generation
            is_causal = (L > 1)
        
        # Flash Attention via PyTorch's scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
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
        self.attention = LlamaAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.hidden_dim)
        self.attention_norm = RMSNorm(config.d_model, config.eps)
        self.ffn_norm = RMSNorm(config.d_model, config.eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """KV cache and attention_mask are optional."""
        attn_out = self.attention(
            self.attention_norm(x), cos, sin,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
