"""LLaMA Layer Components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import LlamaConfig
from ..utils.rope import precompute_rope_frequencies, apply_rope


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Simpler than LayerNorm - no mean subtraction, just normalize by RMS.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def repeat_kv(x, n_rep):
    """
    Repeat K/V heads to match the number of Q heads (for GQA).
    
    Args:
        x: (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
    
    Returns:
        (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class LlamaAttention(nn.Module):
    """Multi-head attention with RoPE and Grouped Query Attention (GQA)."""
    
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
        
        # Precompute RoPE frequencies
        cos, sin = precompute_rope_frequencies(self.head_dim, config.max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x, mask=None, kv_cache=None):
        B, L, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            past_k, past_v = kv_cache
            offset = past_k.size(2)
            Q = apply_rope(Q, self.cos[offset:offset+L], self.sin[offset:offset+L])
            K = apply_rope(K, self.cos[offset:offset+L], self.sin[offset:offset+L])
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        else:
            Q = apply_rope(Q, self.cos[:L], self.sin[:L])
            K = apply_rope(K, self.cos[:L], self.sin[:L])
        
        # Store before expanding (memory savings)
        kv_to_cache = (K, V)
        
        # Expand K/V heads for GQA
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        
        # Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out), kv_to_cache


class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward network."""
    
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    """A single LLaMA transformer block."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.hidden_dim)
        self.attention_norm = RMSNorm(config.d_model, config.eps)
        self.ffn_norm = RMSNorm(config.d_model, config.eps)
    
    def forward(self, x, mask=None, kv_cache=None):
        attn_out, new_kv = self.attention(self.attention_norm(x), mask, kv_cache)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv
