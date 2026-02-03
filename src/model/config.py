"""LLaMA Configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LlamaConfig:
    """LLaMA model configuration."""
    
    vocab_size: int = 32000
    d_model: int = 512           # Model dimension
    num_heads: int = 8           # Query heads
    num_kv_heads: int | None = None  # Key/Value heads (GQA). None = same as num_heads
    num_layers: int = 8          # Transformer blocks
    hidden_dim: int = 1376       # FFN hidden (≈2.7 * d_model)
    max_seq_len: int = 2048      # Maximum sequence length
    dropout: float = 0.0         # LLaMA doesn't use dropout
    eps: float = 1e-6            # RMSNorm epsilon
    tie_word_embeddings: bool = False  # Whether embed_tokens and lm_head share weights
    rope_base: int = 10000       # RoPE base frequency (10000 for LLaMA 1/2, 500000 for LLaMA 3)
    
    def __post_init__(self) -> None:
        # Default num_kv_heads to num_heads (standard MHA)
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        
        # Validation
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
