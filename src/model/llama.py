"""Full LLaMA Model."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from .config import LlamaConfig
from .generation_config import GenerationConfig
from .generator import TextGenerator
from .kv_cache import KVCache
from .layers import LlamaBlock, RMSNorm
from ..utils.rope import precompute_rope_frequencies


class Llama(nn.Module):
    """Complete LLaMA model."""
    
    def __init__(self, config: LlamaConfig, init_weights: bool = True):
        """
        Args:
            config: Model configuration
            init_weights: If True, initialize weights randomly. Set to False when
                         loading pretrained weights to skip unnecessary init.
        """
        super().__init__()
        self.config: LlamaConfig = config
        
        self.embed_tokens: nn.Embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers: nn.ModuleList = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_layers)])
        self.norm: RMSNorm = RMSNorm(config.d_model, config.eps)
        self.lm_head: nn.Linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # RoPE: compute once, share across all layers
        head_dim: int = config.d_model // config.num_heads
        cos: torch.Tensor
        sin: torch.Tensor
        cos, sin = precompute_rope_frequencies(head_dim, config.max_seq_len, base=config.rope_base)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
        # Weight tying: LLaMA-1/2 use it, TinyLlama does not
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Skip init when loading pretrained (saves time on 1B+ params)
        if init_weights:
            self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def init_rope(self) -> None:
        """Re-initialize RoPE buffers. Call after materializing from meta device."""
        head_dim: int = self.config.d_model // self.config.num_heads
        cos, sin = precompute_rope_frequencies(
            head_dim, self.config.max_seq_len, base=self.config.rope_base
        )
        # Copy data to existing buffers (preserves device)
        self.cos.copy_(cos.to(self.cos.device))
        self.sin.copy_(sin.to(self.sin.device))
    
    def create_kv_caches(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[KVCache]:
        """Create pre-allocated KV caches for all layers.
        
        Args:
            batch_size: Batch size
            max_seq_len: Max sequence length (defaults to config.max_seq_len)
            device: Device for caches (defaults to model device)
            dtype: dtype for caches (defaults to model dtype)
        
        Returns:
            List of KVCache, one per layer
        """
        if max_seq_len is None:
            max_seq_len = self.config.max_seq_len
        if device is None:
            device = self.embed_tokens.weight.device
        if dtype is None:
            dtype = self.embed_tokens.weight.dtype
        
        head_dim: int = self.config.d_model // self.config.num_heads
        
        return [
            KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.config.num_layers)
        ]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) input token IDs
            kv_caches: Optional List[KVCache] for generation (updated in-place)
            attention_mask: Optional (B, L) mask with 1=real token, 0=padding
        
        Returns:
            logits: (B, L, vocab_size)
        """
        x: torch.Tensor = self.embed_tokens(input_ids)
        
        # Transformer layers (KV caches updated in-place)
        for i, layer in enumerate(self.layers):
            layer_cache: KVCache | None = kv_caches[i] if kv_caches is not None else None
            x = layer(x, self.cos, self.sin, kv_cache=layer_cache, attention_mask=attention_mask)
        
        x = self.norm(x)
        logits: torch.Tensor = self.lm_head(x)
        
        return logits
    
    def _get_generator(self) -> TextGenerator:
        """Get or create a TextGenerator for this model."""
        if not hasattr(self, '_generator'):
            self._generator: TextGenerator = TextGenerator(self)
        return self._generator
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive text generation with pre-allocated KV cache.
        
        Note: Consider using TextGenerator directly for more control.
        
        Args:
            input_ids: Input token IDs (B, L)
            config: Generation configuration (preferred)
            **kwargs: Individual parameters (for backwards compatibility)
        
        Returns:
            torch.Tensor: Generated token IDs including prompt
        """
        return self._get_generator().generate(input_ids, config, **kwargs)
    
    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> Iterator[int]:
        """Streaming text generation - yields tokens as they're generated.
        
        Note: Consider using TextGenerator directly for more control.
        
        Args:
            input_ids: Input token IDs (B, L) - must have batch_size=1
            config: Generation configuration (preferred)
            **kwargs: Individual parameters (for backwards compatibility)
        
        Yields:
            int: Token ID for each generated token
        """
        yield from self._get_generator().generate_stream(input_ids, config, **kwargs)
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[torch.Tensor],
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Generate text for multiple prompts in parallel.
        
        Note: Consider using TextGenerator directly for more control.
        
        Args:
            prompts: List of token ID tensors
            config: Generation configuration (preferred)
            **kwargs: Individual parameters (for backwards compatibility)
        
        Returns:
            List of generated token tensors
        """
        return self._get_generator().generate_batch(prompts, config, **kwargs)
    
    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        total: int = sum(p.numel() for p in self.parameters())
        embedding: int = self.embed_tokens.weight.numel()
        
        attn_params: int
        ffn_params: int
        norm_params: int
        if self.layers:
            layer: LlamaBlock = self.layers[0]
            attn_params = sum(p.numel() for p in layer.attention.parameters())
            ffn_params = sum(p.numel() for p in layer.feed_forward.parameters())
            norm_params = sum(p.numel() for p in layer.attention_norm.parameters())
            norm_params += sum(p.numel() for p in layer.ffn_norm.parameters())
        else:
            attn_params = ffn_params = norm_params = 0
        
        return {
            'total': total,
            'embedding': embedding,
            'per_layer_attention': attn_params,
            'per_layer_ffn': ffn_params,
            'per_layer_norms': norm_params,
            'num_layers': len(self.layers)
        }
