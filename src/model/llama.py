"""Full LLaMA Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterator

from .config import LlamaConfig
from .generation_config import GenerationConfig
from .kv_cache import KVCache
from .layers import LlamaBlock, RMSNorm
from .sampler import Sampler
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
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive text generation with pre-allocated KV cache.
        
        Args:
            input_ids: Input token IDs (B, L)
            config: Generation configuration (preferred)
            **kwargs: Individual parameters (for backwards compatibility):
                max_new_tokens, temperature, top_k, top_p, do_sample,
                eos_token_id, repetition_penalty
        
        Examples:
            # Using config (recommended)
            output = model.generate(input_ids, GenerationConfig(temperature=0.7))
            output = model.generate(input_ids, GenerationConfig.greedy())
            
            # Using kwargs (backwards compatible)
            output = model.generate(input_ids, max_new_tokens=100, temperature=0.7)
        """
        # Build config from kwargs if not provided
        if config is None:
            config = GenerationConfig(**kwargs)
        
        self.eval()
        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        
        # Track which sequences have finished (hit eos)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + config.max_new_tokens,
        )
        
        # Create sampler
        sampler: Sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )
        
        logits: torch.Tensor = self(input_ids, kv_caches=kv_caches)
        
        for _ in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]
            
            # Apply repetition penalty
            Sampler.apply_repetition_penalty(next_logits, input_ids, config.repetition_penalty)
            
            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS token
            if config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break
            
            logits = self(next_token, kv_caches=kv_caches)
        
        return input_ids
    
    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> Iterator[int]:
        """Streaming text generation - yields tokens as they're generated.
        
        Args:
            input_ids: Input token IDs (B, L) - must have batch_size=1
            config: Generation configuration (preferred)
            **kwargs: Individual parameters (for backwards compatibility)
        
        Yields:
            int: Token ID for each generated token
        """
        # Build config from kwargs if not provided
        if config is None:
            config = GenerationConfig(**kwargs)
        
        self.eval()
        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        
        if batch_size != 1:
            raise ValueError("Streaming only supports batch_size=1")
        
        # Create sampler
        sampler: Sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )
        
        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + config.max_new_tokens,
        )
        
        # Initial forward pass (populates cache)
        logits: torch.Tensor = self(input_ids, kv_caches=kv_caches)
        
        for _ in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]
            
            # Apply repetition penalty
            Sampler.apply_repetition_penalty(next_logits, input_ids, config.repetition_penalty)
            
            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)
            
            token_id: int = next_token.item()
            yield token_id
            
            # Check for EOS token
            if config.eos_token_id is not None and token_id == config.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits = self(next_token, kv_caches=kv_caches)
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[torch.Tensor],
        config: GenerationConfig | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of token ID tensors, each (1, L_i) or (L_i,)
            config: Generation configuration (preferred)
            **kwargs: Individual parameters (for backwards compatibility)
        
        Returns:
            List of generated token tensors (without padding)
        """
        # Build config from kwargs if not provided
        if config is None:
            config = GenerationConfig(**kwargs)
        
        self.eval()
        device: torch.device = next(self.parameters()).device
        batch_size: int = len(prompts)
        
        # Normalize prompts to 1D tensors
        prompts = [p.squeeze() if p.dim() > 1 else p for p in prompts]
        prompt_lengths: list[int] = [len(p) for p in prompts]
        max_prompt_len: int = max(prompt_lengths)
        
        # Pad prompts to same length (left-padding for causal LM)
        padded_prompts: list[torch.Tensor] = []
        for p in prompts:
            pad_len: int = max_prompt_len - len(p)
            if pad_len > 0:
                padding: torch.Tensor = torch.full((pad_len,), config.pad_token_id, dtype=p.dtype, device=device)
                padded_prompts.append(torch.cat([padding, p]))
            else:
                padded_prompts.append(p.to(device))
        
        input_ids: torch.Tensor = torch.stack(padded_prompts)  # (B, max_prompt_len)
        
        # Create attention mask: 1 for real tokens, 0 for padding
        attention_mask: torch.Tensor = (input_ids != config.pad_token_id).long()
        
        # Track which sequences have finished
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Create KV caches
        kv_caches: list[KVCache] = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_prompt_len + config.max_new_tokens,
        )
        
        # Create sampler
        sampler: Sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )
        
        # Prefill
        logits: torch.Tensor = self(input_ids, kv_caches=kv_caches, attention_mask=attention_mask)
        
        # Generation loop
        generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        
        for _ in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]
            
            # Apply repetition penalty (ignore pad token)
            Sampler.apply_repetition_penalty(
                next_logits, input_ids, config.repetition_penalty,
                ignore_token_id=config.pad_token_id
            )
            
            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)
            
            # Store generated tokens (only for unfinished sequences)
            for i in range(batch_size):
                if not finished[i]:
                    generated_tokens[i].append(next_token[i].item())
            
            # Check for EOS
            if config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break
            
            # Update for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Extend attention mask (new tokens are always real)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            ], dim=1)
            
            logits = self(next_token, kv_caches=kv_caches, attention_mask=attention_mask)
        
        # Return original prompts + generated tokens (without padding)
        results: list[torch.Tensor] = []
        for i, prompt in enumerate(prompts):
            generated: torch.Tensor = torch.tensor(generated_tokens[i], dtype=prompt.dtype, device=device)
            results.append(torch.cat([prompt, generated]))
        
        return results
    
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
