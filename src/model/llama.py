"""Full LLaMA Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterator

from .config import LlamaConfig
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
        cos, sin = precompute_rope_frequencies(head_dim, config.max_seq_len)
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
        cos, sin = precompute_rope_frequencies(head_dim, self.config.max_seq_len)
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
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive text generation with pre-allocated KV cache.
        
        Args:
            input_ids: Input token IDs (B, L)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (ignored if do_sample=False)
            top_k: Top-k sampling (ignored if do_sample=False)
            top_p: Nucleus sampling threshold (ignored if do_sample=False)
            do_sample: If False, use greedy decoding (deterministic)
            eos_token_id: Stop generation when this token is produced
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        """
        self.eval()
        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        
        # Track which sequences have finished (hit eos)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + max_new_tokens,
        )
        
        # Initial forward pass (populates cache)
        logits: torch.Tensor = self(input_ids, kv_caches=kv_caches)
        
        for _ in range(max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in input_ids[i].unique():
                        next_logits[i, token_id] /= repetition_penalty
            
            next_token: torch.Tensor
            if do_sample:
                # Sampling with temperature
                next_logits = next_logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v: torch.Tensor
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits: torch.Tensor
                    sorted_indices: torch.Tensor
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs: torch.Tensor = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Scatter back to original indexing
                    indices_to_remove: torch.Tensor = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                probs: torch.Tensor = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding - just take argmax
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
            
            logits = self(next_token, kv_caches=kv_caches)
        
        return input_ids
    
    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> Iterator[int]:
        """Streaming text generation - yields tokens as they're generated.
        
        Args:
            input_ids: Input token IDs (B, L)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (ignored if do_sample=False)
            top_k: Top-k sampling (ignored if do_sample=False)
            top_p: Nucleus sampling threshold (ignored if do_sample=False)
            do_sample: If False, use greedy decoding (deterministic)
            eos_token_id: Stop generation when this token is produced
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        
        Yields:
            int: Token ID for each generated token
        """
        self.eval()
        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        
        if batch_size != 1:
            raise ValueError("Streaming only supports batch_size=1")
        
        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + max_new_tokens,
        )
        
        # Initial forward pass (populates cache)
        logits: torch.Tensor = self(input_ids, kv_caches=kv_caches)
        
        for _ in range(max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in input_ids[0].unique():
                    next_logits[0, token_id] /= repetition_penalty
            
            next_token: torch.Tensor
            if do_sample:
                # Sampling with temperature
                next_logits = next_logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v: torch.Tensor
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits: torch.Tensor
                    sorted_indices: torch.Tensor
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs: torch.Tensor = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove: torch.Tensor = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                probs: torch.Tensor = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            token_id: int = next_token.item()
            yield token_id
            
            # Check for EOS token
            if eos_token_id is not None and token_id == eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits = self(next_token, kv_caches=kv_caches)
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0,
    ) -> list[torch.Tensor]:
        """Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of token ID tensors, each (1, L_i) or (L_i,)
            max_new_tokens: Maximum tokens to generate per sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: If False, use greedy decoding
            eos_token_id: Stop token
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: Token ID used for padding
        
        Returns:
            List of generated token tensors (without padding)
        """
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
                padding: torch.Tensor = torch.full((pad_len,), pad_token_id, dtype=p.dtype, device=device)
                padded_prompts.append(torch.cat([padding, p]))
            else:
                padded_prompts.append(p.to(device))
        
        input_ids: torch.Tensor = torch.stack(padded_prompts)  # (B, max_prompt_len)
        
        # Create attention mask: 1 for real tokens, 0 for padding
        attention_mask: torch.Tensor = (input_ids != pad_token_id).long()
        
        # Track which sequences have finished
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Create KV caches
        kv_caches: list[KVCache] = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_prompt_len + max_new_tokens,
        )
        
        # Prefill
        logits: torch.Tensor = self(input_ids, kv_caches=kv_caches, attention_mask=attention_mask)
        
        # Generation loop
        generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]
        
        for _ in range(max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in input_ids[i].unique():
                        if token_id != pad_token_id:
                            next_logits[i, token_id] /= repetition_penalty
            
            next_token: torch.Tensor
            if do_sample:
                next_logits = next_logits / temperature
                
                if top_k is not None:
                    v: torch.Tensor
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                if top_p is not None:
                    sorted_logits: torch.Tensor
                    sorted_indices: torch.Tensor
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs: torch.Tensor = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove: torch.Tensor = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                probs: torch.Tensor = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Store generated tokens (only for unfinished sequences)
            for i in range(batch_size):
                if not finished[i]:
                    generated_tokens[i].append(next_token[i].item())
            
            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
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
