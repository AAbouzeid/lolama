"""Full LLaMA Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

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
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # RoPE: compute once, share across all layers
        head_dim = config.d_model // config.num_heads
        cos, sin = precompute_rope_frequencies(head_dim, config.max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
        # Weight tying: LLaMA-1/2 use it, TinyLlama does not
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Skip init when loading pretrained (saves time on 1B+ params)
        if init_weights:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_kv_caches(
        self,
        batch_size: int,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[KVCache]:
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
        
        head_dim = self.config.d_model // self.config.num_heads
        
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
        kv_caches: Optional[List[KVCache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: (B, L) input token IDs
            kv_caches: Optional List[KVCache] for generation (updated in-place)
            attention_mask: Optional (B, L) mask with 1=real token, 0=padding
        
        Returns:
            logits: (B, L, vocab_size)
        """
        x = self.embed_tokens(input_ids)
        
        # Transformer layers (KV caches updated in-place)
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, self.cos, self.sin, kv_cache=layer_cache, attention_mask=attention_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=True,
        eos_token_id=None,
        repetition_penalty=1.0,
    ):
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
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        
        # Track which sequences have finished (hit eos)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Create pre-allocated KV cache
        kv_caches = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + max_new_tokens,
        )
        
        # Initial forward pass (populates cache)
        logits = self(input_ids, kv_caches=kv_caches)
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in input_ids[i].unique():
                        next_logits[i, token_id] /= repetition_penalty
            
            if do_sample:
                # Sampling with temperature
                next_logits = next_logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
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
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        do_sample=True,
        eos_token_id=None,
        repetition_penalty=1.0,
    ):
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
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        
        if batch_size != 1:
            raise ValueError("Streaming only supports batch_size=1")
        
        # Create pre-allocated KV cache
        kv_caches = self.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + max_new_tokens,
        )
        
        # Initial forward pass (populates cache)
        logits = self(input_ids, kv_caches=kv_caches)
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in input_ids[0].unique():
                    next_logits[0, token_id] /= repetition_penalty
            
            if do_sample:
                # Sampling with temperature
                next_logits = next_logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            token_id = next_token.item()
            yield token_id
            
            # Check for EOS token
            if eos_token_id is not None and token_id == eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits = self(next_token, kv_caches=kv_caches)
    
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_tokens.weight.numel()
        
        if self.layers:
            layer = self.layers[0]
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
