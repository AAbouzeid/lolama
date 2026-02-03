"""Full LLaMA Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig
from .layers import LlamaBlock, RMSNorm


class Llama(nn.Module):
    """Complete LLaMA model."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: LLaMA-1/2 use it, TinyLlama does not
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, use_cache=False, past_kv=None):
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        
        # Causal mask
        if past_kv is not None:
            past_len = past_kv[0][0].size(2)
            mask = torch.ones(L, past_len + L, device=x.device)
            mask = torch.tril(mask, diagonal=past_len)
        else:
            mask = torch.tril(torch.ones(L, L, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Transformer layers
        new_kv = []
        for i, layer in enumerate(self.layers):
            layer_past = past_kv[i] if past_kv is not None else None
            x, kv = layer(x, mask, layer_past)
            if use_cache:
                new_kv.append(kv)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        if use_cache:
            return logits, new_kv
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Autoregressive text generation."""
        self.eval()
        logits, kv_cache = self(input_ids, use_cache=True)
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits, kv_cache = self(next_token, use_cache=True, past_kv=kv_cache)
        
        return input_ids
    
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
