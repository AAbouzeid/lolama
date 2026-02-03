"""Token sampling strategies for text generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Sampler:
    """Handles token sampling with temperature, top-k, and top-p filtering.
    
    Consolidates sampling logic that was previously duplicated across
    generate(), generate_stream(), and generate_batch().
    
    Usage:
        sampler = Sampler(temperature=0.7, top_k=50, top_p=0.9)
        next_token = sampler.sample(logits)  # (B, vocab_size) -> (B, 1)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
    ):
        """
        Args:
            temperature: Sampling temperature. Higher = more random.
            top_k: Keep only top-k tokens before sampling.
            top_p: Keep tokens with cumulative probability <= top_p.
            do_sample: If False, use greedy decoding (ignores temperature/top_k/top_p).
        """
        self.temperature: float = temperature
        self.top_k: int | None = top_k
        self.top_p: float | None = top_p
        self.do_sample: bool = do_sample
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token(s) from logits.
        
        Args:
            logits: (B, vocab_size) logits for next token
            
        Returns:
            (B, 1) tensor of sampled token IDs
        """
        if not self.do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Top-k filtering
        if self.top_k is not None:
            top_k: int = min(self.top_k, logits.size(-1))
            top_values: torch.Tensor
            top_values, _ = torch.topk(logits, top_k)
            min_top_value: torch.Tensor = top_values[:, -1:]
            logits = logits.masked_fill(logits < min_top_value, float('-inf'))
        
        # Top-p (nucleus) filtering
        if self.top_p is not None:
            sorted_logits: torch.Tensor
            sorted_indices: torch.Tensor
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs: torch.Tensor = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            # Find tokens to remove (cumulative prob > top_p)
            sorted_mask: torch.Tensor = cumulative_probs > self.top_p
            # Keep at least one token
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            
            # Scatter mask back to original indices
            mask: torch.Tensor = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(mask, float('-inf'))
        
        # Sample from distribution
        probs: torch.Tensor = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
        ignore_token_id: int | None = None,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits.
        
        Args:
            logits: (B, vocab_size) logits to modify
            input_ids: (B, seq_len) tokens to penalize
            penalty: Penalty factor (1.0 = no penalty)
            ignore_token_id: Token ID to skip (e.g., pad token)
            
        Returns:
            Modified logits (in-place modification, also returned for convenience)
        """
        if penalty == 1.0:
            return logits
        
        batch_size: int = logits.shape[0]
        for i in range(batch_size):
            unique_tokens: torch.Tensor = input_ids[i].unique()
            for token_id in unique_tokens:
                if ignore_token_id is not None and token_id == ignore_token_id:
                    continue
                logits[i, token_id] /= penalty
        
        return logits
