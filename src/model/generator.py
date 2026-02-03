"""Text generation utilities - separates generation logic from model."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from .generation_config import GenerationConfig
from .kv_cache import KVCache
from .sampler import Sampler
from ..protocols import GenerativeModel


class TextGenerator:
    """Handles text generation using any model that implements GenerativeModel.
    
    Separates generation logic from the model itself, following
    single-responsibility principle.
    
    Example:
        model = load_model("weights/tinyllama-1.1b")
        generator = TextGenerator(model)
        
        # Generate with config
        output = generator.generate(input_ids, GenerationConfig(temperature=0.7))
        
        # Stream tokens
        for token_id in generator.generate_stream(input_ids):
            print(tokenizer.decode([token_id]), end="", flush=True)
    """
    
    def __init__(self, model: GenerativeModel) -> None:
        """Initialize generator with a model.
        
        Args:
            model: Any model implementing the GenerativeModel protocol
        """
        self.model: GenerativeModel = model
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device
    
    @property
    def config(self):
        """Get the model config."""
        return self.model.config
    
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
        
        Returns:
            torch.Tensor: Generated token IDs including prompt (B, L + generated)
        
        Examples:
            # Using config (recommended)
            output = generator.generate(input_ids, GenerationConfig(temperature=0.7))
            output = generator.generate(input_ids, GenerationConfig.greedy())
            
            # Using kwargs (backwards compatible)
            output = generator.generate(input_ids, max_new_tokens=100, temperature=0.7)
        """
        if config is None:
            config = GenerationConfig(**kwargs)
        
        self.model.eval()
        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        
        # Track which sequences have finished (hit eos)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.model.create_kv_caches(
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
        
        logits: torch.Tensor = self.model(input_ids, kv_caches=kv_caches)
        
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
            
            logits = self.model(next_token, kv_caches=kv_caches)
        
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
        if config is None:
            config = GenerationConfig(**kwargs)
        
        self.model.eval()
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
        kv_caches: list[KVCache] = self.model.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=prompt_len + config.max_new_tokens,
        )
        
        # Initial forward pass (populates cache)
        logits: torch.Tensor = self.model(input_ids, kv_caches=kv_caches)
        
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
            logits = self.model(next_token, kv_caches=kv_caches)
    
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
        if config is None:
            config = GenerationConfig(**kwargs)
        
        self.model.eval()
        device: torch.device = self.device
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
        kv_caches: list[KVCache] = self.model.create_kv_caches(
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
        logits: torch.Tensor = self.model(input_ids, kv_caches=kv_caches, attention_mask=attention_mask)
        
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
            
            logits = self.model(next_token, kv_caches=kv_caches, attention_mask=attention_mask)
        
        # Return original prompts + generated tokens (without padding)
        results: list[torch.Tensor] = []
        for i, prompt in enumerate(prompts):
            generated: torch.Tensor = torch.tensor(generated_tokens[i], dtype=prompt.dtype, device=device)
            results.append(torch.cat([prompt, generated]))
        
        return results
