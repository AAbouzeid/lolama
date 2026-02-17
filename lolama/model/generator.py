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
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive text generation with pre-allocated KV cache.

        Args:
            input_ids: Input token IDs (B, L)
            config: Generation configuration (preferred)
            pixel_values: Optional (B, 3, H, W) image tensor for VLMs.
                         Only used on first forward pass (cached internally after).
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

            # With image for VLM
            output = generator.generate(input_ids, pixel_values=pixel_values)
        """
        if config is None:
            config = GenerationConfig(**kwargs)

        self.model.eval()

        self.model.reset_image_cache()

        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        max_len: int = prompt_len + config.max_new_tokens

        # Track which sequences have finished (hit eos)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Pre-allocate token buffer (like KV cache — fill by index, never reallocate)
        all_ids: torch.Tensor = torch.empty(batch_size, max_len, dtype=input_ids.dtype, device=input_ids.device)
        all_ids[:, :prompt_len] = input_ids
        current_len: int = prompt_len

        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.model.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_len,
        )

        # Create sampler
        sampler: Sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )

        # First forward pass - include pixel_values for VLMs
        logits: torch.Tensor
        if pixel_values is not None:
            logits = self.model(input_ids, pixel_values=pixel_values, kv_caches=kv_caches)
        else:
            logits = self.model(input_ids, kv_caches=kv_caches)

        for _ in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]

            # Apply repetition penalty over all tokens so far
            Sampler.apply_repetition_penalty(next_logits, all_ids[:, :current_len], config.repetition_penalty)

            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)

            # Store in pre-allocated buffer (no allocation)
            all_ids[:, current_len] = next_token.squeeze(-1)
            current_len += 1

            # Check for EOS token
            if config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break

            # Subsequent passes don't need pixel_values (cached in KV)
            logits = self.model(next_token, kv_caches=kv_caches)

        return all_ids[:, :current_len]
    
    @torch.inference_mode()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> Iterator[int]:
        """Streaming text generation - yields tokens as they're generated.

        Args:
            input_ids: Input token IDs (B, L) - must have batch_size=1
            config: Generation configuration (preferred)
            pixel_values: Optional (B, 3, H, W) image tensor for VLMs.
                         Only used on first forward pass (cached internally after).
            **kwargs: Individual parameters (for backwards compatibility)

        Yields:
            int: Token ID for each generated token
        """
        if config is None:
            config = GenerationConfig(**kwargs)

        self.model.eval()

        self.model.reset_image_cache()

        batch_size: int = input_ids.shape[0]
        prompt_len: int = input_ids.shape[1]
        max_len: int = prompt_len + config.max_new_tokens

        if batch_size != 1:
            raise ValueError("Streaming only supports batch_size=1")

        # Create sampler
        sampler: Sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )

        # Pre-allocate token buffer (like KV cache — fill by index, never reallocate)
        all_ids: torch.Tensor = torch.empty(1, max_len, dtype=input_ids.dtype, device=input_ids.device)
        all_ids[:, :prompt_len] = input_ids
        current_len: int = prompt_len

        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = self.model.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_len,
        )

        # Initial forward pass (populates cache) - include pixel_values for VLMs
        logits: torch.Tensor
        if pixel_values is not None:
            logits = self.model(input_ids, pixel_values=pixel_values, kv_caches=kv_caches)
        else:
            logits = self.model(input_ids, kv_caches=kv_caches)

        for _ in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]

            # Apply repetition penalty over all tokens so far
            Sampler.apply_repetition_penalty(next_logits, all_ids[:, :current_len], config.repetition_penalty)

            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)

            # Store in pre-allocated buffer (no allocation)
            all_ids[:, current_len] = next_token.squeeze(-1)
            current_len += 1

            # Queue next forward pass BEFORE extracting token value.
            # On MPS this keeps the GPU busy while we sync for the yield.
            logits = self.model(next_token, kv_caches=kv_caches)

            # Extract token (forces sync, but next forward is already queued)
            token_id: int = int(next_token.item())

            if config.eos_token_id is not None and token_id == config.eos_token_id:
                break

            yield token_id
    
    @torch.inference_mode()
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
        max_total_len: int = max_prompt_len + config.max_new_tokens

        # Pad prompts to same length (left-padding for causal LM)
        padded_prompts: list[torch.Tensor] = []
        for p in prompts:
            pad_len: int = max_prompt_len - len(p)
            if pad_len > 0:
                padding: torch.Tensor = torch.full((pad_len,), config.pad_token_id, dtype=p.dtype, device=device)
                padded_prompts.append(torch.cat([padding, p]))
            else:
                padded_prompts.append(p.to(device))

        # Pre-allocate token and mask buffers (like KV cache — fill by index)
        all_ids: torch.Tensor = torch.full(
            (batch_size, max_total_len), config.pad_token_id,
            dtype=padded_prompts[0].dtype, device=device,
        )
        all_ids[:, :max_prompt_len] = torch.stack(padded_prompts)

        all_mask: torch.Tensor = torch.zeros(
            batch_size, max_total_len, dtype=torch.long, device=device,
        )
        all_mask[:, :max_prompt_len] = (all_ids[:, :max_prompt_len] != config.pad_token_id).long()
        current_len: int = max_prompt_len

        # Track which sequences have finished
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Create KV caches
        kv_caches: list[KVCache] = self.model.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_total_len,
        )

        # Create sampler
        sampler: Sampler = Sampler(
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )

        # Prefill
        logits: torch.Tensor = self.model(
            all_ids[:, :max_prompt_len], kv_caches=kv_caches,
            attention_mask=all_mask[:, :max_prompt_len],
        )

        # Generation loop
        generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]

            # Apply repetition penalty (ignore pad token)
            Sampler.apply_repetition_penalty(
                next_logits, all_ids[:, :current_len], config.repetition_penalty,
                ignore_token_id=config.pad_token_id
            )

            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)

            # Store in pre-allocated buffers (no allocation)
            all_ids[:, current_len] = next_token.squeeze(-1)
            all_mask[:, current_len] = 1
            current_len += 1

            # Store generated tokens (only for unfinished sequences)
            for i in range(batch_size):
                if not finished[i]:
                    generated_tokens[i].append(int(next_token[i].item()))

            # Check for EOS
            if config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break

            logits = self.model(
                next_token, kv_caches=kv_caches,
                attention_mask=all_mask[:, :current_len],
            )

        # Return original prompts + generated tokens (without padding)
        results: list[torch.Tensor] = []
        for i, prompt in enumerate(prompts):
            generated: torch.Tensor = torch.tensor(generated_tokens[i], dtype=prompt.dtype, device=device)
            results.append(torch.cat([prompt, generated]))

        return results
