"""Text generation utilities - separates generation logic from model."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from .generation_config import GenerationConfig
from .kv_cache import KVCache
from .sampler import Sampler
from ..protocols import GenerativeModel


def _create_kv_caches_safe(
    model: GenerativeModel,
    batch_size: int,
    max_seq_len: int,
) -> list[KVCache]:
    """Create KV caches with an actionable error on OOM."""
    try:
        return model.create_kv_caches(batch_size=batch_size, max_seq_len=max_seq_len)
    except (RuntimeError, torch.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            raise RuntimeError(
                f"Out of memory allocating KV cache "
                f"(batch_size={batch_size}, max_seq_len={max_seq_len}). "
                f"Try reducing --max-tokens, using --quantize, or a smaller model."
            ) from e
        raise


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
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive text generation with pre-allocated KV cache.

        Args:
            input_ids: Input token IDs (B, L)
            config: Generation configuration (preferred)
            pixel_values: Optional (B, 3, H, W) image tensor for VLMs.
                         Only used on first forward pass (cached internally after).
            attention_mask: Optional (B, L) mask with 1=attend, 0=padding.
                          Propagated through prefill and decode steps so padded
                          positions are never attended to. Cannot be combined
                          with pixel_values.
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

        # Validate total length fits within model's context window
        model_config = self.config
        max_seq_len: int = getattr(model_config, 'max_seq_len', 0) or getattr(getattr(model_config, 'llm_config', None), 'max_seq_len', 0)
        if max_seq_len > 0 and max_len > max_seq_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) + max_new_tokens ({config.max_new_tokens}) = {max_len} "
                f"exceeds model max_seq_len ({max_seq_len})"
            )

        # Validate: attention_mask with pixel_values is unsupported
        if attention_mask is not None and pixel_values is not None:
            raise ValueError(
                "attention_mask is not supported with pixel_values. "
                "The model handles masking internally for VLM inputs."
            )

        # Track attention mask when provided (None preserves the is_causal fast path)
        if attention_mask is not None:
            all_mask: torch.Tensor | None = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=input_ids.device,
            )
            all_mask[:, :prompt_len] = attention_mask
        else:
            all_mask = None

        # Track which sequences have finished (hit eos)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Pre-allocate token buffer (like KV cache — fill by index, never reallocate)
        all_ids: torch.Tensor = torch.empty(batch_size, max_len, dtype=input_ids.dtype, device=input_ids.device)
        all_ids[:, :prompt_len] = input_ids
        current_len: int = prompt_len

        # Create pre-allocated KV cache
        kv_caches: list[KVCache] = _create_kv_caches_safe(self.model,
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
        prefill_mask: torch.Tensor | None = all_mask[:, :prompt_len] if all_mask is not None else None
        logits: torch.Tensor
        if pixel_values is not None:
            logits = self.model(input_ids, pixel_values=pixel_values, kv_caches=kv_caches)
        else:
            logits = self.model(input_ids, kv_caches=kv_caches, attention_mask=prefill_mask)

        for i in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]

            # Apply repetition penalty over all tokens so far
            Sampler.apply_repetition_penalty(next_logits, all_ids[:, :current_len], config.repetition_penalty)

            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)

            # Store in pre-allocated buffer (no allocation)
            all_ids[:, current_len] = next_token.squeeze(-1)
            if all_mask is not None:
                all_mask[:, current_len] = 1
            current_len += 1

            # Check for EOS token
            if config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break

            # Skip forward pass on last iteration (its logits would be unused)
            if i < config.max_new_tokens - 1:
                decode_mask: torch.Tensor | None = all_mask[:, :current_len] if all_mask is not None else None
                logits = self.model(next_token, kv_caches=kv_caches, attention_mask=decode_mask)

        return all_ids[:, :current_len]
    
    @torch.inference_mode()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Iterator[int]:
        """Streaming text generation - yields tokens as they're generated.

        Args:
            input_ids: Input token IDs (B, L) - must have batch_size=1
            config: Generation configuration (preferred)
            pixel_values: Optional (B, 3, H, W) image tensor for VLMs.
                         Only used on first forward pass (cached internally after).
            attention_mask: Optional (B, L) mask with 1=attend, 0=padding.
                          Propagated through prefill and decode steps so padded
                          positions are never attended to. Cannot be combined
                          with pixel_values.
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

        # Validate total length fits within model's context window
        model_config = self.config
        max_seq_len: int = getattr(model_config, 'max_seq_len', 0) or getattr(getattr(model_config, 'llm_config', None), 'max_seq_len', 0)
        if max_seq_len > 0 and max_len > max_seq_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) + max_new_tokens ({config.max_new_tokens}) = {max_len} "
                f"exceeds model max_seq_len ({max_seq_len})"
            )

        if batch_size != 1:
            raise ValueError("Streaming only supports batch_size=1")

        # Validate: attention_mask with pixel_values is unsupported
        if attention_mask is not None and pixel_values is not None:
            raise ValueError(
                "attention_mask is not supported with pixel_values. "
                "The model handles masking internally for VLM inputs."
            )

        # Track attention mask when provided (None preserves the is_causal fast path)
        if attention_mask is not None:
            all_mask: torch.Tensor | None = torch.zeros(
                batch_size, max_len, dtype=torch.long, device=input_ids.device,
            )
            all_mask[:, :prompt_len] = attention_mask
        else:
            all_mask = None

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
        kv_caches: list[KVCache] = _create_kv_caches_safe(self.model,
            batch_size=batch_size,
            max_seq_len=max_len,
        )

        # Initial forward pass (populates cache) - include pixel_values for VLMs
        prefill_mask: torch.Tensor | None = all_mask[:, :prompt_len] if all_mask is not None else None
        logits: torch.Tensor
        if pixel_values is not None:
            logits = self.model(input_ids, pixel_values=pixel_values, kv_caches=kv_caches)
        else:
            logits = self.model(input_ids, kv_caches=kv_caches, attention_mask=prefill_mask)

        for i in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]

            # Apply repetition penalty over all tokens so far
            Sampler.apply_repetition_penalty(next_logits, all_ids[:, :current_len], config.repetition_penalty)

            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)

            # Store in pre-allocated buffer (no allocation)
            all_ids[:, current_len] = next_token.squeeze(-1)
            if all_mask is not None:
                all_mask[:, current_len] = 1
            current_len += 1

            # Extract token value
            token_id: int = int(next_token.item())

            if config.eos_token_id is not None and token_id == config.eos_token_id:
                break

            # Yield before dispatching next forward pass so generator
            # cancellation (close/throw) doesn't leave KV cache in an
            # inconsistent state from an already-issued forward pass.
            yield token_id

            # Skip forward pass on last iteration (its logits would be unused)
            if i < config.max_new_tokens - 1:
                decode_mask: torch.Tensor | None = all_mask[:, :current_len] if all_mask is not None else None
                logits = self.model(next_token, kv_caches=kv_caches, attention_mask=decode_mask)
    
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

        # Validate total length fits within model's context window
        model_config = self.config
        max_seq_len: int = getattr(model_config, 'max_seq_len', 0) or getattr(getattr(model_config, 'llm_config', None), 'max_seq_len', 0)
        if max_seq_len > 0 and max_total_len > max_seq_len:
            raise ValueError(
                f"max_prompt_len ({max_prompt_len}) + max_new_tokens ({config.max_new_tokens}) = {max_total_len} "
                f"exceeds model max_seq_len ({max_seq_len})"
            )

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
        # Build mask from actual prompt lengths (not pad_token_id comparison,
        # which would break if a real token ID equals pad_token_id)
        for b_idx, p_len in enumerate(prompt_lengths):
            pad_len: int = max_prompt_len - p_len
            all_mask[b_idx, pad_len:max_prompt_len] = 1
        current_len: int = max_prompt_len

        # Track which sequences have finished
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Create KV caches
        kv_caches: list[KVCache] = _create_kv_caches_safe(self.model,
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

        for i in range(config.max_new_tokens):
            next_logits: torch.Tensor = logits[:, -1, :]

            # Apply repetition penalty (ignore pad token)
            Sampler.apply_repetition_penalty(
                next_logits, all_ids[:, :current_len], config.repetition_penalty,
                ignore_token_id=config.pad_token_id
            )

            # Sample next token
            next_token: torch.Tensor = sampler.sample(next_logits)

            # For finished sequences, substitute pad token so they don't
            # pollute the KV cache with garbage continuation tokens
            if finished.any():
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, config.pad_token_id),
                    next_token,
                )

            # Store in pre-allocated buffers (no allocation)
            all_ids[:, current_len] = next_token.squeeze(-1)
            # Only mark unfinished sequences as attended
            all_mask[:, current_len] = (~finished).long()
            current_len += 1

            # Store generated tokens (only for unfinished sequences)
            for b in range(batch_size):
                if not finished[b]:
                    generated_tokens[b].append(int(next_token[b].item()))

            # Check for EOS
            if config.eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break

            # Skip forward pass on last iteration (its logits would be unused)
            if i < config.max_new_tokens - 1:
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
