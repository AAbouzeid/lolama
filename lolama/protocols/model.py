"""Model-related protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from ..model.config import LlamaConfig
    from ..model.vlm_config import VLMConfig
    from ..model.kv_cache import KVCache


@runtime_checkable
class GenerativeModel(Protocol):
    """Protocol defining what a model needs to support text generation.

    Any model implementing this protocol can be used with TextGenerator.
    This decouples generation logic from specific model implementations.
    Both text-only models (Llama) and vision-language models (LLaVA)
    satisfy this protocol.

    Example:
        class MyCustomModel(nn.Module):
            config: LlamaConfig

            def __call__(self, input_ids, kv_caches=None, ...):
                ...

            def create_kv_caches(self, batch_size, max_seq_len):
                ...

        # MyCustomModel satisfies GenerativeModel protocol
        generator = TextGenerator(MyCustomModel())
    """

    config: LlamaConfig | VLMConfig

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs (B, L). Mutually exclusive with inputs_embeds.
            pixel_values: Optional (B, 3, H, W) images for VLMs.
            inputs_embeds: Optional (B, L, D) pre-computed embeddings.
            kv_caches: Optional KV caches for incremental decoding.
            attention_mask: Optional attention mask (B, L).

        Returns:
            Logits tensor (B, L, vocab_size)
        """
        ...

    def create_kv_caches(self, batch_size: int, max_seq_len: int) -> list[KVCache]:
        """Create KV caches for generation.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length

        Returns:
            List of KVCache, one per layer
        """
        ...

    def reset_image_cache(self) -> None:
        """Reset cached image state between generations.

        Text-only models implement this as a no-op.
        VLMs use it to clear the flag that prevents re-encoding images.
        """
        ...

    def eval(self) -> GenerativeModel:
        """Set model to evaluation mode."""
        ...

    def parameters(self):
        """Return model parameters (used for device detection)."""
        ...
