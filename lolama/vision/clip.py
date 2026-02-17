"""CLIP Vision Transformer for LLaVA.

Implements the vision encoder from CLIP (ViT-L/14 @ 336px).
This is a standard Vision Transformer with:
- Patch embeddings (14x14 patches from 336x336 images = 576 patches)
- Learnable CLS token + position embeddings
- Pre-LayerNorm transformer blocks
- No RoPE (uses absolute position embeddings)
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.vlm_config import VisionConfig


class CLIPVisionEmbeddings(nn.Module):
    """Patch + position embeddings for CLIP.

    Converts image into patch embeddings and adds position information.

    Input: (batch, 3, H, W) image
    Output: (batch, num_patches+1, hidden_size) embeddings
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config

        # Patch embedding: Conv2d is equivalent to linear projection of flattened patches
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # Learnable CLS token
        self.class_embedding = nn.Parameter(torch.zeros(config.hidden_size))

        # Position embeddings (CLS + patches)
        self.position_embedding = nn.Embedding(
            config.num_positions, config.hidden_size
        )

        # Register position IDs buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.num_positions).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, H, W) normalized images

        Returns:
            (batch, num_patches+1, hidden_size) patch embeddings with positions
        """
        batch_size = pixel_values.shape[0]

        # Patch embeddings: (B, 3, H, W) -> (B, hidden_size, H//patch, W//patch)
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions: (B, hidden_size, num_patches_h, num_patches_w) -> (B, num_patches, hidden_size)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Prepend CLS token: (B, 1, hidden_size)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-head self-attention for CLIP.

    Standard attention without RoPE (CLIP uses absolute position embeddings).
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # QKV projections (combined for efficiency)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size) attention output
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention: (B, L, H, D) -> (B, H, L, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # Reshape back: (B, H, L, D) -> (B, L, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    """Feed-forward network for CLIP with quick GELU activation."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")  # Quick GELU
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """Single transformer block for CLIP.

    Pre-LayerNorm architecture:
        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """Stack of CLIP transformer layers."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of (final_hidden_states, all_hidden_states or None)
        """
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


class CLIPVisionTransformer(nn.Module):
    """Complete CLIP Vision Transformer.

    Architecture:
        Image -> Patch Embed + CLS + Pos -> Transformer Layers -> LayerNorm -> Features

    For LLaVA, we typically extract features from the second-to-last layer
    (vision_feature_layer=-2) and exclude the CLS token.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = CLIPVisionEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Args:
            pixel_values: (batch, 3, H, W) normalized images
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of:
                - last_hidden_state: (batch, num_patches+1, hidden_size)
                - hidden_states: List of hidden states from each layer (if requested)
        """
        # Embed patches
        hidden_states = self.embeddings(pixel_values)

        # Pre-encoder layer norm (some CLIP variants have this)
        hidden_states = self.pre_layernorm(hidden_states)

        # Transformer encoder
        last_hidden_state, all_hidden_states = self.encoder(
            hidden_states, output_hidden_states=output_hidden_states
        )

        # Post-encoder layer norm
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state, all_hidden_states

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        vision_feature_layer: int = -2,
        vision_feature_select_strategy: str = "default",
    ) -> torch.Tensor:
        """Extract image features for LLaVA.

        Args:
            pixel_values: (batch, 3, H, W) normalized images
            vision_feature_layer: Which layer to extract features from (-2 for second-to-last)
            vision_feature_select_strategy: "default" (exclude CLS) or "full" (include CLS)

        Returns:
            (batch, num_patches, hidden_size) image features (without CLS by default)
        """
        _, hidden_states = self.forward(pixel_values, output_hidden_states=True)

        # Get features from specified layer
        image_features = hidden_states[vision_feature_layer]

        # Remove CLS token if using default strategy
        if vision_feature_select_strategy == "default":
            image_features = image_features[:, 1:]  # Remove first token (CLS)

        return image_features
