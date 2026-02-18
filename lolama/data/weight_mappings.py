"""Centralized HuggingFace -> lolama weight name mappings.

Single source of truth for all weight renaming. Used by:
- loader.py (LLM loading)
- vlm_loader.py (VLM loading — safetensors and HF paths)
- tests/test_parity.py (parity verification)
"""

from __future__ import annotations


# Per-layer mapping: (HF suffix, lolama suffix)
# These are the 9 weight keys inside each transformer layer.
_LLM_LAYER_PAIRS: list[tuple[str, str]] = [
    ("self_attn.q_proj.weight", "attention.q_proj.weight"),
    ("self_attn.k_proj.weight", "attention.k_proj.weight"),
    ("self_attn.v_proj.weight", "attention.v_proj.weight"),
    ("self_attn.o_proj.weight", "attention.o_proj.weight"),
    ("mlp.gate_proj.weight", "feed_forward.w_gate.weight"),
    ("mlp.up_proj.weight", "feed_forward.w_up.weight"),
    ("mlp.down_proj.weight", "feed_forward.w_down.weight"),
    ("input_layernorm.weight", "attention_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
]


def build_llm_weight_mapping(
    num_layers: int,
    hf_prefix: str = "model",
    our_prefix: str = "",
) -> dict[str, str]:
    """Build HF -> lolama weight mapping for an LLM (embed, layers, norm, lm_head).

    Args:
        num_layers: Number of transformer layers.
        hf_prefix: Prefix on HF keys (e.g. "model" for standalone LLM,
                   "model.language_model" for VLM HF state_dict,
                   "language_model.model" for VLM safetensors).
        our_prefix: Prefix on lolama keys (e.g. "" for standalone LLM,
                    "language_model" for VLM).

    Returns:
        Dict mapping HF key -> lolama key.
    """
    def _hf(suffix: str) -> str:
        return f"{hf_prefix}.{suffix}" if hf_prefix else suffix

    def _our(suffix: str) -> str:
        return f"{our_prefix}.{suffix}" if our_prefix else suffix

    mapping: dict[str, str] = {}

    # Embeddings
    mapping[_hf("embed_tokens.weight")] = _our("embed_tokens.weight")

    # Transformer layers
    for i in range(num_layers):
        for hf_suffix, our_suffix in _LLM_LAYER_PAIRS:
            mapping[_hf(f"layers.{i}.{hf_suffix}")] = _our(f"layers.{i}.{our_suffix}")

    # Final norm
    mapping[_hf("norm.weight")] = _our("norm.weight")

    # LM head (no "model." prefix in HF for standalone; varies for VLM)
    # Caller adds lm_head separately since its prefix pattern differs.

    return mapping


def build_llm_weight_mapping_with_lm_head(
    num_layers: int,
) -> dict[str, str]:
    """Full standalone LLM mapping including lm_head (the common case).

    Convenience wrapper for loader.py and test_parity.py.
    """
    mapping = build_llm_weight_mapping(num_layers, hf_prefix="model", our_prefix="")
    mapping["lm_head.weight"] = "lm_head.weight"
    return mapping
