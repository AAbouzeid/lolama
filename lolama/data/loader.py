"""Weight loading from HuggingFace with local-first resolution."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model import Llama, LlamaConfig
from ..utils.logging import get_data_logger
from .registry import MODEL_REGISTRY

logger = get_data_logger()

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def resolve_model_source(model_name_or_path: str) -> dict[str, str | Path | bool | None]:
    """Resolve model source using priority: weights/ -> HF cache -> download."""
    # Explicit local path
    path = Path(model_name_or_path)
    if path.exists():
        return {
            "local_path": path,
            "hf_name": None,
            "trust_remote_code": False,
        }

    key = model_name_or_path.lower()
    if key in MODEL_REGISTRY:
        info = MODEL_REGISTRY[key]
        local_path = WEIGHTS_DIR / info["folder"]
        return {
            "local_path": local_path if local_path.exists() else None,
            "hf_name": info["hf_name"],
            "trust_remote_code": info["trust_remote_code"],
        }

    # Check if previously auto-saved to weights/ (e.g. openlm-research/open_llama_3b_v2
    # -> weights/openlm-research_open_llama_3b_v2/)
    auto_folder = model_name_or_path.replace("/", "_").replace("\\", "_")
    auto_path = WEIGHTS_DIR / auto_folder
    if auto_path.exists() and any(auto_path.iterdir()):
        return {
            "local_path": auto_path,
            "hf_name": model_name_or_path,
            "trust_remote_code": False,
        }

    # Fallback: treat as HF name (will be auto-downloaded and saved on first use)
    return {
        "local_path": None,
        "hf_name": model_name_or_path,
        "trust_remote_code": False,
    }


def _try_from_pretrained(
    model_name: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """Load tokenizer with priority: weights/ -> cache -> download."""
    source = resolve_model_source(model_name_or_path)

    if source["local_path"] is not None:
        tokenizer = AutoTokenizer.from_pretrained(source["local_path"])
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source["hf_name"],
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(
                source["hf_name"],
                trust_remote_code=trust_remote_code,
                local_files_only=False,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class WeightLoadingError(Exception):
    """Raised when weight loading fails to meet the required threshold."""
    pass


def load_weights_from_hf(
    our_model: Llama,
    hf_model_path: str | Path,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    strict_threshold: float = 0.95,
) -> Llama:
    """Load weights from HuggingFace model into our model.

    Args:
        our_model: Target model to load weights into
        hf_model_path: Path or HF model name
        trust_remote_code: Whether to trust remote code
        local_files_only: Only use local files
        strict_threshold: Minimum fraction of weights that must load successfully.
            Default 0.95 (95%). Set to 0.0 to disable strict checking.

    Returns:
        Model with loaded weights

    Raises:
        WeightLoadingError: If match rate falls below strict_threshold
    """
    logger.info(f"Loading weights from {hf_model_path}...")
    hf_model = _try_from_pretrained(
        hf_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    hf_state = hf_model.state_dict()

    # Build mapping (HF key -> Our key)
    mapping: dict[str, str] = {}
    mapping['model.embed_tokens.weight'] = 'embed_tokens.weight'

    num_layers = our_model.config.num_layers
    for i in range(num_layers):
        prefix_hf = f'model.layers.{i}'
        prefix_our = f'layers.{i}'

        mapping[f'{prefix_hf}.self_attn.q_proj.weight'] = f'{prefix_our}.attention.q_proj.weight'
        mapping[f'{prefix_hf}.self_attn.k_proj.weight'] = f'{prefix_our}.attention.k_proj.weight'
        mapping[f'{prefix_hf}.self_attn.v_proj.weight'] = f'{prefix_our}.attention.v_proj.weight'
        mapping[f'{prefix_hf}.self_attn.o_proj.weight'] = f'{prefix_our}.attention.o_proj.weight'
        mapping[f'{prefix_hf}.mlp.gate_proj.weight'] = f'{prefix_our}.feed_forward.w_gate.weight'
        mapping[f'{prefix_hf}.mlp.up_proj.weight'] = f'{prefix_our}.feed_forward.w_up.weight'
        mapping[f'{prefix_hf}.mlp.down_proj.weight'] = f'{prefix_our}.feed_forward.w_down.weight'
        mapping[f'{prefix_hf}.input_layernorm.weight'] = f'{prefix_our}.attention_norm.weight'
        mapping[f'{prefix_hf}.post_attention_layernorm.weight'] = f'{prefix_our}.ffn_norm.weight'

    mapping['model.norm.weight'] = 'norm.weight'
    mapping['lm_head.weight'] = 'lm_head.weight'

    # Get current state dict for shape comparison
    our_state = our_model.state_dict()

    # Track all issues for diagnostics
    new_state_dict: dict[str, torch.Tensor] = {}
    matched: int = 0
    missing_in_hf: list[str] = []
    missing_in_ours: list[str] = []
    shape_mismatches: list[dict] = []

    for hf_key, our_key in mapping.items():
        if hf_key not in hf_state:
            missing_in_hf.append(hf_key)
            continue

        hf_tensor = hf_state[hf_key]

        if our_key not in our_state:
            missing_in_ours.append(our_key)
            continue

        our_tensor = our_state[our_key]

        if hf_tensor.shape != our_tensor.shape:
            shape_mismatches.append({
                'hf_key': hf_key,
                'our_key': our_key,
                'hf_shape': tuple(hf_tensor.shape),
                'our_shape': tuple(our_tensor.shape),
            })
            continue

        new_state_dict[our_key] = hf_tensor.to(our_tensor.dtype)
        matched += 1

    # Calculate match rate
    total_expected = len(mapping)
    match_rate = matched / total_expected if total_expected > 0 else 0.0

    # Load weights
    missing_after_load, unexpected = our_model.load_state_dict(new_state_dict, strict=False)

    # Report results
    logger.info(f"Successfully loaded {matched}/{total_expected} weights ({match_rate:.1%})")

    if missing_after_load:
        # Filter out expected missing keys (RoPE buffers)
        rope_buffers = [k for k in missing_after_load if 'cos' in k or 'sin' in k]
        other_missing = [k for k in missing_after_load if k not in rope_buffers]
        if rope_buffers:
            logger.debug(f"{len(rope_buffers)} RoPE buffers not in checkpoint (expected)")
        if other_missing:
            logger.warning(f"{len(other_missing)} unexpected missing keys: {other_missing[:5]}")

    if missing_in_hf:
        logger.error(f"{len(missing_in_hf)} keys missing in HuggingFace model: {missing_in_hf[:5]}")

    if missing_in_ours:
        logger.error(f"{len(missing_in_ours)} keys missing in our model: {missing_in_ours[:5]}")

    if shape_mismatches:
        for m in shape_mismatches[:5]:
            logger.error(f"Shape mismatch: {m['our_key']}: HF={m['hf_shape']} vs Ours={m['our_shape']}")

    # Strict threshold check
    if match_rate < strict_threshold:
        error_msg = (
            f"Weight loading failed: {match_rate:.1%} matched, "
            f"but {strict_threshold:.0%} required. "
            f"Missing in HF: {len(missing_in_hf)}, "
            f"Missing in model: {len(missing_in_ours)}, "
            f"Shape mismatches: {len(shape_mismatches)}"
        )
        raise WeightLoadingError(error_msg)

    if matched == total_expected:
        logger.info("All weights loaded successfully!")

    return our_model


def create_config_from_hf(
    hf_model_name: str | Path,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> LlamaConfig:
    """Create LlamaConfig from HuggingFace model."""
    from transformers import AutoConfig

    logger.info(f"Loading config from {hf_model_name}...")
    hf_config = AutoConfig.from_pretrained(
        hf_model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    vocab_size: int = hf_config.vocab_size
    d_model: int = hf_config.hidden_size
    num_heads: int = hf_config.num_attention_heads
    num_kv_heads: int = getattr(hf_config, 'num_key_value_heads', num_heads)
    num_layers: int = hf_config.num_hidden_layers
    hidden_dim: int = hf_config.intermediate_size
    max_seq_len: int = getattr(hf_config, 'max_position_embeddings', 2048)
    eps: float = getattr(hf_config, 'rms_norm_eps', 1e-6)
    tie_word_embeddings: bool = getattr(hf_config, 'tie_word_embeddings', False)
    # rope_theta location varies: top-level in some models, nested in rope_parameters in others
    rope_params: dict | None = getattr(hf_config, 'rope_parameters', None)
    if rope_params and 'rope_theta' in rope_params:
        rope_base: int = int(rope_params['rope_theta'])
    else:
        rope_base: int = int(getattr(hf_config, 'rope_theta', 10000))  # LLaMA 3 uses 500000

    config: LlamaConfig = LlamaConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        eps=eps,
        tie_word_embeddings=tie_word_embeddings,
        rope_base=rope_base,
    )

    logger.debug(f"Config: vocab_size={vocab_size}, d_model={d_model}, num_heads={num_heads}, "
                 f"num_kv_heads={num_kv_heads}, num_layers={num_layers}, hidden_dim={hidden_dim}, "
                 f"rope_base={rope_base}, tie_word_embeddings={tie_word_embeddings}")

    return config


def download_model(
    hf_name: str,
    save_dir: Path,
    trust_remote_code: bool = False,
    from_cache: bool = False,
) -> None:
    """Download and save HF model + tokenizer to local weights/ directory.

    Args:
        hf_name: HuggingFace model name
        save_dir: Local directory to save to
        trust_remote_code: Whether to trust remote code
        from_cache: If True, only use local HF cache (no download)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {hf_name} to {save_dir}...")

    try:
        hf_model = _try_from_pretrained(hf_name, trust_remote_code, local_files_only=True)
    except OSError:
        if from_cache:
            raise
        hf_model = _try_from_pretrained(hf_name, trust_remote_code, local_files_only=False)

    hf_model.save_pretrained(save_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_remote_code, local_files_only=True)
    except OSError:
        if from_cache:
            raise
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_remote_code, local_files_only=False)

    tokenizer.save_pretrained(save_dir)

    total_params = sum(p.numel() for p in hf_model.parameters())
    size_mb = total_params * 2 / 1024**2
    logger.info(f"Saved to {save_dir} ({total_params:,} params, {size_mb:.1f} MB fp16)")


def load_model(
    model_name_or_path: str,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
    compile_model: bool = False,
) -> Llama:
    """Load a pretrained model from HuggingFace.

    If weights are not found locally, downloads and saves them to the
    weights/ directory for future use (no reliance on HF cache).

    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load on (default: auto-detect best available)
        dtype: Model dtype (default: float16 to match HF models)
        compile_model: If True, use torch.compile() for speedup (requires PyTorch 2.0+)

    Returns:
        Llama model with loaded weights
    """
    from ..utils.device import resolve_device

    if device is None:
        device = resolve_device()

    logger.info(f"Loading model: {model_name_or_path}")

    source = resolve_model_source(model_name_or_path)
    trust_remote_code = source["trust_remote_code"]

    # Auto-download to weights/ if not found locally
    if source["local_path"] is None and source["hf_name"] is not None:
        hf_name = source["hf_name"]
        folder_name = hf_name.replace("/", "_").replace("\\", "_")
        save_dir = WEIGHTS_DIR / folder_name
        if not (save_dir.exists() and any(save_dir.iterdir())):
            download_model(hf_name, save_dir, trust_remote_code)
        source["local_path"] = save_dir

    model_path = source["local_path"] if source["local_path"] is not None else source["hf_name"]

    config = create_config_from_hf(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    logger.info("Creating model architecture...")
    # Use meta device to skip weight initialization entirely (much faster)
    with torch.device('meta'):
        our_model = Llama(config, init_weights=False)

    # Materialize on CPU first (HF weights load to CPU), then move to target device
    our_model = our_model.to_empty(device='cpu').to(dtype)

    # Re-initialize RoPE buffers (meta device doesn't compute them)
    our_model.init_rope()

    total_params: int = sum(p.numel() for p in our_model.parameters())
    logger.info(f"Total parameters: {total_params:,}, dtype: {dtype}")

    our_model = load_weights_from_hf(
        our_model,
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    # Move to target device (skip if already on CPU)
    if device != "cpu":
        logger.info(f"Moving model to {device}...")
        our_model = our_model.to(device)

    # Optional: torch.compile() for speedup (PyTorch 2.0+)
    if compile_model:
        logger.info("Compiling model with torch.compile()...")
        our_model = torch.compile(our_model)

    logger.info("Model ready!")

    return our_model
