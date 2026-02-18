"""Weight loading from HuggingFace with local-first resolution."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from ..model import Llama, LlamaConfig
from ..utils.logging import get_data_logger
from .registry import MODEL_REGISTRY, WEIGHTS_DIR

logger = get_data_logger()


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
        # Preferred location: registry folder name
        local_path = WEIGHTS_DIR / info["folder"]
        if not local_path.exists():
            # Fallback: auto-saved folder from HF name
            auto_folder = info["hf_name"].replace("/", "_").replace("\\", "_")
            auto_path = WEIGHTS_DIR / auto_folder
            if auto_path.exists() and any(auto_path.iterdir()):
                local_path = auto_path
            else:
                local_path = None
        return {
            "local_path": local_path,
            "hf_name": info["hf_name"],
            "trust_remote_code": info["trust_remote_code"],
            "folder": info["folder"],
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


def _try_tokenizer_strategies(
    path_or_name: str | Path,
    trust_remote_code: bool,
    strategies: list[dict],
) -> AutoTokenizer:
    """Try loading a tokenizer using a ranked list of strategies.

    Each strategy dict may contain ``use_fast`` and ``local_files_only``.

    Fallback policy:
      - **OSError** (files not found): skip remaining strategies at this
        locality. Also skip fast strategies at the next locality if the
        fast tokenizer already failed with a parse error (no point retrying).
      - **Other exceptions** (parse failures): try the next strategy in
        order (typically the slow-tokenizer variant at the same locality).

    Returns the first tokenizer that loads successfully.
    Raises RuntimeError if all strategies fail.
    """
    last_error: Exception | None = None
    skip: set[int] = set()  # strategy indices to skip
    fast_parse_failed: bool = False  # fast tokenizer had a parse error

    for i, strategy in enumerate(strategies):
        if i in skip:
            continue

        try:
            tok = AutoTokenizer.from_pretrained(
                path_or_name,
                trust_remote_code=trust_remote_code,
                **strategy,
            )
            if i > 0:
                logger.debug(f"Tokenizer loaded via fallback strategy #{i}: {strategy}")
            return tok
        except OSError as e:
            last_error = e
            logger.debug(f"Tokenizer strategy #{i} ({strategy}) OSError: {e}")
            # Files missing at this locality — skip all remaining
            # strategies here, and skip fast at the next locality if
            # fast already failed with a parse error.
            cur_local = strategy.get("local_files_only", False)
            for j in range(i + 1, len(strategies)):
                sj = strategies[j]
                if sj.get("local_files_only", False) == cur_local:
                    skip.add(j)
                elif fast_parse_failed and sj.get("use_fast") is not False:
                    skip.add(j)
            continue
        except Exception as e:
            last_error = e
            logger.debug(f"Tokenizer strategy #{i} ({strategy}) failed: {e}")
            if strategy.get("use_fast") is not False:
                fast_parse_failed = True
            continue

    raise RuntimeError(
        f"Failed to load tokenizer from '{path_or_name}'. "
        f"Installed transformers={transformers.__version__}. "
        "Please use transformers>=4.39.0 (and a recent tokenizers build), "
        "or re-download model tokenizer files."
    ) from last_error


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """Load tokenizer with priority: weights/ -> cache -> download.

    Strategies tried in order (OSError skips locality, parse errors try slow):
      Local path:  fast -> slow
      Remote name: local-fast -> local-slow -> online-fast -> online-slow
    """
    source = resolve_model_source(model_name_or_path)

    if source["local_path"] is not None:
        tokenizer = _try_tokenizer_strategies(
            source["local_path"],
            trust_remote_code=False,
            strategies=[
                {},                     # fast (default)
                {"use_fast": False},    # slow fallback
            ],
        )
    else:
        tokenizer = _try_tokenizer_strategies(
            source["hf_name"],
            trust_remote_code=trust_remote_code,
            strategies=[
                {"local_files_only": True},                         # local fast
                {"local_files_only": True, "use_fast": False},      # local slow
                {"local_files_only": False},                        # online fast
                {"local_files_only": False, "use_fast": False},     # online slow
            ],
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class WeightLoadingError(Exception):
    """Raised when weight loading fails to meet the required threshold."""
    pass


def _load_hf_state_dict(
    hf_model_path: str | Path,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load an HF-keyed state dict, preferring direct safetensors over model instantiation."""
    import json
    path = Path(hf_model_path)

    # Single-file safetensors
    safetensors_file = path / "model.safetensors"
    if safetensors_file.exists():
        from safetensors.torch import load_file
        logger.info(f"Loading weights from {safetensors_file} (direct)")
        return load_file(str(safetensors_file))

    # Sharded safetensors (multiple files + index)
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        from safetensors.torch import load_file
        with open(index_file) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        # Collect unique shard filenames
        shard_files: set[str] = set(weight_map.values())
        logger.info(f"Loading weights from {len(shard_files)} safetensors shards (direct)")
        state_dict: dict[str, torch.Tensor] = {}
        for fname in sorted(shard_files):
            shard_path = (path / fname).resolve()
            state_dict.update(load_file(str(shard_path)))
        return state_dict

    # Fallback: instantiate HF model to extract state dict
    logger.info(f"Loading weights from {hf_model_path} (via HF model)...")
    hf_model = _try_from_pretrained(
        str(hf_model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return hf_model.state_dict()


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
    hf_state = _load_hf_state_dict(
        hf_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    # Build mapping (HF key -> Our key)
    from .weight_mappings import build_llm_weight_mapping_with_lm_head
    mapping: dict[str, str] = build_llm_weight_mapping_with_lm_head(our_model.config.num_layers)

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


def create_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float16,
) -> Llama:
    """Create model architecture from config without loading weights.

    Useful when you need the model structure (e.g., to load quantized
    weights) without the cost of loading full HF weights.

    Args:
        model_name_or_path: HuggingFace model name or local path
        dtype: Model dtype (default: float16)

    Returns:
        Llama model with empty weights (on CPU)
    """
    source = resolve_model_source(model_name_or_path)

    model_path = source["local_path"] if source["local_path"] is not None else source["hf_name"]

    config = create_config_from_hf(
        model_path,
        trust_remote_code=source["trust_remote_code"],
        local_files_only=True,
    )

    logger.info("Creating model architecture...")
    with torch.device('meta'):
        model = Llama(config, init_weights=False)

    model = model.to_empty(device='cpu').to(dtype)
    model.init_rope()

    return model


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
        # Use registry folder name when available, otherwise derive from HF name
        folder_name = source.get("folder") or hf_name.replace("/", "_").replace("\\", "_")
        save_dir = WEIGHTS_DIR / folder_name
        if not (save_dir.exists() and any(save_dir.iterdir())):
            download_model(hf_name, save_dir, trust_remote_code)
        source["local_path"] = save_dir

    model_path = source["local_path"] if source["local_path"] is not None else source["hf_name"]

    our_model = create_model(model_name_or_path, dtype=dtype)

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
