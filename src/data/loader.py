"""Weight loading from HuggingFace with local-first resolution."""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model import Llama, LlamaConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

MODEL_REGISTRY = {
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "folder": "tinyllama-1.1b",
        "trust_remote_code": False,
    },
    "phi2": {
        "hf_name": "microsoft/phi-2",
        "folder": "phi-2",
        "trust_remote_code": True,
    },
    "llama7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "folder": "llama-7b",
        "trust_remote_code": False,
    },
}


def resolve_model_source(model_name_or_path):
    """
    Resolve model source using priority:
      1) weights/ directory
      2) HuggingFace cache
      3) Download from HuggingFace
    """
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

    # Fallback: treat as HF name
    return {
        "local_path": None,
        "hf_name": model_name_or_path,
        "trust_remote_code": False,
    }


def _try_from_pretrained(model_name, trust_remote_code, local_files_only):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )


def load_tokenizer(model_name_or_path, trust_remote_code=False):
    """
    Load tokenizer with priority:
      weights/ -> cache -> download
    """
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


def load_weights_from_hf(our_model, hf_model_path, trust_remote_code=False, local_files_only=False):
    """Load weights from HuggingFace model into our model."""
    print(f"Loading weights from {hf_model_path}...")
    hf_model = _try_from_pretrained(
        hf_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    
    hf_state = hf_model.state_dict()
    
    print(f"HF model has {len(hf_state)} parameters")
    print(f"Our model has {len(list(our_model.state_dict().keys()))} parameters")
    print()
    
    # Build mapping (HF key -> Our key)
    mapping = {}
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
    
    # Build new state dict with HF weights mapped to our keys
    new_state_dict = {}
    matched = 0
    shape_mismatches = []
    
    for hf_key, our_key in mapping.items():
        if hf_key not in hf_state:
            continue
        
        hf_tensor = hf_state[hf_key]
        
        if our_key not in our_state:
            continue
        
        our_tensor = our_state[our_key]
        
        if hf_tensor.shape != our_tensor.shape:
            shape_mismatches.append({
                'key': our_key,
                'hf_shape': hf_tensor.shape,
                'our_shape': our_tensor.shape
            })
            continue
        
        new_state_dict[our_key] = hf_tensor.to(our_tensor.dtype)
        matched += 1
    
    # Load all weights at once
    missing, unexpected = our_model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"   Note: {len(missing)} keys not in new_state_dict (expected for RoPE buffers)")
    
    print(f"✅ Successfully loaded {matched}/{len(mapping)} weights")
    
    if shape_mismatches:
        print(f"❌ {len(shape_mismatches)} shape mismatches!")
        for m in shape_mismatches[:5]:
            print(f"     {m['key']}: HF={m['hf_shape']} vs Ours={m['our_shape']}")
    
    if matched == len(mapping):
        print("\n🎉 All weights loaded successfully!")
    
    return our_model


def create_config_from_hf(hf_model_name, trust_remote_code=False, local_files_only=False):
    """Create LlamaConfig from HuggingFace model."""
    from transformers import AutoConfig
    
    print(f"Loading config from {hf_model_name}...")
    hf_config = AutoConfig.from_pretrained(
        hf_model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    
    vocab_size = hf_config.vocab_size
    d_model = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, 'num_key_value_heads', num_heads)
    num_layers = hf_config.num_hidden_layers
    hidden_dim = hf_config.intermediate_size
    max_seq_len = getattr(hf_config, 'max_position_embeddings', 2048)
    eps = getattr(hf_config, 'rms_norm_eps', 1e-6)
    tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)
    
    config = LlamaConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        eps=eps,
        tie_word_embeddings=tie_word_embeddings,
    )
    
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads} (Q heads)")
    print(f"  num_kv_heads: {num_kv_heads} (K/V heads - GQA)")
    print(f"  num_layers: {num_layers}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  tie_word_embeddings: {tie_word_embeddings}")
    print()
    
    return config


def load_model(model_name_or_path, device="cpu"):
    """
    Load a pretrained model from HuggingFace.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load on
    
    Returns:
        Llama model with loaded weights
    """
    print("=" * 60)
    print(f"Loading Model: {model_name_or_path}")
    print("=" * 60)
    print()

    source = resolve_model_source(model_name_or_path)
    model_path = source["local_path"] if source["local_path"] is not None else source["hf_name"]
    trust_remote_code = source["trust_remote_code"]

    # Try local cache first, then download if needed
    try:
        config = create_config_from_hf(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=source["local_path"] is None,
        )
    except OSError:
        config = create_config_from_hf(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=False,
        )
    
    print("Creating model architecture...")
    our_model = Llama(config)
    
    total_params = sum(p.numel() for p in our_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()
    
    if source["local_path"] is not None:
        our_model = load_weights_from_hf(
            our_model,
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    else:
        try:
            our_model = load_weights_from_hf(
                our_model,
                model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except OSError:
            our_model = load_weights_from_hf(
                our_model,
                model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=False,
            )
    
    print(f"\nMoving model to {device}...")
    our_model = our_model.to(device)
    
    print("\n" + "=" * 60)
    print("Model ready!")
    print("=" * 60)
    
    return our_model
