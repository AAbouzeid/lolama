"""Weight-only Quantization for LLaMA.

Implements int8 weight quantization with on-the-fly dequantization.
This reduces memory by ~4x (fp32->int8) or ~2x (fp16->int8).

Usage:
    model = load_model(...)
    quantize_model_int8(model)  # In-place quantization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_model_logger

logger = get_model_logger()


class QuantizedLinear(nn.Module):
    """Linear layer with int8 quantized weights.
    
    Stores weights as int8 + scale, dequantizes during forward pass.
    Uses per-channel (per-output-feature) quantization for better accuracy.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Quantized weights: int8
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # Scale per output channel for dequantization
        self.register_buffer(
            'weight_scale',
            torch.ones(out_features, dtype=dtype)
        )
        
        # Optional bias (not quantized)
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None
    
    @staticmethod
    def from_linear(linear: nn.Linear, dtype: torch.dtype | None = None) -> QuantizedLinear:
        """Convert a regular Linear layer to QuantizedLinear.
        
        Args:
            linear: The linear layer to quantize
            dtype: Target dtype for scales and computation (default: weight dtype)
        
        Returns:
            QuantizedLinear with int8 weights
        """
        if dtype is None:
            dtype = linear.weight.dtype
        
        device: torch.device = linear.weight.device
        
        qlayer: QuantizedLinear = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype,
        )
        
        # Quantize weights per output channel (do on CPU for speed, then move)
        weight: torch.Tensor = linear.weight.data.float().cpu()
        
        # Per-channel absmax quantization
        # Scale each row (output channel) independently
        absmax: torch.Tensor = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale: torch.Tensor = absmax / 127.0  # int8 range is [-127, 127]
        
        # Quantize: weight_int8 = round(weight / scale)
        weight_int8: torch.Tensor = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        
        # Set quantized weights
        qlayer.weight_int8 = weight_int8
        qlayer.weight_scale = scale.squeeze(1).to(dtype)
        
        if linear.bias is not None:
            qlayer.bias = linear.bias.data.to(dtype)
        
        # Move entire module to original device
        return qlayer.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        # Use cached dequantized weights if available
        if hasattr(self, '_cached_weight') and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        
        # Dequantize weights: weight = weight_int8 * scale
        # Use float32 for dequantization to avoid precision issues, then convert
        weight_f32: torch.Tensor = self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)
        weight: torch.Tensor = weight_f32.to(x.dtype)
        
        return F.linear(x, weight, self.bias)
    
    def dequantize_and_cache(self, dtype: torch.dtype = torch.float16) -> None:
        """Pre-dequantize weights and cache them for fast inference.
        
        This trades memory for speed - weights are stored as fp16 during inference.
        """
        weight_f32: torch.Tensor = self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)
        self._cached_weight: torch.Tensor = weight_f32.to(dtype).to(self.weight_int8.device)
    
    def clear_cache(self) -> None:
        """Clear cached dequantized weights to save memory."""
        if hasattr(self, '_cached_weight'):
            self._cached_weight = None
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantized=int8'


def quantize_model_int8(model: nn.Module, skip_layers: list[str] | None = None) -> nn.Module:
    """Quantize all Linear layers in a model to int8.
    
    Args:
        model: The model to quantize (modified in-place)
        skip_layers: List of layer name patterns to skip (e.g., ['lm_head'])
    
    Returns:
        The quantized model (same object, modified in-place)
    """
    skip_layers = skip_layers or []
    
    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_layers)
    
    # Find all Linear layers and calculate original size
    linear_layers: list[tuple[str, nn.Linear]] = []
    original_linear_size: int = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            linear_layers.append((name, module))
            # Track original size before we replace them
            original_linear_size += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                original_linear_size += module.bias.numel() * module.bias.element_size()
    
    logger.info(f"Quantizing {len(linear_layers)} Linear layers to int8...")
    logger.debug(f"Original Linear layers: {original_linear_size / 1e6:.1f} MB")
    
    # Replace each Linear with QuantizedLinear
    for name, linear in linear_layers:
        # Navigate to parent module
        parts: list[str] = name.split('.')
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the layer
        qlayer: QuantizedLinear = QuantizedLinear.from_linear(linear)
        setattr(parent, parts[-1], qlayer)
    
    # Calculate memory of quantized layers
    quantized_size: int = sum(
        m.weight_int8.numel() * 1 + m.weight_scale.numel() * m.weight_scale.element_size()
        for m in model.modules() if isinstance(m, QuantizedLinear)
    )
    
    reduction_pct: float = (1 - quantized_size / original_linear_size) * 100
    logger.info(f"Quantized to {quantized_size / 1e6:.1f} MB ({reduction_pct:.0f}% reduction)")
    
    return model


def dequantize_model_for_inference(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """Pre-dequantize all QuantizedLinear layers for fast inference.
    
    This caches dequantized fp16 weights in memory for faster forward passes.
    Trades memory for speed - use when you want storage savings but normal inference speed.
    
    Args:
        model: Model with QuantizedLinear layers
        dtype: Target dtype for dequantized weights
    
    Returns:
        The same model with cached dequantized weights
    """
    count: int = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.dequantize_and_cache(dtype)
            count += 1
    
    if count > 0:
        logger.info(f"Pre-dequantized {count} layers for fast inference")
    
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB (parameters + buffers)."""
    param_size: int = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size: int = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1e6


def save_quantized_model(
    model: nn.Module,
    output_dir: str,
    source_dir: str | None = None,
) -> None:
    """Save a quantized model to a directory.
    
    Creates a standalone quantized model directory with:
    - model.pt: Quantized weights
    - quantization_config.json: Quantization metadata
    - Copied files from source: tokenizer, config, etc.
    
    Args:
        model: The quantized model
        output_dir: Directory to save to (e.g., 'weights/tinyllama-1.1b-int8')
        source_dir: Original model directory to copy tokenizer/config from
    """
    import json
    import shutil
    from pathlib import Path
    
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy tokenizer and config files from source
    if source_dir is not None:
        source_path: Path = Path(source_dir)
        files_to_copy: list[str] = [
            'tokenizer.json',
            'tokenizer_config.json', 
            'special_tokens_map.json',
            'chat_template.jinja',
            'generation_config.json',
            'config.json',
        ]
        for filename in files_to_copy:
            src_file: Path = source_path / filename
            if src_file.exists():
                shutil.copy2(src_file, output_path / filename)
                logger.debug(f"Copied {filename}")
    
    # Save quantized weights
    weights_path: Path = output_path / "model.pt"
    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config.__dict__ if hasattr(model, 'config') else None,
        'quantized': True,
        'quantization': {
            'method': 'int8_weight_only',
            'bits': 8,
        },
    }, weights_path)
    
    size_mb: float = weights_path.stat().st_size / 1e6
    logger.debug(f"Saved model.pt ({size_mb:.1f} MB)")
    
    # Save quantization config
    quant_config: dict = {
        'quantization_method': 'int8_weight_only',
        'bits': 8,
        'skip_layers': ['lm_head', 'embed_tokens'],
    }
    quant_config_path: Path = output_path / "quantization_config.json"
    with open(quant_config_path, 'w') as f:
        json.dump(quant_config, f, indent=2)
    
    logger.info(f"Saved quantized model to {output_dir}/")


def load_quantized_model(
    model_dir: str,
    model: nn.Module,
) -> nn.Module:
    """Load quantized weights from a directory into a model.
    
    Note: The model architecture must already have QuantizedLinear layers.
    Call quantize_model_int8() first to convert the architecture, then load weights.
    
    Args:
        model_dir: Path to the quantized model directory
        model: Model with QuantizedLinear layers (call quantize_model_int8 first)
    
    Returns:
        Model with loaded quantized weights
    """
    from pathlib import Path
    
    model_path: Path = Path(model_dir)
    weights_path: Path = model_path / "model.pt"
    
    if not weights_path.exists():
        raise ValueError(f"No model.pt found in {model_dir}")
    
    checkpoint: dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if not checkpoint.get('quantized', False):
        raise ValueError(f"{weights_path} is not a quantized model checkpoint")
    
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded quantized model from {model_dir}/")
    
    return model


def is_quantized_model_dir(path: str) -> bool:
    """Check if a path is a quantized model directory."""
    from pathlib import Path
    
    model_path: Path = Path(path)
    weights_file: Path = model_path / "model.pt"
    quant_config: Path = model_path / "quantization_config.json"
    
    return weights_file.exists() and quant_config.exists()
