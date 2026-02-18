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


def _load_state_dict_compat(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a state dict across PyTorch versions.

    Newer PyTorch supports assign=True for lower-overhead loads; older versions
    don't. Try assign first, then fall back.
    """
    try:
        model.load_state_dict(state_dict, assign=True)
    except TypeError:
        model.load_state_dict(state_dict)


class QuantizedLinear(nn.Module):
    """Linear layer with int8 quantized weights.

    Stores weights as int8 + scale, dequantizes during forward pass.
    Uses per-channel (per-output-feature) quantization for better accuracy.

    Accelerated paths (selected automatically by device):
      - MPS:  Metal fused W8A16 kernel (dequant in registers, never materializes fp16 in DRAM)
      - CUDA: torch._int_mm W8A8 (dynamic activation quantization + int8 GEMM)
      - CPU:  Naive dequant to fp16 + F.linear (fallback)
    """

    # Class-level backend cache: None (unchecked), "metal", "int_mm", or "naive"
    _backend: dict[str, str] = {}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        num_outlier_cols: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Quantized weights: int8 (only normal columns when outliers are split out)
        normal_cols: int = in_features - num_outlier_cols
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, normal_cols, dtype=torch.int8)
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

        # Outlier buffers: only registered when num_outlier_cols > 0
        if num_outlier_cols > 0:
            self.register_buffer(
                'outlier_indices',
                torch.zeros(num_outlier_cols, dtype=torch.int32)
            )
            self.register_buffer(
                'normal_indices',
                torch.zeros(normal_cols, dtype=torch.int32)
            )
            self.register_buffer(
                'weight_outlier_fp16',
                torch.zeros(out_features, num_outlier_cols, dtype=torch.float16)
            )
        else:
            self.outlier_indices = None
            self.normal_indices = None
            self.weight_outlier_fp16 = None
    
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        dtype: torch.dtype | None = None,
        outlier_threshold: float = 0.0,
    ) -> QuantizedLinear:
        """Convert a regular Linear layer to QuantizedLinear.

        Args:
            linear: The linear layer to quantize
            dtype: Target dtype for scales and computation (default: weight dtype)
            outlier_threshold: Columns with absmax > mean + threshold*std are kept
                in fp16. 0.0 disables outlier detection.

        Returns:
            QuantizedLinear with int8 weights
        """
        if dtype is None:
            dtype = linear.weight.dtype

        device: torch.device = linear.weight.device

        # Quantize weights per output channel (do on CPU for speed, then move)
        weight: torch.Tensor = linear.weight.data.float().cpu()
        K: int = linear.in_features

        # --- Outlier detection ---
        outlier_mask: torch.Tensor | None = None
        num_outlier_cols: int = 0
        if outlier_threshold > 0.0:
            col_absmax: torch.Tensor = weight.abs().max(dim=0)[0]  # [K]
            mean: float = col_absmax.mean().item()
            std: float = col_absmax.std().item()
            outlier_mask = col_absmax > (mean + outlier_threshold * std)
            num_outlier_cols = int(outlier_mask.sum().item())

            # Cap at 10% of K
            max_outliers: int = K // 10
            if num_outlier_cols > max_outliers:
                logger.warning(
                    f"Outlier count {num_outlier_cols} exceeds 10% of K={K}, "
                    f"capping to top {max_outliers}"
                )
                _, topk_indices = col_absmax.topk(max_outliers)
                outlier_mask = torch.zeros(K, dtype=torch.bool)
                outlier_mask[topk_indices] = True
                num_outlier_cols = max_outliers

        qlayer: QuantizedLinear = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype,
            num_outlier_cols=num_outlier_cols,
        )

        if num_outlier_cols > 0 and outlier_mask is not None:
            outlier_idx: torch.Tensor = outlier_mask.nonzero(as_tuple=False).squeeze(1)
            normal_idx: torch.Tensor = (~outlier_mask).nonzero(as_tuple=False).squeeze(1)

            qlayer.outlier_indices = outlier_idx.to(torch.int32)
            qlayer.normal_indices = normal_idx.to(torch.int32)
            qlayer.weight_outlier_fp16 = weight[:, outlier_idx].to(torch.float16)

            # Quantize only normal columns
            weight_normal: torch.Tensor = weight[:, normal_idx]
            absmax: torch.Tensor = weight_normal.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale: torch.Tensor = absmax / 127.0
            weight_int8: torch.Tensor = (weight_normal / scale).round().clamp(-127, 127).to(torch.int8)
        else:
            # Standard path: quantize all columns
            absmax: torch.Tensor = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale: torch.Tensor = absmax / 127.0  # int8 range is [-127, 127]
            weight_int8: torch.Tensor = (weight / scale).round().clamp(-127, 127).to(torch.int8)

        # Set quantized weights
        qlayer.weight_int8 = weight_int8
        qlayer.weight_scale = scale.squeeze(1).to(dtype)

        if linear.bias is not None:
            qlayer.bias = linear.bias.data.to(dtype)

        # Move entire module to original device
        return qlayer.to(device)
    
    @classmethod
    def _resolve_backend(cls, device: torch.device) -> str:
        """Resolve the best backend for a device type (cached per device type)."""
        device_type: str = str(device.type)
        if device_type in cls._backend:
            return cls._backend[device_type]

        backend = "naive"

        if device_type == "mps":
            try:
                from ..metal import is_available
                if is_available():
                    backend = "metal"
                    logger.info("Using Metal fused W8A16 kernel")
            except Exception:
                pass
        elif device_type == "cuda":
            if hasattr(torch, "_int_mm"):
                backend = "int_mm"
                logger.info("Using torch._int_mm W8A8 kernel")

        cls._backend[device_type] = backend
        return backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization.

        Dispatches to the best available backend:
          1. Cached fp16 weights (--fast mode)
          2. Mixed-precision outlier path (fp16 outliers + int8 normals)
          3. Metal fused W8A16 kernel (MPS)
          4. torch._int_mm W8A8 (CUDA)
          5. Naive dequant + F.linear (CPU / fallback)
        """
        # Use cached dequantized weights if available
        if hasattr(self, '_cached_weight') and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)

        backend: str = self._resolve_backend(self.weight_int8.device)

        # Mixed-precision path: outlier columns in fp16, rest in int8
        if self.outlier_indices is not None:
            out = self._forward_mixed(x, backend)
        elif backend == "metal":
            out = self._forward_metal(x)
        elif backend == "int_mm":
            out = self._forward_int_mm(x)
        else:
            out = self._forward_naive(x)

        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """Dequant to input dtype + F.linear (CPU / fallback).

        Performs dequantization directly in the input's dtype to avoid
        materializing a full fp32 weight matrix (saves ~N*K*2 bytes per call).
        """
        dtype: torch.dtype = x.dtype
        weight: torch.Tensor = self.weight_int8.to(dtype) * self.weight_scale.to(dtype).unsqueeze(1)
        return F.linear(x, weight)

    def _forward_metal(self, x: torch.Tensor) -> torch.Tensor:
        """Metal fused W8A16 dequant+matmul (MPS)."""
        from ..metal import dequant_matmul

        return dequant_matmul(x, self.weight_int8, self.weight_scale)

    def _forward_int_mm(self, x: torch.Tensor) -> torch.Tensor:
        """torch._int_mm W8A8 path (CUDA). Dynamic per-token activation quantization."""
        # Flatten batched input [*, K] → [M, K]
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        # Dynamic per-token activation quantization
        x_scale: torch.Tensor = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        x_int8: torch.Tensor = (x / x_scale).round().clamp(-127, 127).to(torch.int8)

        # int8 GEMM: [M, K] @ [K, N] → [M, N] int32
        out_int32: torch.Tensor = torch._int_mm(x_int8, self.weight_int8.t())

        # Rescale to original dtype
        out: torch.Tensor = (out_int32.float() * x_scale * self.weight_scale.float().unsqueeze(0)).to(x.dtype)

        # Restore batch dimensions
        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out.size(-1))

        return out

    def _forward_mixed(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        """Mixed-precision path: int8 for normal columns, fp16 for outlier columns."""
        x_normal: torch.Tensor = x[..., self.normal_indices.long()].contiguous()
        x_outlier: torch.Tensor = x[..., self.outlier_indices.long()].contiguous()

        # int8 matmul on normal columns (existing kernel, unchanged)
        if backend == "metal":
            out_normal = self._forward_metal(x_normal)
        elif backend == "int_mm":
            out_normal = self._forward_int_mm(x_normal)
        else:
            out_normal = self._forward_naive(x_normal)

        # Outlier matmul is tiny (~6-20 cols). CPU backends may not support
        # fp16 GEMM, so use fp32 there and keep fp16/bf16 on accelerators.
        if x_outlier.device.type == "cpu":
            out_outlier: torch.Tensor = F.linear(
                x_outlier.float(),
                self.weight_outlier_fp16.float(),
            )
        else:
            out_outlier = F.linear(
                x_outlier.to(self.weight_outlier_fp16.dtype),
                self.weight_outlier_fp16,
            )

        return out_normal + out_outlier.to(out_normal.dtype)

    def dequantize_and_cache(self, dtype: torch.dtype = torch.float16) -> None:
        """Pre-dequantize weights and cache them for fast inference.

        This trades memory for speed - weights are stored as fp16 during inference.
        """
        weight_normal: torch.Tensor = (
            self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)
        )

        if self.outlier_indices is not None:
            # Reconstruct full [N, K] weight from normal + outlier columns
            N: int = self.out_features
            K: int = self.in_features
            full_weight: torch.Tensor = torch.zeros(N, K, dtype=torch.float32,
                                                     device=self.weight_int8.device)
            full_weight[:, self.normal_indices.long()] = weight_normal
            full_weight[:, self.outlier_indices.long()] = self.weight_outlier_fp16.float()
            self._cached_weight: torch.Tensor = full_weight.to(dtype)
        else:
            self._cached_weight: torch.Tensor = weight_normal.to(dtype).to(self.weight_int8.device)
    
    def clear_cache(self) -> None:
        """Clear cached dequantized weights to save memory."""
        if hasattr(self, '_cached_weight'):
            self._cached_weight = None

    def _apply(self, fn, recurse=True):
        """Clear cached weights when the module is moved (e.g., .to(device), .half())."""
        self.clear_cache()
        # PyTorch's Module._apply signature differs across versions; call the
        # base implementation without passing recurse for broad compatibility.
        return super()._apply(fn)
    
    def extra_repr(self) -> str:
        s = f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantized=int8'
        if self.outlier_indices is not None:
            s += f', outlier_cols={self.outlier_indices.numel()}'
        return s


def apply_quantization_structure(model: nn.Module, skip_layers: list[str] | None = None) -> nn.Module:
    """Replace Linear layers with empty QuantizedLinear (no quantization math).

    Use this when loading saved int8 weights: creates the right module
    structure so load_quantized_model() can fill in the weights.

    Args:
        model: The model to modify (in-place)
        skip_layers: List of layer name patterns to skip (e.g., ['lm_head'])

    Returns:
        The model with QuantizedLinear placeholders
    """
    skip_layers = skip_layers or []

    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_layers)

    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not should_skip(name):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            qlayer = QuantizedLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None, dtype=module.weight.dtype,
            )
            setattr(parent, parts[-1], qlayer)
            count += 1

    logger.info(f"Applied int8 structure to {count} layers (awaiting weight load)")
    return model


def quantize_model_int8(
    model: nn.Module,
    skip_layers: list[str] | None = None,
    outlier_threshold: float = 0.0,
) -> nn.Module:
    """Quantize all Linear layers in a model to int8.

    Args:
        model: The model to quantize (modified in-place)
        skip_layers: List of layer name patterns to skip (e.g., ['lm_head'])
        outlier_threshold: Keep columns with absmax > mean + threshold*std in fp16.
            0.0 disables outlier detection.

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
    if outlier_threshold > 0.0:
        logger.info(f"Outlier detection enabled (threshold={outlier_threshold})")
    logger.debug(f"Original Linear layers: {original_linear_size / 1e6:.1f} MB")

    # Replace each Linear with QuantizedLinear
    total_outlier_cols: int = 0
    for name, linear in linear_layers:
        # Navigate to parent module
        parts: list[str] = name.split('.')
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace the layer
        qlayer: QuantizedLinear = QuantizedLinear.from_linear(
            linear, outlier_threshold=outlier_threshold,
        )
        setattr(parent, parts[-1], qlayer)
        if qlayer.outlier_indices is not None:
            total_outlier_cols += qlayer.outlier_indices.numel()

    # Calculate memory of quantized layers
    quantized_size: int = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            quantized_size += m.weight_int8.numel() * 1
            quantized_size += m.weight_scale.numel() * m.weight_scale.element_size()
            if m.outlier_indices is not None:
                quantized_size += m.weight_outlier_fp16.numel() * m.weight_outlier_fp16.element_size()
                quantized_size += m.outlier_indices.numel() * m.outlier_indices.element_size()
                quantized_size += m.normal_indices.numel() * m.normal_indices.element_size()

    reduction_pct: float = (1 - quantized_size / original_linear_size) * 100
    logger.info(f"Quantized to {quantized_size / 1e6:.1f} MB ({reduction_pct:.0f}% reduction)")
    if total_outlier_cols > 0:
        logger.info(f"Total outlier columns across all layers: {total_outlier_cols}")

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
    outlier_threshold: float = 0.0,
) -> None:
    """Save a quantized model to a directory.

    Creates a standalone quantized model directory with:
    - model.safetensors: Quantized weights (mmap-friendly format)
    - quantization_config.json: Quantization metadata
    - Copied files from source: tokenizer, config, etc.

    Args:
        model: The quantized model
        output_dir: Directory to save to (e.g., 'weights/tinyllama-1.1b-int8')
        source_dir: Original model directory to copy tokenizer/config from
        outlier_threshold: The threshold used for outlier detection (persisted in config)
    """
    import json
    import shutil
    from pathlib import Path
    from safetensors.torch import save_file

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

    # Save quantized weights as safetensors (mmap-friendly, ~4x faster to load)
    weights_path: Path = output_path / "model.safetensors"
    save_file(model.state_dict(), str(weights_path))

    size_mb: float = weights_path.stat().st_size / 1e6
    logger.debug(f"Saved model.safetensors ({size_mb:.1f} MB)")

    # Save quantization config -- derive skip_layers from unquantized Linear layers
    skip_layers: list[str] = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, QuantizedLinear)
    ]
    # Detect outlier layers
    outlier_layers: list[str] = [
        name for name, module in model.named_modules()
        if isinstance(module, QuantizedLinear) and module.outlier_indices is not None
    ]
    quant_config: dict = {
        'quantization_method': 'int8_weight_only',
        'bits': 8,
        'quantized': True,
        'skip_layers': skip_layers,
        'outlier_detection': len(outlier_layers) > 0,
        'outlier_layers': len(outlier_layers),
        'outlier_threshold': outlier_threshold,
    }
    quant_config_path: Path = output_path / "quantization_config.json"
    with open(quant_config_path, 'w') as f:
        json.dump(quant_config, f, indent=2)

    logger.info(f"Saved quantized model to {output_dir}/")


def _register_outlier_buffers_from_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor],
) -> int:
    """Pre-register outlier buffers on QuantizedLinear modules before load_state_dict.

    When loading a model saved with outlier detection, the QuantizedLinear modules
    created by apply_quantization_structure() won't have outlier buffers. This scans
    the state_dict for outlier keys and registers placeholder buffers so that
    load_state_dict(..., assign=True) can assign the real tensors.

    Returns:
        Number of layers with outlier buffers registered.
    """
    # Collect module names that have outlier_indices in the state_dict
    outlier_modules: dict[str, torch.Tensor] = {}
    for key in state_dict:
        if key.endswith('.outlier_indices'):
            module_name = key.rsplit('.outlier_indices', 1)[0]
            outlier_modules[module_name] = state_dict[key]

    count: int = 0
    for module_name, outlier_idx in outlier_modules.items():
        # Navigate to the module
        parts = module_name.split('.')
        mod = model
        for part in parts:
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is None or not isinstance(mod, QuantizedLinear):
            continue

        # Register placeholder buffers (will be overwritten by assign=True).
        # Must delete plain None attributes first so register_buffer succeeds.
        normal_key = f'{module_name}.normal_indices'
        weight_key = f'{module_name}.weight_outlier_fp16'

        for attr in ('outlier_indices', 'normal_indices', 'weight_outlier_fp16'):
            if hasattr(mod, attr) and attr not in mod._buffers:
                delattr(mod, attr)

        if normal_key in state_dict:
            mod.register_buffer('outlier_indices', torch.zeros_like(outlier_idx))
            mod.register_buffer('normal_indices', torch.zeros_like(state_dict[normal_key]))
        if weight_key in state_dict:
            mod.register_buffer('weight_outlier_fp16', torch.zeros_like(state_dict[weight_key]))

        # Resize weight_int8 placeholder to match the saved (smaller) shape
        weight_int8_key = f'{module_name}.weight_int8'
        if weight_int8_key in state_dict:
            saved_shape = state_dict[weight_int8_key].shape
            if mod.weight_int8.shape != saved_shape:
                mod.register_buffer('weight_int8', torch.zeros(saved_shape, dtype=torch.int8))

        count += 1

    if count > 0:
        logger.info(f"Registered outlier buffers for {count} layers from state_dict")
    return count


def load_quantized_model(
    model_dir: str,
    model: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """Load quantized weights from a directory into a model.

    Supports safetensors (fast, mmap) with fallback to legacy .pt format.
    When device != "cpu", loads tensors directly to the target device and
    uses assign=True to avoid redundant CPU allocation.

    Note: The model architecture must already have QuantizedLinear layers.
    Call apply_quantization_structure() first to set up the module tree.

    Args:
        model_dir: Path to the quantized model directory
        model: Model with QuantizedLinear layers
        device: Target device to load weights onto (default: "cpu")

    Returns:
        Model with loaded quantized weights
    """
    from pathlib import Path

    model_path: Path = Path(model_dir)
    safetensors_path: Path = model_path / "model.safetensors"
    legacy_path: Path = model_path / "model.pt"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict: dict = load_file(str(safetensors_path), device=device)
        _register_outlier_buffers_from_state_dict(model, state_dict)
        _load_state_dict_compat(model, state_dict)
        logger.info(f"Loaded quantized model from {model_dir}/")
    elif legacy_path.exists():
        logger.warning(
            "Loading legacy .pt checkpoint with pickle (weights_only=False). "
            "Only load .pt files you trust. Prefer model.safetensors format."
        )
        checkpoint: dict = torch.load(legacy_path, map_location=device, weights_only=False)
        if not checkpoint.get('quantized', False):
            raise ValueError(f"{legacy_path} is not a quantized model checkpoint")
        _register_outlier_buffers_from_state_dict(model, checkpoint['state_dict'])
        _load_state_dict_compat(model, checkpoint['state_dict'])
        logger.info(f"Loaded quantized model from {model_dir}/ (legacy .pt)")
    else:
        raise ValueError(f"No model.safetensors or model.pt found in {model_dir}")

    return model


def is_quantized_model_dir(path: str) -> bool:
    """Check if a path is a quantized model directory."""
    from pathlib import Path

    model_path: Path = Path(path)
    quant_config: Path = model_path / "quantization_config.json"
    has_weights: bool = (
        (model_path / "model.safetensors").exists()
        or (model_path / "model.pt").exists()
    )

    return has_weights and quant_config.exists()
