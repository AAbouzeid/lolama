"""Quantization correctness tests.

Verifies int8 weight quantization roundtrip accuracy, outlier-aware
mixed-precision, and end-to-end model quantization.
"""

import torch
import torch.nn as nn

from lolama.model.quantize import (
    QuantizedLinear,
    quantize_model_int8,
    get_model_size_mb,
)


class TestQuantizedLinear:
    """Unit tests for the QuantizedLinear layer."""

    def test_from_linear_close(self):
        """Quantized forward should approximate the original Linear."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 64, bias=False)
        x = torch.randn(2, 8, 128)

        with torch.no_grad():
            expected = linear(x)

        qlayer = QuantizedLinear.from_linear(linear)
        with torch.no_grad():
            actual = qlayer(x)

        rel_error = (expected - actual).abs().mean() / expected.abs().mean()
        assert rel_error < 0.05, f"Relative error {rel_error:.4f} exceeds 5%"

    def test_weight_int8_range(self):
        """Quantized weights must be in [-127, 127]."""
        linear = nn.Linear(64, 32, bias=False)
        qlayer = QuantizedLinear.from_linear(linear)
        assert qlayer.weight_int8.min() >= -127
        assert qlayer.weight_int8.max() <= 127
        assert qlayer.weight_int8.dtype == torch.int8

    def test_scale_positive(self):
        """Per-channel scales must be strictly positive."""
        linear = nn.Linear(64, 32, bias=False)
        qlayer = QuantizedLinear.from_linear(linear)
        assert (qlayer.weight_scale > 0).all()

    def test_bias_preserved(self):
        """Bias should pass through unquantized."""
        torch.manual_seed(42)
        linear = nn.Linear(64, 32, bias=True)
        qlayer = QuantizedLinear.from_linear(linear)

        assert qlayer.bias is not None
        assert torch.allclose(qlayer.bias.float(), linear.bias.data.float(), atol=1e-6)

    def test_no_bias(self):
        """Layers without bias should have bias=None."""
        linear = nn.Linear(64, 32, bias=False)
        qlayer = QuantizedLinear.from_linear(linear)
        assert qlayer.bias is None

    def test_output_shape(self):
        """Output shape must match original Linear."""
        linear = nn.Linear(128, 64, bias=False)
        qlayer = QuantizedLinear.from_linear(linear)
        x = torch.randn(4, 16, 128)
        with torch.no_grad():
            out = qlayer(x)
        assert out.shape == (4, 16, 64)


class TestOutlierQuantization:
    """Tests for LLM.int8()-style outlier-aware quantization."""

    def test_detects_outlier_column(self):
        """A column with extreme values should be flagged as outlier."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 64, bias=False)
        linear.weight.data[:, 5] *= 100  # inject outlier in column 5

        qlayer = QuantizedLinear.from_linear(linear, outlier_threshold=3.0)
        assert qlayer.outlier_indices is not None
        assert 5 in qlayer.outlier_indices.tolist()

    def test_outlier_weight_fp16(self):
        """Outlier columns should be stored in fp16."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 64, bias=False)
        linear.weight.data[:, 5] *= 100

        qlayer = QuantizedLinear.from_linear(linear, outlier_threshold=3.0)
        assert qlayer.weight_outlier_fp16 is not None
        assert qlayer.weight_outlier_fp16.dtype == torch.float16

    def test_outlier_forward_close(self):
        """Mixed-precision forward should approximate original."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 64, bias=False)
        linear.weight.data[:, 5] *= 100
        x = torch.randn(2, 8, 128)

        with torch.no_grad():
            expected = linear(x)

        qlayer = QuantizedLinear.from_linear(linear, outlier_threshold=3.0)
        with torch.no_grad():
            actual = qlayer(x)

        rel_error = (expected - actual).abs().mean() / expected.abs().mean()
        assert rel_error < 0.05, f"Relative error {rel_error:.4f} exceeds 5%"

    def test_no_outliers_when_disabled(self):
        """threshold=0.0 should produce no outlier buffers."""
        linear = nn.Linear(128, 64, bias=False)
        qlayer = QuantizedLinear.from_linear(linear, outlier_threshold=0.0)
        assert qlayer.outlier_indices is None

    def test_outlier_cap_at_10_percent(self):
        """Number of outlier columns should be capped at 10% of K."""
        torch.manual_seed(42)
        linear = nn.Linear(100, 64, bias=False)
        # Make 30 columns extreme — should be capped to 10
        linear.weight.data[:, :30] *= 100

        qlayer = QuantizedLinear.from_linear(linear, outlier_threshold=0.1)
        if qlayer.outlier_indices is not None:
            assert qlayer.outlier_indices.numel() <= 10


class TestQuantizeModel:
    """End-to-end model quantization tests."""

    def test_size_reduction(self, tiny_model):
        """Quantized model should use less memory."""
        original = get_model_size_mb(tiny_model)
        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        quantized = get_model_size_mb(tiny_model)
        assert quantized < original

    def test_forward_after_quantize(self, tiny_model, tiny_config):
        """Model should still produce valid logits after quantization."""
        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            out = tiny_model(ids)
        assert out.shape == (1, 8, tiny_config.vocab_size)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_skip_layers_respected(self, tiny_model):
        """Skipped layers should remain as nn.Linear."""
        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        assert isinstance(tiny_model.lm_head, nn.Linear)
        assert not isinstance(tiny_model.lm_head, QuantizedLinear)
