"""Quantization correctness tests.

Verifies int8 weight quantization roundtrip accuracy, outlier-aware
mixed-precision, and end-to-end model quantization.
"""

import json

import torch
import torch.nn as nn

from lolama.model.quantize import (
    QuantizedLinear,
    quantize_model_int8,
    apply_quantization_structure,
    save_quantized_model,
    load_quantized_model,
    is_quantized_model_dir,
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


class TestQuantizationRoundTrip:
    """Save/load round-trip tests for quantized models."""

    def test_save_load_weight_exactness(self, tiny_model, tiny_config, tmp_path):
        """Dequantized weights should be identical after save/load round-trip."""
        torch.manual_seed(42)
        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])

        # Capture weights before save
        pre_save_state = {k: v.clone() for k, v in tiny_model.state_dict().items()}

        save_quantized_model(tiny_model, str(tmp_path))

        # Load into a fresh model
        from lolama.model.llama import Llama
        model2 = Llama(tiny_config)
        apply_quantization_structure(model2, skip_layers=["lm_head", "embed_tokens"])
        load_quantized_model(str(tmp_path), model2)

        # Every tensor in state_dict should match exactly
        post_load_state = model2.state_dict()
        assert set(pre_save_state.keys()) == set(post_load_state.keys())
        for key in pre_save_state:
            assert torch.equal(pre_save_state[key], post_load_state[key]), (
                f"Mismatch in {key}: shapes {pre_save_state[key].shape} vs {post_load_state[key].shape}"
            )

    def test_round_trip_forward_identical(self, tiny_model, tiny_config, tmp_path):
        """Forward pass after save/load should match pre-save output."""
        torch.manual_seed(42)
        ids = torch.randint(0, tiny_config.vocab_size, (1, 8))

        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        with torch.no_grad():
            expected = tiny_model(ids)

        save_quantized_model(tiny_model, str(tmp_path))

        from lolama.model.llama import Llama
        model2 = Llama(tiny_config)
        apply_quantization_structure(model2, skip_layers=["lm_head", "embed_tokens"])
        load_quantized_model(str(tmp_path), model2)
        model2.eval()

        with torch.no_grad():
            actual = model2(ids)

        assert torch.allclose(expected, actual, atol=1e-6)

    def test_outlier_metadata_preserved(self, tiny_model, tiny_config, tmp_path):
        """Outlier indices/weights should persist through save/load."""
        torch.manual_seed(42)

        # Inject outliers into a specific layer to guarantee detection
        for name, module in tiny_model.named_modules():
            if isinstance(module, nn.Linear) and "q_proj" in name:
                module.weight.data[:, 0] *= 100
                break

        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"],
                            outlier_threshold=3.0)

        # Verify outliers exist before save
        has_outliers = any(
            isinstance(m, QuantizedLinear) and m.outlier_indices is not None
            for m in tiny_model.modules()
        )
        assert has_outliers, "Expected at least one layer with outliers"

        save_quantized_model(tiny_model, str(tmp_path), outlier_threshold=3.0)

        from lolama.model.llama import Llama
        model2 = Llama(tiny_config)
        apply_quantization_structure(model2, skip_layers=["lm_head", "embed_tokens"])
        load_quantized_model(str(tmp_path), model2)

        # Verify outliers survived round-trip
        has_outliers_after = any(
            isinstance(m, QuantizedLinear) and m.outlier_indices is not None
            for m in model2.modules()
        )
        assert has_outliers_after, "Outlier metadata lost during save/load"

    def test_config_json_valid(self, tiny_model, tmp_path):
        """quantization_config.json should contain expected fields."""
        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        save_quantized_model(tiny_model, str(tmp_path), outlier_threshold=6.0)

        config_path = tmp_path / "quantization_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert config["quantization_method"] == "int8_weight_only"
        assert config["bits"] == 8
        assert config["quantized"] is True
        assert isinstance(config["skip_layers"], list)
        assert config["outlier_threshold"] == 6.0

    def test_safetensors_format(self, tiny_model, tmp_path):
        """Saved model should use safetensors format."""
        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        save_quantized_model(tiny_model, str(tmp_path))

        assert (tmp_path / "model.safetensors").exists()
        assert not (tmp_path / "model.pt").exists()

    def test_is_quantized_model_dir(self, tiny_model, tmp_path):
        """is_quantized_model_dir should return True after save."""
        assert not is_quantized_model_dir(str(tmp_path))

        quantize_model_int8(tiny_model, skip_layers=["lm_head", "embed_tokens"])
        save_quantized_model(tiny_model, str(tmp_path))

        assert is_quantized_model_dir(str(tmp_path))
