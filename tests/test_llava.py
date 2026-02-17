"""Tests for LLaVA vision-language model."""

import pytest
import torch

from lolama.model.vlm_config import VLMConfig
from lolama.model.llava import LLaVA
from lolama.model.generator import TextGenerator
from lolama.vision.processor import CLIPImageProcessor


class TestVLMConfig:
    """Tests for VLMConfig dataclass."""

    def test_default_config(self):
        config = VLMConfig()
        assert config.vision_config is not None
        assert config.llm_config is not None
        assert config.projector_hidden_size == 4096
        assert config.image_token_id == 32000
        assert config.vision_feature_layer == -2

    def test_properties(self):
        config = VLMConfig()
        assert config.vision_hidden_size == config.vision_config.hidden_size
        assert config.text_hidden_size == config.llm_config.d_model
        assert config.num_image_tokens == config.vision_config.num_patches


class TestLLaVA:
    """Tests for LLaVA model."""

    def test_init(self, tiny_vlm_config):
        model = LLaVA(tiny_vlm_config)
        assert model.vision_tower is not None
        assert model.multi_modal_projector is not None
        assert model.language_model is not None

    def test_encode_images(self, tiny_llava, tiny_vlm_config):
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3,
            tiny_vlm_config.vision_config.image_size,
            tiny_vlm_config.vision_config.image_size,
        )

        image_features = tiny_llava.encode_images(pixel_values)

        # Should be (batch, num_patches, llm_hidden_size)
        expected_patches = tiny_vlm_config.vision_config.num_patches
        expected_hidden = tiny_vlm_config.llm_config.d_model
        assert image_features.shape == (batch_size, expected_patches, expected_hidden)

    def test_forward_text_only(self, tiny_llava, tiny_vlm_config):
        """Test forward pass with only text (no image)."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        logits = tiny_llava(input_ids=input_ids)

        assert logits.shape == (batch_size, seq_len, tiny_vlm_config.llm_config.vocab_size)

    def test_forward_with_image(self, tiny_llava, tiny_vlm_config):
        """Test forward pass with image and text."""
        batch_size = 1
        image_token_id = tiny_vlm_config.image_token_id

        # Create input with image token
        # [text, <image>, text]
        input_ids = torch.tensor([[1, 2, 3, image_token_id, 5, 6]])
        pixel_values = torch.randn(
            batch_size, 3,
            tiny_vlm_config.vision_config.image_size,
            tiny_vlm_config.vision_config.image_size,
        )

        logits = tiny_llava(input_ids=input_ids, pixel_values=pixel_values)

        # Output sequence length: original - 1 (image token) + num_patches
        num_patches = tiny_vlm_config.vision_config.num_patches
        expected_seq_len = input_ids.shape[1] - 1 + num_patches
        assert logits.shape == (batch_size, expected_seq_len, tiny_vlm_config.llm_config.vocab_size)

    def test_forward_no_image_token(self, tiny_llava, tiny_vlm_config):
        """Test forward with no image token in text - uses text only path."""
        batch_size = 1
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # No image token

        # Without pixel_values, should use text-only path
        logits = tiny_llava(input_ids=input_ids)

        # When no image, sequence length unchanged
        assert logits.shape[1] == input_ids.shape[1]

    def test_create_kv_caches(self, tiny_llava, tiny_vlm_config):
        batch_size, max_seq_len = 2, 64
        kv_caches = tiny_llava.create_kv_caches(batch_size, max_seq_len)

        # Should have one cache per layer
        assert len(kv_caches) == tiny_vlm_config.llm_config.num_layers

        # Each cache should have correct shape
        head_dim = tiny_vlm_config.llm_config.d_model // tiny_vlm_config.llm_config.num_heads
        num_kv_heads = tiny_vlm_config.llm_config.num_kv_heads
        # Note: max_seq_len is increased by num_image_tokens
        expected_max_len = max_seq_len + tiny_vlm_config.num_image_tokens
        for cache in kv_caches:
            assert cache.k_cache.shape == (batch_size, num_kv_heads, expected_max_len, head_dim)

    def test_generation_with_image(self, tiny_llava, tiny_vlm_config):
        """Test generation with image input."""
        batch_size = 1
        image_token_id = tiny_vlm_config.image_token_id

        input_ids = torch.tensor([[1, image_token_id, 2]])
        pixel_values = torch.randn(
            batch_size, 3,
            tiny_vlm_config.vision_config.image_size,
            tiny_vlm_config.vision_config.image_size,
        )

        generator = TextGenerator(tiny_llava)
        output = generator.generate(
            input_ids,
            pixel_values=pixel_values,
            max_new_tokens=5,
            temperature=1.0,
        )

        # Should generate tokens
        assert output.shape[1] > input_ids.shape[1]

    def test_image_cache_reset(self, tiny_llava, tiny_vlm_config):
        """Test that image cache is properly reset between generations."""
        input_ids = torch.tensor([[1, tiny_vlm_config.image_token_id, 2]])
        pixel_values = torch.randn(
            1, 3,
            tiny_vlm_config.vision_config.image_size,
            tiny_vlm_config.vision_config.image_size,
        )

        generator = TextGenerator(tiny_llava)

        # First generation
        output1 = generator.generate(
            input_ids.clone(),
            pixel_values=pixel_values,
            max_new_tokens=3,
        )

        # Second generation with same input (cache should be reset)
        output2 = generator.generate(
            input_ids.clone(),
            pixel_values=pixel_values,
            max_new_tokens=3,
        )

        # Both should produce valid outputs
        assert output1.shape[1] > input_ids.shape[1]
        assert output2.shape[1] > input_ids.shape[1]

    def test_count_parameters(self, tiny_llava):
        params = tiny_llava.count_parameters()

        assert "total" in params
        assert "vision_tower" in params
        assert "projector" in params
        assert "language_model" in params
        assert params["total"] == params["vision_tower"] + params["projector"] + params["language_model"]

    def test_device_property(self, tiny_llava):
        assert tiny_llava.device == tiny_llava.language_model.embed_tokens.weight.device

    def test_dtype_property(self, tiny_llava):
        assert tiny_llava.dtype == tiny_llava.language_model.embed_tokens.weight.dtype


class TestMergeInputIdsWithImageFeatures:
    """Tests for the image-text merging logic."""

    def test_single_image_middle(self, tiny_llava, tiny_vlm_config):
        """Test merging with image token in the middle."""
        image_token_id = tiny_vlm_config.image_token_id
        input_ids = torch.tensor([[1, 2, image_token_id, 4, 5]])

        num_patches = tiny_vlm_config.vision_config.num_patches
        image_features = torch.randn(1, num_patches, tiny_vlm_config.llm_config.d_model)

        merged = tiny_llava._merge_input_ids_with_image_features(input_ids, image_features)

        # Original: 5 tokens, replace 1 image token with num_patches features
        expected_len = 5 - 1 + num_patches
        assert merged.shape == (1, expected_len, tiny_vlm_config.llm_config.d_model)

    def test_image_at_start(self, tiny_llava, tiny_vlm_config):
        """Test merging with image token at the start."""
        image_token_id = tiny_vlm_config.image_token_id
        input_ids = torch.tensor([[image_token_id, 2, 3, 4]])

        num_patches = tiny_vlm_config.vision_config.num_patches
        image_features = torch.randn(1, num_patches, tiny_vlm_config.llm_config.d_model)

        merged = tiny_llava._merge_input_ids_with_image_features(input_ids, image_features)

        expected_len = 4 - 1 + num_patches
        assert merged.shape == (1, expected_len, tiny_vlm_config.llm_config.d_model)

    def test_image_at_end(self, tiny_llava, tiny_vlm_config):
        """Test merging with image token at the end."""
        image_token_id = tiny_vlm_config.image_token_id
        input_ids = torch.tensor([[1, 2, 3, image_token_id]])

        num_patches = tiny_vlm_config.vision_config.num_patches
        image_features = torch.randn(1, num_patches, tiny_vlm_config.llm_config.d_model)

        merged = tiny_llava._merge_input_ids_with_image_features(input_ids, image_features)

        expected_len = 4 - 1 + num_patches
        assert merged.shape == (1, expected_len, tiny_vlm_config.llm_config.d_model)

    def test_batch_processing(self, tiny_llava, tiny_vlm_config):
        """Test merging works with batch_size > 1."""
        image_token_id = tiny_vlm_config.image_token_id
        batch_size = 2
        input_ids = torch.tensor([
            [1, image_token_id, 3],
            [4, image_token_id, 6],
        ])

        num_patches = tiny_vlm_config.vision_config.num_patches
        image_features = torch.randn(batch_size, num_patches, tiny_vlm_config.llm_config.d_model)

        merged = tiny_llava._merge_input_ids_with_image_features(input_ids, image_features)

        expected_len = 3 - 1 + num_patches
        assert merged.shape == (batch_size, expected_len, tiny_vlm_config.llm_config.d_model)


class TestLlamaInputsEmbeds:
    """Tests for Llama inputs_embeds support."""

    def test_forward_with_inputs_embeds(self, tiny_model, tiny_config):
        """Test Llama forward with inputs_embeds instead of input_ids."""
        batch_size, seq_len = 2, 10
        inputs_embeds = torch.randn(batch_size, seq_len, tiny_config.d_model)

        logits = tiny_model(inputs_embeds=inputs_embeds)

        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

    def test_cannot_specify_both(self, tiny_model, tiny_config):
        """Test that specifying both input_ids and inputs_embeds raises error."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        inputs_embeds = torch.randn(batch_size, seq_len, tiny_config.d_model)

        with pytest.raises(ValueError, match="Cannot specify both"):
            tiny_model(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def test_must_specify_one(self, tiny_model):
        """Test that specifying neither raises error."""
        with pytest.raises(ValueError, match="Must specify either"):
            tiny_model()


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, tiny_vlm_config):
        """Test complete image -> text pipeline."""
        from PIL import Image

        # Create model and processor
        model = LLaVA(tiny_vlm_config)
        model.eval()
        processor = CLIPImageProcessor(image_size=tiny_vlm_config.vision_config.image_size)

        # Create dummy image
        image = Image.new("RGB", (100, 100), color="red")

        # Process image
        processed = processor.preprocess(image)
        pixel_values = processed["pixel_values"]

        # Create input with image token
        image_token_id = tiny_vlm_config.image_token_id
        input_ids = torch.tensor([[1, image_token_id, 2, 3]])

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids=input_ids, pixel_values=pixel_values)

        # Should produce valid logits
        assert logits.shape[0] == 1
        assert logits.shape[2] == tiny_vlm_config.llm_config.vocab_size
        assert not torch.isnan(logits).any()

    def test_streaming_generation(self, tiny_llava, tiny_vlm_config):
        """Test streaming generation with image."""
        image_token_id = tiny_vlm_config.image_token_id
        input_ids = torch.tensor([[1, image_token_id, 2]])
        pixel_values = torch.randn(
            1, 3,
            tiny_vlm_config.vision_config.image_size,
            tiny_vlm_config.vision_config.image_size,
        )

        generator = TextGenerator(tiny_llava)
        tokens = list(generator.generate_stream(
            input_ids,
            pixel_values=pixel_values,
            max_new_tokens=3,
        ))

        # Should generate some tokens
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
