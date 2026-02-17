import pytest

from lolama.model.config import LlamaConfig
from lolama.model.vlm_config import VisionConfig, VLMConfig
from lolama.model.llama import Llama
from lolama.model.llava import LLaVA
from lolama.utils.rope import precompute_rope_frequencies


@pytest.fixture
def tiny_config():
    return LlamaConfig(
        vocab_size=128,
        d_model=64,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        hidden_dim=172,
        max_seq_len=64,
    )


@pytest.fixture
def tiny_model(tiny_config):
    model = Llama(tiny_config)
    model.eval()
    return model


@pytest.fixture
def rope_freqs(tiny_config):
    head_dim = tiny_config.d_model // tiny_config.num_heads
    cos, sin = precompute_rope_frequencies(
        dim=head_dim,
        max_seq_len=tiny_config.max_seq_len,
        base=tiny_config.rope_base,
    )
    return cos, sin


# VLM fixtures
@pytest.fixture
def tiny_vision_config():
    """Small vision config for testing."""
    return VisionConfig(
        image_size=56,  # Small for fast tests
        patch_size=14,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
    )


@pytest.fixture
def tiny_vlm_config(tiny_config, tiny_vision_config):
    """Small VLM config for testing."""
    # Adjust LLM config for VLM (vocab needs image token)
    llm_config = LlamaConfig(
        vocab_size=128,
        d_model=64,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        hidden_dim=172,
        max_seq_len=128,  # Longer for image tokens
    )
    return VLMConfig(
        vision_config=tiny_vision_config,
        llm_config=llm_config,
        projector_hidden_size=64,
        image_token_id=100,  # Within vocab
    )


@pytest.fixture
def tiny_llava(tiny_vlm_config):
    """Small LLaVA model for testing."""
    model = LLaVA(tiny_vlm_config)
    model.eval()
    return model
