import pytest

from lolama.model.generation_config import GenerationConfig


def test_valid_config():
    config = GenerationConfig(max_new_tokens=10, temperature=0.8)
    assert config.max_new_tokens == 10


def test_max_new_tokens_invalid():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)


def test_temperature_zero_is_greedy():
    config = GenerationConfig(temperature=0.0)
    assert config.do_sample is False
    assert config.temperature == 1.0


def test_temperature_negative_invalid():
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-0.1)


def test_top_k_invalid():
    with pytest.raises(ValueError):
        GenerationConfig(top_k=0)


def test_top_p_invalid_zero():
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0.0)


def test_top_p_invalid_above_one():
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.1)


def test_repetition_penalty_invalid():
    with pytest.raises(ValueError):
        GenerationConfig(repetition_penalty=0.5)


def test_greedy_classmethod():
    config = GenerationConfig.greedy(max_new_tokens=20)
    assert config.do_sample is False
    assert config.max_new_tokens == 20


def test_sampling_classmethod():
    config = GenerationConfig.sampling(max_new_tokens=30, temperature=0.7, top_p=0.9)
    assert config.do_sample is True
    assert config.temperature == 0.7
    assert config.top_p == 0.9
