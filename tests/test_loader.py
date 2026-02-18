from pathlib import Path
import json

from lolama.data import loader


class _DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"


def test_load_tokenizer_local_falls_back_to_slow(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setattr(
        loader,
        "resolve_model_source",
        lambda _: {"local_path": model_dir, "hf_name": None, "trust_remote_code": False},
    )

    calls: list[dict] = []

    def _fake_from_pretrained(path, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Exception("fast tokenizer parse failure")
        return _DummyTokenizer()

    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    tok = loader.load_tokenizer("tinyllama")

    assert tok.pad_token == tok.eos_token
    assert len(calls) == 2
    assert calls[1].get("use_fast") is False


def test_load_tokenizer_local_valueerror_does_not_mutate_config(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    cfg_path = model_dir / "tokenizer_config.json"
    original = {"tokenizer_class": "SomeFutureTokenizer"}
    cfg_path.write_text(json.dumps(original))

    monkeypatch.setattr(
        loader,
        "resolve_model_source",
        lambda _: {"local_path": model_dir, "hf_name": None, "trust_remote_code": False},
    )

    calls: list[dict] = []

    def _fake_from_pretrained(path, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("unknown tokenizer class")
        return _DummyTokenizer()

    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    tok = loader.load_tokenizer("tinyllama")

    assert tok.pad_token == tok.eos_token
    assert json.loads(cfg_path.read_text()) == original
    assert calls[1].get("use_fast") is False


def test_load_tokenizer_remote_fast_online_parse_falls_back_to_slow(monkeypatch):
    """Regression: OSError -> online-fast parse failure -> online-slow."""
    monkeypatch.setattr(
        loader,
        "resolve_model_source",
        lambda _: {"local_path": None, "hf_name": "dummy/model", "trust_remote_code": False},
    )

    calls: list[dict] = []

    def _fake_from_pretrained(path, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # local-fast: not cached
            raise OSError("not in local cache")
        if len(calls) == 2:
            # online-fast: parse failure (not OSError)
            raise RuntimeError("fast tokenizer JSON parse error")
        # online-slow: succeeds
        return _DummyTokenizer()

    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    tok = loader.load_tokenizer("dummy/model")

    assert tok.pad_token == tok.eos_token
    # Call 1: local-fast (OSError)
    assert calls[0]["local_files_only"] is True
    assert calls[0].get("use_fast") is None
    # Call 2: online-fast (RuntimeError)
    assert calls[1]["local_files_only"] is False
    assert calls[1].get("use_fast") is None
    # Call 3: online-slow (success)
    assert calls[2]["local_files_only"] is False
    assert calls[2]["use_fast"] is False


def test_load_tokenizer_remote_slow_retry_allows_online(monkeypatch):
    monkeypatch.setattr(
        loader,
        "resolve_model_source",
        lambda _: {"local_path": None, "hf_name": "dummy/model", "trust_remote_code": False},
    )

    calls: list[dict] = []

    def _fake_from_pretrained(path, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Exception("fast parse failed")
        if len(calls) == 2:
            raise OSError("not in local cache")
        return _DummyTokenizer()

    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    tok = loader.load_tokenizer("dummy/model")

    assert tok.pad_token == tok.eos_token
    assert calls[0]["local_files_only"] is True
    assert calls[1]["local_files_only"] is True
    assert calls[1]["use_fast"] is False
    assert calls[2]["local_files_only"] is False
    assert calls[2]["use_fast"] is False
