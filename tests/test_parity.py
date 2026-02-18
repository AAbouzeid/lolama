"""Logits and generation parity tests against HuggingFace Transformers.

Creates a tiny LLaMA model in both HF and lolama, copies weights exactly,
and verifies that forward-pass logits and greedy generation match.
Runs on CPU in ~2 seconds — no GPU or model downloads required.
"""

import pytest
import torch
from transformers import LlamaConfig as HFLlamaConfig, LlamaForCausalLM

from lolama.model.config import LlamaConfig
from lolama.model.llama import Llama
from lolama.model.generator import TextGenerator
from lolama.model.generation_config import GenerationConfig
from lolama.model.quantize import quantize_model_int8

# Tiny model dims — fast on CPU, identical structure to real LLaMA
_VOCAB = 256
_D_MODEL = 64
_NUM_HEADS = 2
_NUM_KV_HEADS = 2
_NUM_LAYERS = 2
_HIDDEN_DIM = 172
_MAX_SEQ_LEN = 64
_EPS = 1e-6

# Weight mapping: HF key -> lolama key (single source of truth)
from lolama.data.weight_mappings import build_llm_weight_mapping_with_lm_head
_WEIGHT_MAP: dict[str, str] = build_llm_weight_mapping_with_lm_head(_NUM_LAYERS)


@pytest.fixture
def hf_model():
    """Tiny HF LlamaForCausalLM with deterministic init."""
    config = HFLlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=_D_MODEL,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        num_hidden_layers=_NUM_LAYERS,
        intermediate_size=_HIDDEN_DIM,
        max_position_embeddings=_MAX_SEQ_LEN,
        rms_norm_eps=_EPS,
        tie_word_embeddings=False,
        rope_theta=10000,
    )
    torch.manual_seed(42)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def our_model(hf_model):
    """lolama Llama with weights copied from the HF fixture."""
    config = LlamaConfig(
        vocab_size=_VOCAB,
        d_model=_D_MODEL,
        num_heads=_NUM_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        num_layers=_NUM_LAYERS,
        hidden_dim=_HIDDEN_DIM,
        max_seq_len=_MAX_SEQ_LEN,
        eps=_EPS,
    )
    model = Llama(config, init_weights=False)

    hf_state = hf_model.state_dict()
    new_state = {our_key: hf_state[hf_key] for hf_key, our_key in _WEIGHT_MAP.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


# ── Logits parity ────────────────────────────────────────────────────

class TestLogitsParity:
    """Forward-pass logits must match HF Transformers (fp32, CPU)."""

    ATOL = 1e-4
    RTOL = 1e-4

    def test_single_token(self, hf_model, our_model):
        ids = torch.tensor([[42]])
        with torch.no_grad():
            hf_out = hf_model(ids).logits
            our_out = our_model(ids)

        assert hf_out.shape == our_out.shape
        diff = (hf_out - our_out).abs().max().item()
        assert torch.allclose(hf_out, our_out, atol=self.ATOL, rtol=self.RTOL), \
            f"Max abs diff: {diff:.6e}"

    def test_short_sequence(self, hf_model, our_model):
        ids = torch.tensor([[10, 20, 30, 40, 50]])
        with torch.no_grad():
            hf_out = hf_model(ids).logits
            our_out = our_model(ids)

        diff = (hf_out - our_out).abs().max().item()
        assert torch.allclose(hf_out, our_out, atol=self.ATOL, rtol=self.RTOL), \
            f"Max abs diff: {diff:.6e}"

    def test_batch(self, hf_model, our_model):
        ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
        with torch.no_grad():
            hf_out = hf_model(ids).logits
            our_out = our_model(ids)

        diff = (hf_out - our_out).abs().max().item()
        assert torch.allclose(hf_out, our_out, atol=self.ATOL, rtol=self.RTOL), \
            f"Max abs diff: {diff:.6e}"

    def test_argmax_identical(self, hf_model, our_model):
        """Top-1 predicted token must be identical at every position."""
        ids = torch.tensor([[5, 15, 25, 35, 45, 55]])
        with torch.no_grad():
            hf_top1 = hf_model(ids).logits.argmax(dim=-1)
            our_top1 = our_model(ids).argmax(dim=-1)

        assert torch.equal(hf_top1, our_top1)


# ── Generation parity ────────────────────────────────────────────────

class TestGenerationParity:
    """Greedy generation through our KV-cached generator must match
    step-by-step greedy decoding with HF (no cache, recompute each step).
    Any divergence reveals a KV-cache or position-encoding bug.
    """

    MAX_NEW = 10

    def _hf_greedy(self, hf_model, input_ids, n_tokens):
        """Greedy decode step-by-step with HF (no cache, deterministic)."""
        ids = input_ids.clone()
        tokens = []
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = hf_model(ids).logits[:, -1, :]
                next_tok = logits.argmax(dim=-1, keepdim=True)
                tokens.append(next_tok.item())
                ids = torch.cat([ids, next_tok], dim=1)
        return tokens

    def test_greedy_matches_hf(self, hf_model, our_model):
        ids = torch.tensor([[10, 20, 30, 40, 50]])
        hf_tokens = self._hf_greedy(hf_model, ids, self.MAX_NEW)

        gen = TextGenerator(our_model)
        config = GenerationConfig.greedy(max_new_tokens=self.MAX_NEW)
        our_output = gen.generate(ids, config)
        our_tokens = our_output[0, ids.shape[1]:].tolist()

        assert hf_tokens == our_tokens, (
            f"Greedy generation diverged.\n"
            f"  HF:   {hf_tokens}\n"
            f"  Ours: {our_tokens}"
        )

    def test_greedy_second_prompt(self, hf_model, our_model):
        """Parity on a different prompt to avoid lucky one-offs."""
        ids = torch.tensor([[100, 200, 150]])
        hf_tokens = self._hf_greedy(hf_model, ids, self.MAX_NEW)

        gen = TextGenerator(our_model)
        config = GenerationConfig.greedy(max_new_tokens=self.MAX_NEW)
        our_output = gen.generate(ids, config)
        our_tokens = our_output[0, ids.shape[1]:].tolist()

        assert hf_tokens == our_tokens, (
            f"Greedy generation diverged.\n"
            f"  HF:   {hf_tokens}\n"
            f"  Ours: {our_tokens}"
        )


# ── Quantized parity ────────────────────────────────────────────────

class TestQuantizedParity:
    """Quantized lolama models should maintain reasonable parity with
    the unquantized HF reference. Int8 quantization introduces small errors,
    so we use relaxed tolerances and check argmax agreement rate.
    """

    @pytest.fixture
    def quantized_model(self, our_model):
        """lolama model after int8 quantization (no outlier detection)."""
        quantize_model_int8(our_model, skip_layers=["lm_head", "embed_tokens"])
        our_model.eval()
        return our_model

    @pytest.fixture
    def quantized_model_outlier(self, hf_model):
        """lolama model after int8 quantization with outlier detection."""
        config = LlamaConfig(
            vocab_size=_VOCAB, d_model=_D_MODEL, num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS, num_layers=_NUM_LAYERS,
            hidden_dim=_HIDDEN_DIM, max_seq_len=_MAX_SEQ_LEN, eps=_EPS,
        )
        model = Llama(config, init_weights=False)
        hf_state = hf_model.state_dict()
        new_state = {our_key: hf_state[hf_key] for hf_key, our_key in _WEIGHT_MAP.items()}
        model.load_state_dict(new_state, strict=False)
        quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"],
                            outlier_threshold=3.0)
        model.eval()
        return model

    def test_quantized_logits_bounded_error(self, hf_model, quantized_model):
        """Quantized logits should have bounded relative error vs HF."""
        ids = torch.tensor([[10, 20, 30, 40, 50]])
        with torch.no_grad():
            hf_out = hf_model(ids).logits
            q_out = quantized_model(ids)

        rel_error = (hf_out - q_out).abs().mean() / hf_out.abs().mean()
        assert rel_error < 0.10, f"Relative error {rel_error:.4f} exceeds 10%"

    def test_quantized_argmax_agreement(self, hf_model, quantized_model):
        """Top-1 predictions should agree on most positions."""
        ids = torch.tensor([[5, 15, 25, 35, 45, 55]])
        with torch.no_grad():
            hf_top1 = hf_model(ids).logits.argmax(dim=-1)
            q_top1 = quantized_model(ids).argmax(dim=-1)

        agreement = (hf_top1 == q_top1).float().mean().item()
        assert agreement >= 0.5, (
            f"Argmax agreement {agreement:.0%} below 50%.\n"
            f"  HF:   {hf_top1.tolist()}\n"
            f"  Ours: {q_top1.tolist()}"
        )

    def test_quantized_generation_reasonable(self, quantized_model):
        """Quantized greedy generation should produce valid (non-degenerate) output."""
        ids = torch.tensor([[10, 20, 30, 40, 50]])
        gen = TextGenerator(quantized_model)
        config = GenerationConfig.greedy(max_new_tokens=10)
        output = gen.generate(ids, config)
        generated = output[0, ids.shape[1]:].tolist()

        # Should generate exactly max_new_tokens (or stop at EOS)
        assert len(generated) > 0, "No tokens generated"
        # Should not degenerate into repeating a single token
        assert len(set(generated)) > 1 or len(generated) <= 2, (
            f"Degenerate output: {generated}"
        )

    def test_outlier_quantized_logits_bounded(self, hf_model, quantized_model_outlier):
        """Outlier-aware quantized logits should have bounded error vs HF."""
        ids = torch.tensor([[10, 20, 30, 40, 50]])
        with torch.no_grad():
            hf_out = hf_model(ids).logits
            q_out = quantized_model_outlier(ids)

        rel_error = (hf_out - q_out).abs().mean() / hf_out.abs().mean()
        assert rel_error < 0.10, f"Outlier-aware relative error {rel_error:.4f} exceeds 10%"
