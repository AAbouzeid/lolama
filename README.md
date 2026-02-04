# lolama

**A from-scratch PyTorch implementation of the LLaMA architecture with Hugging Face weight compatibility.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **Pure PyTorch Implementation** — Complete LLaMA transformer built from scratch, no hidden abstractions
- **Hugging Face Compatible** — Load pre-trained weights directly from the HF Hub
- **Efficient Inference** — KV caching and streaming generation for fast token output
- **Int8 Quantization** — ~4x memory reduction with weight-only quantization
- **Multi-Device Support** — CUDA, Apple Silicon (MPS), and CPU backends
- **Unified CLI** — Download, generate, and quantize models from the command line

---

## Quick Start

### Installation

```bash
git clone https://github.com/AAbouzeid/lolama.git
cd lolama
pip install -e .
```

### Generate Text

```bash
# Generate with TinyLlama (default, ~1.1B params)
lolama generate "The key to understanding transformers is"

# Use a specific model by alias
lolama generate -m open_llama_3b "The key to understanding transformers is"

# Stream output in real-time
lolama generate --stream "Once upon a time"

# Interactive chat mode
lolama chat
```

### Python API

```python
from lolama import load_model, load_tokenizer, TextGenerator

model = load_model("tinyllama")          # alias works here too
tokenizer = load_tokenizer("tinyllama")
generator = TextGenerator(model, tokenizer)

for token in generator.generate_stream("The meaning of life is", max_new_tokens=100):
    print(token, end="", flush=True)
```

---

## Supported Models

| Alias | Params | Description | Hugging Face ID |
|-------|--------|-------------|-----------------|
| `tinyllama` | 1.1B | TinyLlama 1.1B Chat — small, fast, instruction-tuned | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `open_llama_3b` | 3B | OpenLLaMA 3B v2 — compact base model | `openlm-research/open_llama_3b_v2` |
| `open_llama_7b` | 7B | OpenLLaMA 7B v2 — full-size base model | `openlm-research/open_llama_7b_v2` |
| `llama7b` | 7B | LLaMA 2 7B — Meta's base model (gated, requires access) | `meta-llama/Llama-2-7b-hf` |

List all models and aliases:

```bash
lolama models
```

Download for offline use:

```bash
lolama download tinyllama    # Downloads to ./weights/
```

---

## CLI Reference

### `lolama models`

List available model aliases.

```bash
lolama models
```

### `lolama generate`

Generate text from a prompt.

```bash
lolama generate [OPTIONS] [PROMPT]

Options:
  -m, --model TEXT        Model alias, HF name, or local path (default: tinyllama)
  --max-tokens INT        Maximum tokens to generate (default: 256)
  --temperature FLOAT     Sampling temperature (default: 0.7)
  --top-p FLOAT           Nucleus sampling threshold (default: 0.9)
  --stream                Stream tokens as they're generated
  --quantize              Use int8 quantization
  --fast                  Pre-dequantize weights for faster inference
```

**Examples:**

```bash
# Use a specific model
lolama generate -m open_llama_3b "Write a haiku about recursion"

# Deterministic output with greedy decoding
lolama generate --temperature 0.0 "The capital of France is"

# Memory-efficient generation with quantization
lolama generate --quantize "Explain quantum computing"
```

### `lolama chat`

Interactive chat session (REPL).

```bash
lolama chat [OPTIONS]

Options:
  -m, --model TEXT        Model alias, HF name, or local path (default: tinyllama)
  --max-tokens INT        Maximum tokens per response (default: 256)
  --temperature FLOAT     Sampling temperature (default: 0.7)
  --top-p FLOAT           Nucleus sampling threshold (default: 0.9)
  --stream                Stream tokens as they're generated
  --quantize              Use int8 quantization
  --fast                  Pre-dequantize weights for faster inference
```

### `lolama download`

Download and cache models locally.

```bash
lolama download MODEL_ALIAS [OPTIONS]

Options:
  --from-cache    Use local HF cache only (no download)
```

### `lolama quantize`

Test or apply int8 quantization.

```bash
lolama quantize [OPTIONS]

Options:
  -m, --model TEXT    Model alias, HF name, or local path (default: tinyllama)
  --save              Save quantized weights to disk
```

---

## Architecture

lolama implements the full LLaMA architecture:

```
Llama
├── embed_tokens          # Token embeddings
├── layers (×N)           # Transformer blocks
│   ├── attention         # Multi-head attention with RoPE
│   │   ├── q_proj        # Query projection
│   │   ├── k_proj        # Key projection (GQA supported)
│   │   ├── v_proj        # Value projection (GQA supported)
│   │   └── o_proj        # Output projection
│   ├── attention_norm    # RMSNorm (pre-attention)
│   ├── feed_forward      # SwiGLU FFN
│   │   ├── gate_proj     # Gating projection
│   │   ├── up_proj       # Up projection
│   │   └── down_proj     # Down projection
│   └── ffn_norm          # RMSNorm (pre-FFN)
├── norm                  # Final RMSNorm
└── lm_head               # Output projection to vocabulary
```

**Key Implementation Details:**

- **RoPE (Rotary Position Embeddings)** — Precomputed sin/cos buffers shared across layers
- **Grouped Query Attention** — Configurable KV heads for memory-efficient attention
- **SwiGLU Activation** — Gated linear unit with SiLU for improved training dynamics
- **RMSNorm** — Computed in float32 for numerical stability
- **KV Cache** — Pre-allocated cache eliminates concatenation overhead during generation

---

## Quantization

Reduce memory usage with int8 weight-only quantization:

```python
from lolama import load_model
from lolama.model import quantize_model_int8, get_model_size_mb

model = load_model("tinyllama")
print(f"Original: {get_model_size_mb(model):.0f} MB")

quantize_model_int8(model)
print(f"Quantized: {get_model_size_mb(model):.0f} MB")  # ~4x smaller
```

**How it works:**
- Per-channel quantization of weight matrices
- Scales stored as float16 for efficient dequantization
- Embedding and LM head layers preserved at full precision
- On-the-fly dequantization during forward pass

---

## Configuration

Create custom model configurations:

```python
from lolama import Llama, LlamaConfig

config = LlamaConfig(
    vocab_size=32000,
    d_model=2048,
    num_heads=32,
    num_kv_heads=8,      # GQA: fewer KV heads than query heads
    num_layers=22,
    hidden_dim=5632,     # FFN hidden dimension
    max_seq_len=4096,
    rope_base=10000,     # RoPE frequency base
)

model = Llama(config)
```

---

## Project Structure

```
lolama/
├── lolama/
│   ├── model/           # Core model components
│   │   ├── llama.py     # Main Llama class
│   │   ├── layers.py    # Attention, FFN, RMSNorm
│   │   ├── generator.py # Text generation
│   │   ├── kv_cache.py  # KV cache implementation
│   │   └── quantize.py  # Int8 quantization
│   ├── data/            # Model loading & registry
│   ├── utils/           # RoPE, device detection, logging
│   └── cli.py           # Command-line interface
├── tests/               # Comprehensive test suite
└── weights/             # Local model storage
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+ (for tokenizers and weight loading)
- CUDA 11.7+ (optional, for GPU acceleration)

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy lolama/
```

---

## Acknowledgments

- [Meta AI](https://github.com/meta-llama/llama) for the original LLaMA architecture
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [Andrej Karpathy](https://github.com/karpathy/llama2.c) for educational LLaMA implementations

---

## License

MIT License. See [LICENSE](LICENSE) for details.
