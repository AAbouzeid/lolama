#!/usr/bin/env python3
"""
Debug: Compare HuggingFace vs Our Model
=======================================
Check if HF model works but ours doesn't.
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import load_model, load_tokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare input
prompt = "Hello, how are you?"
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
if not isinstance(input_ids, torch.Tensor):
    input_ids = input_ids["input_ids"]
input_ids = input_ids.to(device)

print(f"Input tokens: {input_ids.shape}")
print(f"Input: {tokenizer.decode(input_ids[0])}")
print()

# ============================================================
# Test HuggingFace Model
# ============================================================
print("=" * 60)
print("HuggingFace Model")
print("=" * 60)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)
hf_model.eval()

with torch.no_grad():
    hf_output = hf_model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
print(f"HF Output: {hf_text}")
print()

# ============================================================
# Test Our Model
# ============================================================
print("=" * 60)
print("Our Model")
print("=" * 60)

our_model = load_model(MODEL_NAME, device=device)
our_model.eval()

# Check if weights match for a specific layer
hf_state = hf_model.state_dict()
our_state = our_model.state_dict()

# Compare a specific weight
hf_key = "model.layers.0.self_attn.q_proj.weight"
our_key = "layers.0.attention.q_proj.weight"

if hf_key in hf_state and our_key in our_state:
    hf_weight = hf_state[hf_key]
    our_weight = our_state[our_key]
    
    print(f"Weight comparison (layer 0 Q proj):")
    print(f"  HF shape: {hf_weight.shape}, dtype: {hf_weight.dtype}")
    print(f"  Our shape: {our_weight.shape}, dtype: {our_weight.dtype}")
    print(f"  Match: {torch.allclose(hf_weight.float(), our_weight.float(), atol=1e-4)}")
    print(f"  Max diff: {(hf_weight.float() - our_weight.float()).abs().max().item()}")
    print()

with torch.no_grad():
    our_output = our_model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
    )

our_text = tokenizer.decode(our_output[0], skip_special_tokens=True)
print(f"Our Output: {our_text}")
print()

# ============================================================
# Compare logits directly
# ============================================================
print("=" * 60)
print("Logits Comparison (first forward pass)")
print("=" * 60)

with torch.no_grad():
    hf_logits = hf_model(input_ids).logits
    our_logits = our_model(input_ids)

print(f"HF logits shape: {hf_logits.shape}")
print(f"Our logits shape: {our_logits.shape}")

# Compare last position
hf_last = hf_logits[0, -1, :10].float()
our_last = our_logits[0, -1, :10].float()

print(f"\nFirst 10 logits at last position:")
print(f"HF:  {hf_last.tolist()}")
print(f"Our: {our_last.tolist()}")

diff = (hf_logits.float() - our_logits.float()).abs()
print(f"\nLogits diff - max: {diff.max().item():.4f}, mean: {diff.mean().item():.4f}")
