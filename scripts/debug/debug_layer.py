#!/usr/bin/env python3
"""Debug: Compare layer by layer."""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import Llama, LlamaConfig
from src.data import create_config_from_hf, load_weights_from_hf

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cpu"  # Use CPU for easier debugging

# Load tokenizer and create input
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
input_ids = tokenizer.encode("Hello", return_tensors="pt")
print(f"Input: {input_ids}")

# Load HF model
print("\nLoading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use fp32 for comparison
    low_cpu_mem_usage=True
).to(device)
hf_model.eval()

# Load our model
print("Loading our model...")
config = create_config_from_hf(MODEL_NAME, local_files_only=True)
our_model = Llama(config).to(device)
our_model = load_weights_from_hf(our_model, MODEL_NAME, local_files_only=True)
our_model.eval()

# Convert our model to fp32 explicitly
our_model = our_model.float()

print("\n" + "=" * 60)
print("Step-by-step comparison")
print("=" * 60)

with torch.no_grad():
    # Step 0: Verify embedding weights match
    hf_emb_weight = hf_model.model.embed_tokens.weight
    our_emb_weight = our_model.embed_tokens.weight
    
    print(f"\n0. Embedding WEIGHTS:")
    print(f"   HF shape:  {hf_emb_weight.shape}, dtype: {hf_emb_weight.dtype}")
    print(f"   Our shape: {our_emb_weight.shape}, dtype: {our_emb_weight.dtype}")
    print(f"   HF[0, :5]:  {hf_emb_weight[0, :5].tolist()}")
    print(f"   Our[0, :5]: {our_emb_weight[0, :5].tolist()}")
    print(f"   Weights match: {torch.allclose(hf_emb_weight.float(), our_emb_weight.float(), atol=1e-4)}")
    print(f"   Max weight diff: {(hf_emb_weight.float() - our_emb_weight.float()).abs().max().item()}")
    
    # Check specific token embedding
    token_id = input_ids[0, 0].item()
    print(f"\n   Token ID {token_id} embedding:")
    print(f"   HF:  {hf_emb_weight[token_id, :5].tolist()}")
    print(f"   Our: {our_emb_weight[token_id, :5].tolist()}")
    
    # Step 1: Embedding
    hf_emb = hf_model.model.embed_tokens(input_ids)
    our_emb = our_model.embed_tokens(input_ids)
    
    print(f"\n1. Embedding OUTPUT:")
    print(f"   HF:  {hf_emb[0, 0, :5].tolist()}")
    print(f"   Our: {our_emb[0, 0, :5].tolist()}")
    print(f"   Match: {torch.allclose(hf_emb, our_emb, atol=1e-4)}")
    print(f"   Max diff: {(hf_emb - our_emb).abs().max().item()}")
    
    # Step 2: First layer norm
    hf_normed = hf_model.model.layers[0].input_layernorm(hf_emb)
    our_normed = our_model.layers[0].attention_norm(our_emb)
    
    print(f"\n2. First LayerNorm (input to attention):")
    print(f"   HF:  {hf_normed[0, 0, :5].tolist()}")
    print(f"   Our: {our_normed[0, 0, :5].tolist()}")
    print(f"   Match: {torch.allclose(hf_normed, our_normed, atol=1e-4)}")
    print(f"   Max diff: {(hf_normed - our_normed).abs().max().item()}")
    
    # Step 3: Q, K, V projections
    hf_q = hf_model.model.layers[0].self_attn.q_proj(hf_normed)
    hf_k = hf_model.model.layers[0].self_attn.k_proj(hf_normed)
    hf_v = hf_model.model.layers[0].self_attn.v_proj(hf_normed)
    
    our_q = our_model.layers[0].attention.q_proj(our_normed)
    our_k = our_model.layers[0].attention.k_proj(our_normed)
    our_v = our_model.layers[0].attention.v_proj(our_normed)
    
    print(f"\n3. Q projection:")
    print(f"   HF:  {hf_q[0, 0, :5].tolist()}")
    print(f"   Our: {our_q[0, 0, :5].tolist()}")
    print(f"   Match: {torch.allclose(hf_q, our_q, atol=1e-4)}")
    print(f"   Max diff: {(hf_q - our_q).abs().max().item()}")
    
    print(f"\n4. K projection:")
    print(f"   HF shape:  {hf_k.shape}")
    print(f"   Our shape: {our_k.shape}")
    print(f"   HF:  {hf_k[0, 0, :5].tolist()}")
    print(f"   Our: {our_k[0, 0, :5].tolist()}")
    print(f"   Match: {torch.allclose(hf_k, our_k, atol=1e-4)}")
    
    # Step 4: After RoPE
    # This is where HF applies RoPE internally
    # We need to manually compute what happens after RoPE
    
    print(f"\n5. Checking RoPE...")
    # Get cos/sin from our model
    cos = our_model.layers[0].attention.cos
    sin = our_model.layers[0].attention.sin
    print(f"   cos shape: {cos.shape}")
    print(f"   sin shape: {sin.shape}")
    print(f"   cos[:1, :5]: {cos[:1, :5].tolist()}")
    
    # Compare with HF's rotary embedding
    hf_rotary = hf_model.model.layers[0].self_attn.rotary_emb
    hf_cos, hf_sin = hf_rotary(hf_v, position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0))
    print(f"   HF cos shape: {hf_cos.shape}")
    print(f"   HF cos[:1, :5]: {hf_cos[0, :1, :5].tolist()}")
    
    # Check if frequencies match
    print(f"\n6. Comparing cos values:")
    # Reshape our cos to match HF format
    our_cos_for_cmp = cos[:input_ids.shape[1], :]
    hf_cos_for_cmp = hf_cos[0, :, :]
    print(f"   Our cos shape: {our_cos_for_cmp.shape}")
    print(f"   HF cos shape:  {hf_cos_for_cmp.shape}")
    print(f"   Our cos[0, :5]: {our_cos_for_cmp[0, :5].tolist()}")
    print(f"   HF cos[0, :5]:  {hf_cos_for_cmp[0, :5].tolist()}")
