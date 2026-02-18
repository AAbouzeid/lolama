"""Quick validation: tests + normal/quantized inference."""
import subprocess, sys, torch

def run_tests():
    print("=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True, text=True,
    )
    # Print only summary
    lines = r.stdout.strip().split("\n")
    for line in lines:
        if "passed" in line or "failed" in line or "error" in line or "FAILED" in line:
            print(line)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:] if r.stderr else "")
        print("TEST SUITE FAILED")
        sys.exit(1)
    print()

def run_model():
    print("=" * 60)
    print("RUNNING MODEL VALIDATION")
    print("=" * 60)
    from lolama.data import load_model, load_tokenizer, create_model
    from lolama.model import (
        TextGenerator, GenerationConfig, apply_quantization_structure,
        quantize_model_int8, save_quantized_model, load_quantized_model,
        get_model_size_mb,
    )
    import tempfile

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prompt = "Who is the current US president"

    # --- Normal inference ---
    print(f"\n[1/2] Normal inference (device={device})")
    model = load_model("tinyllama", device=device)
    tokenizer = load_tokenizer("tinyllama")
    gen = TextGenerator(model)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    config = GenerationConfig(max_new_tokens=40, temperature=0.7, top_p=0.9,
                              eos_token_id=tokenizer.eos_token_id)
    out = gen.generate(ids, config)
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {text}")
    assert len(text.strip()) > 0, "Empty output from normal model"
    print("  NORMAL INFERENCE OK")
    del model, gen
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Quantized inference (save + reload) ---
    print(f"\n[2/2] Quantized inference (device={device})")
    model_q = load_model("tinyllama", device="cpu")
    quantize_model_int8(model_q, skip_layers=["lm_head", "embed_tokens"])
    with tempfile.TemporaryDirectory() as tmpdir:
        save_quantized_model(model_q, tmpdir)
        del model_q

        model_r = create_model("tinyllama")
        apply_quantization_structure(model_r, skip_layers=["lm_head", "embed_tokens"])
        load_quantized_model(tmpdir, model_r, device=device)

    gen_r = TextGenerator(model_r)
    out_q = gen_r.generate(ids, config)
    text_q = tokenizer.decode(out_q[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {text_q}")
    assert len(text_q.strip()) > 0, "Empty output from quantized model"
    print("  QUANTIZED INFERENCE OK")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
    run_model()
