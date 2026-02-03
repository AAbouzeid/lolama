#!/usr/bin/env python3
"""
Text Generation
===============
Generate text using a loaded model.

Usage:
    python scripts/generate.py "The meaning of life is"
    python scripts/generate.py --chat                     # Interactive mode (Ctrl+C to exit)
    python scripts/generate.py --chat --no-stream         # Interactive without streaming
    python scripts/generate.py --stream "Tell me a joke"
    python scripts/generate.py --quantize "Hello"        
    python scripts/generate.py --quantize --fast "Hello" 
    python scripts/generate.py --model weights/tinyllama-1.1b "Hello"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer, resolve_model_source
from src.model import (
    quantize_model_int8,
    dequantize_model_for_inference,
    get_model_size_mb,
    save_quantized_model,
    load_quantized_model,
    is_quantized_model_dir,
    Llama,
)


def get_quantized_dir(model_path: str) -> Path:
    """Get the directory where quantized model would be saved."""
    path = Path(model_path)
    if path.exists():
        name = path.name
    else:
        name = model_path.replace("/", "_").replace("\\", "_")
    
    return WEIGHTS_DIR / f"{name}-int8"


def get_source_dir(model_path: str) -> Path | None:
    """Get the source directory for copying tokenizer files."""
    source = resolve_model_source(model_path)
    if source["local_path"] is not None:
        return Path(source["local_path"])
    return None


def generate_response(
    model: Llama,
    tokenizer,
    prompt: str,
    device: str,
    stream: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> None:
    """Generate a response for a single prompt."""
    # Prepare input
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    if stream:
        # Streaming generation
        generated_tokens: list[int] = []
        prev_text: str = ""
        
        for token_id in model.generate_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        ):
            generated_tokens.append(token_id)
            full_text: str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            new_text: str = full_text[len(prev_text):]
            print(new_text, end="", flush=True)
            prev_text = full_text
        
        print()
    else:
        # Batch generation
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_ids = output_ids[0, input_ids.shape[1]:]
        output_text: str = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(output_text)


def interactive_chat(model: Llama, tokenizer, device: str, stream: bool = True) -> None:
    """Interactive chat loop. Ctrl+C to exit."""
    print()
    print("=" * 50)
    print("Interactive Chat Mode")
    print("Type your message and press Enter.")
    print("Ctrl+C to exit.")
    print("=" * 50)
    print()
    
    try:
        while True:
            try:
                prompt: str = input("You: ").strip()
                if not prompt:
                    continue
                
                print()
                print("Assistant: ", end="", flush=True)
                generate_response(model, tokenizer, prompt, device, stream=stream)
                print()
                
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\n\nExiting chat...")


def main() -> None:
    # Parse args
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt: str | None = None
    stream: bool = False
    quantize: bool = False
    fast_mode: bool = False
    chat_mode: bool = False
    
    args: list[str] = sys.argv[1:]
    
    if "--chat" in args:
        chat_mode = True
        stream = True  # Default to streaming in chat mode
        args.remove("--chat")
    
    if "--stream" in args:
        stream = True
        args.remove("--stream")
    
    if "--no-stream" in args:
        stream = False
        args.remove("--no-stream")
    
    if "--quantize" in args:
        quantize = True
        args.remove("--quantize")
    
    if "--fast" in args:
        fast_mode = True
        args.remove("--fast")
    
    if "--model" in args:
        idx: int = args.index("--model")
        model_path = args[idx + 1]
        args = args[:idx] + args[idx+2:]
    
    if args:
        prompt = " ".join(args)
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load model (with optional quantization)
    quantized_dir: Path = get_quantized_dir(model_path)
    tokenizer_source: str = model_path
    
    if quantize and is_quantized_model_dir(str(quantized_dir)):
        print(f"Found saved quantized model: {quantized_dir}/")
        model = load_model(model_path, device="cpu")
        quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])
        load_quantized_model(str(quantized_dir), model)
        model = model.to(device)
        
        if fast_mode:
            dequantize_model_for_inference(model)
        
        tokenizer_source = str(quantized_dir)
        print(f"Model size: {get_model_size_mb(model):.1f} MB")
    elif quantize:
        model = load_model(model_path, device=device)
        size_before: float = get_model_size_mb(model)
        print(f"Quantizing model (before: {size_before:.1f} MB)...")
        quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])
        
        source_dir: Path | None = get_source_dir(model_path)
        print(f"\nSaving quantized model to {quantized_dir}/")
        save_quantized_model(model, str(quantized_dir), str(source_dir) if source_dir else None)
        
        if fast_mode:
            dequantize_model_for_inference(model)
        
        print(f"\nModel size: {get_model_size_mb(model):.1f} MB")
        print()
    else:
        model = load_model(model_path, device=device)
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_source)
    
    # Interactive chat mode
    if chat_mode:
        interactive_chat(model, tokenizer, device, stream=stream)
        return
    
    # Single prompt mode
    if prompt is None:
        prompt = "The meaning of life is"
    
    print()
    print(f"Prompt: \"{prompt}\"")
    print("-" * 50)
    
    if stream:
        print("Output: ", end="", flush=True)
    else:
        print("Output: ", end="")
    
    generate_response(model, tokenizer, prompt, device, stream=stream)


if __name__ == "__main__":
    main()
