"""Unified CLI for lolama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _load_generator(args: argparse.Namespace):
    """Shared model/tokenizer setup for generate and chat commands."""
    from .data import load_model, load_tokenizer, resolve_model_source
    from .model import (
        TextGenerator,
        quantize_model_int8,
        dequantize_model_for_inference,
        get_model_size_mb,
        save_quantized_model,
        load_quantized_model,
        is_quantized_model_dir,
    )
    from .utils import resolve_device

    device = resolve_device()
    print(f"Device: {device}")
    print()

    weights_dir = Path(__file__).parent.parent / "weights"
    model_path = args.model

    def get_quantized_dir(mp: str) -> Path:
        p = Path(mp)
        name = p.name if p.exists() else mp.replace("/", "_").replace("\\", "_")
        return weights_dir / f"{name}-int8"

    quantized_dir = get_quantized_dir(model_path)

    if args.quantize and is_quantized_model_dir(str(quantized_dir)):
        print(f"Found saved quantized model: {quantized_dir}/")
        model = load_model(model_path, device="cpu")
        quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"])
        load_quantized_model(str(quantized_dir), model)
        model = model.to(device)
        if args.fast:
            dequantize_model_for_inference(model)
        print(f"Model size: {get_model_size_mb(model):.1f} MB")
    elif args.quantize:
        model = load_model(model_path, device=device)
        size_before = get_model_size_mb(model)
        print(f"Quantizing model (before: {size_before:.1f} MB)...")
        quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"])
        source = resolve_model_source(model_path)
        source_dir = source["local_path"]
        print(f"\nSaving quantized model to {quantized_dir}/")
        save_quantized_model(model, str(quantized_dir), str(source_dir) if source_dir else None)
        if args.fast:
            dequantize_model_for_inference(model)
        print(f"\nModel size: {get_model_size_mb(model):.1f} MB")
        print()
    else:
        model = load_model(model_path, device=device)

    tokenizer = load_tokenizer(model_path)
    generator = TextGenerator(model)

    def tokenize_prompt(prompt: str) -> torch.Tensor:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
            return input_ids.to(device)
        return tokenizer.encode(prompt, return_tensors="pt").to(device)

    def respond(prompt: str, stream: bool) -> None:
        input_ids = tokenize_prompt(prompt)
        if stream:
            generated_tokens: list[int] = []
            prev_text = ""
            for token_id in generator.generate_stream(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            ):
                generated_tokens.append(token_id)
                full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(full_text[len(prev_text):], end="", flush=True)
                prev_text = full_text
            print()
        else:
            with torch.no_grad():
                output_ids = generator.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, input_ids.shape[1]:]
            print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return respond


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate text from a prompt."""
    respond = _load_generator(args)
    prompt = args.prompt or "The meaning of life is"
    print(f'Prompt: "{prompt}"')
    print("-" * 50)
    print("Output: ", end="", flush=True)
    respond(prompt, stream=args.stream)


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat session."""
    respond = _load_generator(args)
    print("=" * 50)
    print("Interactive Chat Mode (Ctrl+C to exit)")
    print("=" * 50)
    print()
    try:
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            print("\nAssistant: ", end="", flush=True)
            respond(prompt, stream=args.stream)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting chat...")


def cmd_models(args: argparse.Namespace) -> None:
    """List available model aliases."""
    from .data.registry import MODEL_REGISTRY

    print()
    print(f"  {'Alias':<16} {'Params':<8} {'Description':<50} {'HuggingFace ID'}")
    print(f"  {'─' * 16} {'─' * 8} {'─' * 50} {'─' * 40}")
    for alias, info in MODEL_REGISTRY.items():
        print(
            f"  {alias:<16} {info['params']:<8} {info['description']:<50} {info['hf_name']}"
        )
    print()
    print("Usage:  lolama generate \"hello\" -m tinyllama --stream")
    print("        lolama chat -m tinyllama")
    print("        lolama download tinyllama")
    print()


def cmd_download(args: argparse.Namespace) -> None:
    """Download a model from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data.registry import MODEL_REGISTRY

    weights_dir = Path(__file__).parent.parent / "weights"

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    info = MODEL_REGISTRY[args.model]
    hf_name = info["hf_name"]
    save_dir = weights_dir / info["folder"]
    trust = info["trust_remote_code"]

    print(f"Model: {hf_name}")
    print(f"Save to: {save_dir}")
    print()

    if save_dir.exists() and any(save_dir.iterdir()):
        print("Model already saved locally.")
        print(f"Location: {save_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust,
            local_files_only=True,
        )
    except OSError:
        if args.from_cache:
            print("Not found in cache and --from-cache was set.")
            sys.exit(1)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust,
            local_files_only=False,
        )

    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust, local_files_only=True)
    except OSError:
        if args.from_cache:
            print("Tokenizer not found in cache and --from-cache was set.")
            sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust, local_files_only=False)

    print(f"Saving to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 2 / 1024**2
    print()
    print("Saved!")
    print(f"Parameters: {total_params:,}")
    print(f"Size: {size_mb:.1f} MB (fp16)")


def cmd_quantize(args: argparse.Namespace) -> None:
    """Quantize a model to int8."""
    from .data import load_model, load_tokenizer, resolve_model_source
    from .model import (
        TextGenerator,
        quantize_model_int8,
        get_model_size_mb,
        save_quantized_model,
    )
    from .utils import resolve_device

    device = resolve_device()
    print(f"Device: {device}")
    print()

    weights_dir = Path(__file__).parent.parent / "weights"
    model_path = args.model

    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer(model_path)

    size_before = get_model_size_mb(model)
    print(f"\nModel size before quantization: {size_before:.1f} MB")

    # Test before
    prompt = "What is 2+2?"
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f'\nPrompt: "{prompt}"')
    print("\n--- Before Quantization (greedy) ---")
    generator = TextGenerator(model)
    with torch.no_grad():
        output_ids = generator.generate(input_ids, max_new_tokens=30, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    print(f"Output: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")

    # Quantize
    print("\n" + "=" * 60)
    print("Quantizing model to int8...")
    print("=" * 60)
    quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"])

    size_after = get_model_size_mb(model)
    print(f"\nModel size after quantization: {size_after:.1f} MB")
    print(f"Compression ratio: {size_before / size_after:.2f}x")

    # Test after
    print("\n--- After Quantization (greedy) ---")
    generator = TextGenerator(model)
    with torch.no_grad():
        output_ids = generator.generate(input_ids, max_new_tokens=30, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    print(f"Output: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")

    # Save if requested
    if args.save:
        p = Path(model_path)
        name = p.name if p.exists() else model_path.replace("/", "_").replace("\\", "_")
        output_dir = weights_dir / f"{name}-int8"
        source = resolve_model_source(model_path)
        source_dir = source["local_path"]
        print(f"\nSaving quantized model to {output_dir}/")
        save_quantized_model(model, str(output_dir), str(source_dir) if source_dir else None)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lolama",
        description="LLaMA from scratch in PyTorch",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shared sampling options for generate and chat
    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-m", "--model", default="tinyllama",
                       help="Model alias, HF name, or local path (default: tinyllama)")
        p.add_argument("--stream", action="store_true", help="Stream output tokens")
        p.add_argument("--quantize", action="store_true", help="Use int8 quantization")
        p.add_argument("--fast", action="store_true", help="Pre-dequantize for faster inference")
        p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
        p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
        p.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", nargs="?", help="The prompt to generate from")
    add_model_args(gen_parser)
    gen_parser.set_defaults(func=cmd_generate)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    add_model_args(chat_parser)
    chat_parser.set_defaults(func=cmd_chat, stream=True)

    # models
    models_parser = subparsers.add_parser("models", help="List available model aliases")
    models_parser.set_defaults(func=cmd_models)

    # download
    dl_parser = subparsers.add_parser("download", help="Download a model from HuggingFace")
    dl_parser.add_argument("model", help="Model alias (see 'lolama models' for list)")
    dl_parser.add_argument("--from-cache", action="store_true", help="Only use local HF cache")
    dl_parser.set_defaults(func=cmd_download)

    # quantize
    q_parser = subparsers.add_parser("quantize", help="Test int8 quantization on a model")
    q_parser.add_argument("-m", "--model", default="tinyllama",
                          help="Model alias, HF name, or local path (default: tinyllama)")
    q_parser.add_argument("--save", action="store_true", help="Save quantized model to weights/")
    q_parser.set_defaults(func=cmd_quantize)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
