"""Unified CLI for lolama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _confirm_download(model_path: str) -> None:
    """Check if model needs downloading and prompt user for confirmation.

    Only registry aliases and local paths are allowed. Arbitrary HF model
    names are rejected.
    """
    from .data import resolve_model_source, download_model
    from .data.registry import MODEL_REGISTRY

    # Local path — always allowed
    if Path(model_path).exists():
        return

    key = model_path.lower()
    info = MODEL_REGISTRY.get(key)

    # Not a local path and not in registry — reject
    if not info:
        aliases = ", ".join(MODEL_REGISTRY.keys())
        print(f"Unknown model: '{model_path}'")
        print(f"Supported models: {aliases}")
        print("Run 'lolama models' for details.")
        sys.exit(1)

    # Check if already downloaded (registry folder or auto-saved folder)
    weights_dir = Path(__file__).parent.parent / "weights"
    save_dir = weights_dir / info["folder"]
    if save_dir.exists() and any(save_dir.iterdir()):
        return

    hf_name = info["hf_name"]
    auto_path = weights_dir / hf_name.replace("/", "_").replace("\\", "_")
    if auto_path.exists() and any(auto_path.iterdir()):
        return

    # Prompt for download
    print(f"Model '{key}' not found locally.")
    print(f"Download {hf_name} (~{info['download_size']})?")

    try:
        answer = input("[y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)

    if answer != "y":
        sys.exit(0)

    print()
    download_model(hf_name, save_dir, info["trust_remote_code"])
    print()


def _load_generator(args: argparse.Namespace):
    """Shared model/tokenizer setup for generate and chat commands."""
    from .data import load_model, load_tokenizer, resolve_model_source
    from .model import (
        TextGenerator,
        quantize_model_int8,
        get_model_size_mb,
        save_quantized_model,
        load_quantized_model,
        is_quantized_model_dir,
    )
    from .utils import resolve_device

    _confirm_download(args.model)

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
                repetition_penalty=args.repetition_penalty,
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
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, input_ids.shape[1]:]
            print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return respond


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate text from a prompt."""
    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: no prompt provided")
        print("Usage: lolama generate \"your prompt here\"")
        print("       echo \"your prompt\" | lolama generate")
        sys.exit(1)

    respond = _load_generator(args)
    print(f'Prompt: "{prompt}"')
    print("-" * 50)
    print("Output: ", end="", flush=True)
    respond(prompt, stream=not args.no_stream)


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
            respond(prompt, stream=not args.no_stream)
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
    print("Usage:  lolama generate \"hello\" -m tinyllama")
    print("        lolama chat -m tinyllama")
    print()



def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("lolama")
    except Exception:
        return "unknown"


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lolama",
        description="LLaMA from scratch in PyTorch",
    )
    parser.add_argument("-V", "--version", action="version", version=f"lolama {_get_version()}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shared sampling options for generate and chat
    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-m", "--model", default="tinyllama",
                       help="Model alias, HF name, or local path (default: tinyllama)")
        p.add_argument("--no-stream", action="store_true", help="Disable streaming (wait for full response)")
        p.add_argument("--quantize", action="store_true", help="Use int8 quantization")
        p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
        p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
        p.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
        p.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", nargs="?", help="The prompt to generate from")
    add_model_args(gen_parser)
    gen_parser.set_defaults(func=cmd_generate)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    add_model_args(chat_parser)
    chat_parser.set_defaults(func=cmd_chat)

    # models
    models_parser = subparsers.add_parser("models", help="List available model aliases")
    models_parser.set_defaults(func=cmd_models)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
