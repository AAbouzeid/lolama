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
    from .data import resolve_model_source, download_model, download_llava_model
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
    # Use appropriate download function based on model type
    if info.get("model_type") == "vlm":
        download_llava_model(hf_name, save_dir, info["trust_remote_code"])
    else:
        download_model(hf_name, save_dir, info["trust_remote_code"])
    print()


def _is_vlm_model(model_name: str) -> bool:
    """Check if model is a vision-language model based on registry."""
    from .data.registry import MODEL_REGISTRY

    key = model_name.lower()
    info = MODEL_REGISTRY.get(key, {})
    return info.get("model_type") == "vlm"


def _load_generator(args: argparse.Namespace):
    """Shared model/tokenizer setup for generate and chat commands."""
    from .data import create_model, load_model, load_tokenizer, resolve_model_source
    from .model import (
        TextGenerator,
        apply_quantization_structure,
        quantize_model_int8,
        get_model_size_mb,
        save_quantized_model,
        load_quantized_model,
        is_quantized_model_dir,
    )
    from .utils import resolve_device, get_logger

    logger = get_logger("lolama.cli")

    _confirm_download(args.model)

    device = resolve_device()
    logger.info(f"Device: {device}")

    weights_dir = Path(__file__).parent.parent / "weights"
    model_path = args.model

    # Check if this is a VLM model
    is_vlm = _is_vlm_model(model_path)
    image_processor = None
    pixel_values = None

    if is_vlm:
        from .data.vlm_loader import load_llava_model
        from .vision import CLIPImageProcessor

        logger.info("Loading VLM model...")

        if args.quantize:
            # Check for saved quantized LLM weights
            p = Path(model_path)
            vlm_name = p.name if p.exists() else model_path.replace("/", "_").replace("\\", "_")
            vlm_quantized_dir = weights_dir / f"{vlm_name}-vlm-int8"

            if is_quantized_model_dir(str(vlm_quantized_dir)):
                # Fast path: load only vision/projector weights (LLM stays on meta),
                # apply quantization structure, load int8 weights direct to device
                logger.info(f"Found saved quantized VLM: {vlm_quantized_dir}/")
                model = load_llava_model(model_path, device="cpu", skip_llm_weights=True)
                apply_quantization_structure(model.language_model, skip_layers=["lm_head", "embed_tokens"])
                logger.info("Loading saved int8 weights...")
                load_quantized_model(str(vlm_quantized_dir), model.language_model, device=device)
                logger.info(f"Moving vision model to {device}...")
                model.vision_tower.to(device)
                model.multi_modal_projector.to(device)
                logger.info(f"Model size: {get_model_size_mb(model):.1f} MB")
            else:
                # First time: load on CPU, quantize, save, move to device
                model = load_llava_model(model_path, device="cpu")
                size_before = get_model_size_mb(model)
                logger.info(f"Quantizing VLM language model (before: {size_before:.1f} MB)...")
                quantize_model_int8(
                    model.language_model,
                    skip_layers=["lm_head", "embed_tokens"],
                )
                source = resolve_model_source(model_path)
                source_dir = source["local_path"]
                logger.info(f"Saving quantized LLM to {vlm_quantized_dir}/")
                save_quantized_model(
                    model.language_model, str(vlm_quantized_dir),
                    str(source_dir) if source_dir else None,
                )
                logger.info(f"Model size after quantization: {get_model_size_mb(model):.1f} MB")
                logger.info(f"Moving quantized model to {device}...")
                model = model.to(device)
        else:
            model = load_llava_model(model_path, device=device)

        # Create image processor
        image_processor = CLIPImageProcessor.from_config(model._vlm_config.vision_config)

        # Load image if provided
        if hasattr(args, "image") and args.image:
            from PIL import Image

            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Error: Image not found: {args.image}")
                sys.exit(1)
            logger.info(f"Loading image: {args.image}")
            image = Image.open(image_path)
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            logger.info("Preprocessing image...")
            processed = image_processor.preprocess(image)
            pixel_values = processed["pixel_values"].to(device, dtype=model.dtype)
            logger.info(f"Image preprocessed: {pixel_values.shape}, dtype: {pixel_values.dtype}")
    else:
        def get_quantized_dir(mp: str) -> Path:
            p = Path(mp)
            name = p.name if p.exists() else mp.replace("/", "_").replace("\\", "_")
            return weights_dir / f"{name}-int8"

        quantized_dir = get_quantized_dir(model_path)

        if args.quantize and is_quantized_model_dir(str(quantized_dir)):
            logger.info(f"Found saved quantized model: {quantized_dir}/")
            model = create_model(model_path)
            logger.info("Applying int8 quantization structure...")
            quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"])
            logger.info("Loading quantized weights...")
            load_quantized_model(str(quantized_dir), model)
            logger.info(f"Moving model to {device}...")
            model = model.to(device)
            logger.info(f"Model size: {get_model_size_mb(model):.1f} MB")
        elif args.quantize:
            model = load_model(model_path, device=device)
            size_before = get_model_size_mb(model)
            logger.info(f"Quantizing model (before: {size_before:.1f} MB)...")
            quantize_model_int8(model, skip_layers=["lm_head", "embed_tokens"])
            source = resolve_model_source(model_path)
            source_dir = source["local_path"]
            logger.info(f"Saving quantized model to {quantized_dir}/")
            save_quantized_model(model, str(quantized_dir), str(source_dir) if source_dir else None)
            logger.info(f"Model size after quantization: {get_model_size_mb(model):.1f} MB")
        else:
            model = load_model(model_path, device=device)

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    logger.info("Creating text generator...")
    generator = TextGenerator(model)

    def tokenize_prompt(prompt: str) -> torch.Tensor:
        # For VLMs, ensure <image> token is in the prompt
        if is_vlm and pixel_values is not None and "<image>" not in prompt:
            prompt = "<image>\n" + prompt

        if getattr(tokenizer, "chat_template", None):
            logger.debug("Applying chat template")
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
            return input_ids.to(device)

        # No chat template — apply default instruct format for VLM models
        if is_vlm:
            prompt = f"[INST] {prompt} [/INST]"

        return tokenizer.encode(prompt, return_tensors="pt").to(device)

    logger.info("Ready.")

    def respond(prompt: str, stream: bool, image_path: str | None = None) -> None:
        nonlocal pixel_values

        # Load image dynamically for chat mode
        if is_vlm and image_path and image_processor:
            from PIL import Image

            path = Path(image_path)
            if not path.exists():
                print(f"Error: Image not found: {image_path}")
                return
            logger.info(f"Loading image: {image_path}")
            image = Image.open(path)
            processed = image_processor.preprocess(image)
            pixel_values = processed["pixel_values"].to(device, dtype=model.dtype)

        input_ids = tokenize_prompt(prompt)
        logger.info(f"Input tokens: {input_ids.shape[1]}, device: {input_ids.device}")
        logger.info(f"Params: max_tokens={args.max_tokens}, temp={args.temperature}, "
                     f"top_p={args.top_p}, rep_penalty={args.repetition_penalty}")
        if stream:
            generated_tokens: list[int] = []
            prev_text = ""
            token_count = 0
            logger.info("Starting streaming generation...")
            for token_id in generator.generate_stream(
                input_ids,
                pixel_values=pixel_values,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            ):
                token_count += 1
                if token_count == 1:
                    logger.info(f"First token generated (prefill done)")
                generated_tokens.append(token_id)
                full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(full_text[len(prev_text):], end="", flush=True)
                prev_text = full_text
            logger.info(f"Generation complete: {token_count} tokens")
            print()
        else:
            with torch.no_grad():
                output_ids = generator.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[0, input_ids.shape[1]:]
            print(tokenizer.decode(generated_ids, skip_special_tokens=True))

        # Clear pixel_values after use (for next query in chat mode)
        if is_vlm:
            pixel_values = None

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
    if hasattr(args, "image") and args.image:
        print(f"Image: {args.image}")
    print("-" * 50)
    print("[Starting inference...]", flush=True)
    print("Output: ", end="", flush=True)
    respond(prompt, stream=not args.no_stream)
    print("\n[Done]", flush=True)


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat session."""
    is_vlm = _is_vlm_model(args.model)
    respond = _load_generator(args)
    print("=" * 50)
    print("Interactive Chat Mode (Ctrl+C to exit)")
    if is_vlm:
        print("  /image <path>  Load an image for the next question")
    print("=" * 50)
    print()

    pending_image: str | None = None
    try:
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            # Handle /image command for VLMs
            if is_vlm and prompt.startswith("/image "):
                pending_image = prompt[len("/image "):].strip()
                if not Path(pending_image).exists():
                    print(f"  Image not found: {pending_image}")
                    pending_image = None
                else:
                    print(f"  Image loaded: {pending_image}")
                    print("  (Ask a question about it)")
                continue

            print("\nAssistant: ", end="", flush=True)
            respond(prompt, stream=not args.no_stream, image_path=pending_image)
            pending_image = None
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


def _print_help() -> None:
    """Print friendly help message."""
    from .data.registry import MODEL_REGISTRY

    print()
    print("lolama - LLaMA from scratch in PyTorch")
    print()
    print("Usage:")
    print('  lolama generate "your prompt"    Generate text from a prompt')
    print("  lolama chat                      Interactive chat session")
    print("  lolama models                    List available models")
    print()
    print("Options:")
    print("  -m, --model MODEL       Model alias or path (default: tinyllama)")
    print("  -i, --image PATH        Image file for VLM models (e.g., llava-1.5-7b)")
    print("  --quantize              Use int8 quantization")
    print("  --max-tokens N          Max tokens to generate (default: 256)")
    print("  --temperature F         Sampling temperature (default: 0.7)")
    print("  --top-p F               Top-p sampling (default: 0.9)")
    print("  --no-stream             Disable streaming output")
    print("  -v, --verbose           Show model loading details")
    print("  -V, --version           Show version")
    print()
    print("Examples:")
    print('  lolama generate "The meaning of life is"')
    print('  lolama generate "Hello" -m open_llama_3b')
    print("  lolama chat -m tinyllama --quantize")
    print('  echo "Hello" | lolama generate')
    print()
    print("Vision-Language (VLM) examples:")
    print('  lolama generate "Describe this image" -m llava-1.5-7b --image photo.jpg')
    print('  lolama generate "What abnormalities do you see?" -m llava-med --image xray.png')
    print()
    models = list(MODEL_REGISTRY.items())[:3]
    print(f"Models: {', '.join(alias for alias, _ in models)}, ...")
    print("Run 'lolama models' for full list.")
    print()


def main() -> None:
    """Main CLI entry point."""
    # Handle 'lolama help' and 'lolama' with no args
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        _print_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        prog="lolama",
        description="LLaMA from scratch in PyTorch",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("-V", "--version", action="version", version=f"lolama {_get_version()}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show model loading details")
    parser.add_argument("--debug", action="store_true", help="Show all debug output")
    subparsers = parser.add_subparsers(dest="command")

    # Shared sampling options for generate and chat
    def add_model_args(p: argparse.ArgumentParser, include_image: bool = False) -> None:
        p.add_argument("-m", "--model", default="tinyllama",
                       help="Model alias or local path (default: tinyllama)")
        p.add_argument("--no-stream", action="store_true", help="Disable streaming (wait for full response)")
        p.add_argument("--quantize", action="store_true", help="Use int8 quantization")
        p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
        p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
        p.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
        p.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
        if include_image:
            p.add_argument("--image", "-i", type=str, help="Path to image file (for VLM models like llava-1.5-7b)")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", nargs="?", help="The prompt to generate from")
    add_model_args(gen_parser, include_image=True)
    gen_parser.set_defaults(func=cmd_generate)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    add_model_args(chat_parser, include_image=True)
    chat_parser.set_defaults(func=cmd_chat)

    # models
    models_parser = subparsers.add_parser("models", help="List available model aliases")
    models_parser.set_defaults(func=cmd_models)

    args = parser.parse_args()

    if args.help or args.command is None:
        _print_help()
        sys.exit(0)

    # Configure logging level
    import logging
    from .utils import set_verbosity
    if args.debug:
        set_verbosity(logging.DEBUG)
    elif args.verbose:
        set_verbosity(logging.INFO)
    else:
        set_verbosity(logging.WARNING)

    args.func(args)


if __name__ == "__main__":
    main()
