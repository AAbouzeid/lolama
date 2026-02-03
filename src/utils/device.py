"""Device detection utilities."""

from __future__ import annotations

import torch


def resolve_device(preferred: str | None = None) -> str:
    """Resolve the best available device.

    Priority order:
    1. If preferred is specified and available, use it
    2. CUDA if available
    3. MPS if available (Apple Silicon)
    4. CPU as fallback

    Args:
        preferred: Optional preferred device ("cuda", "mps", "cpu").
            If specified and available, this device is used.

    Returns:
        Device string suitable for torch operations
    """
    if preferred is not None:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if preferred == "cpu":
            return "cpu"
        # Preferred not available, fall through to auto-detection

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
