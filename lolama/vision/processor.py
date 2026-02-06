"""CLIP Image Processor for LLaVA."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..model.vlm_config import VisionConfig


class CLIPImageProcessor:
    """Preprocesses images for CLIP vision encoder.

    Handles:
    - Resizing to target size (336x336)
    - Converting to tensor
    - Normalizing with CLIP mean/std

    Usage:
        processor = CLIPImageProcessor()
        image = Image.open("photo.jpg")
        result = processor.preprocess(image)
        pixel_values = result["pixel_values"]  # (1, 3, 336, 336)
    """

    # CLIP normalization constants (from OpenAI CLIP)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        image_size: int = 336,
        do_resize: bool = True,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        image_mean: tuple[float, ...] | None = None,
        image_std: tuple[float, ...] | None = None,
    ) -> None:
        self.image_size = image_size
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.image_mean = image_mean or self.CLIP_MEAN
        self.image_std = image_std or self.CLIP_STD

    def preprocess(
        self,
        images: Image.Image | list[Image.Image],
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Preprocess image(s) for CLIP.

        Args:
            images: Single PIL Image or list of PIL Images
            return_tensors: Return format ("pt" for PyTorch tensors)

        Returns:
            Dictionary with "pixel_values" tensor of shape (batch, 3, H, W)
        """
        if isinstance(images, Image.Image):
            images = [images]

        processed = []
        for image in images:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize
            if self.do_resize:
                image = self._resize(image)

            # Center crop
            if self.do_center_crop:
                image = self._center_crop(image)

            # Convert to tensor via numpy (H, W, C) -> (C, H, W), scale to [0, 1]
            tensor = torch.from_numpy(np.array(image)).float()
            tensor = tensor.permute(2, 0, 1) / 255.0

            # Normalize
            if self.do_normalize:
                tensor = self._normalize(tensor)

            processed.append(tensor)

        # Stack into batch
        pixel_values = torch.stack(processed, dim=0)

        return {"pixel_values": pixel_values}

    def _resize(self, image: Image.Image) -> Image.Image:
        """Resize image to target size, maintaining aspect ratio then cropping."""
        # Resize so shortest side is image_size
        width, height = image.size
        if width < height:
            new_width = self.image_size
            new_height = int(height * self.image_size / width)
        else:
            new_height = self.image_size
            new_width = int(width * self.image_size / height)

        return image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        """Center crop to image_size x image_size."""
        width, height = image.size
        left = (width - self.image_size) // 2
        top = (height - self.image_size) // 2
        right = left + self.image_size
        bottom = top + self.image_size
        return image.crop((left, top, right, bottom))

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor with CLIP mean and std.

        Args:
            tensor: (C, H, W) tensor with values in [0, 1]

        Returns:
            Normalized tensor
        """
        mean = torch.tensor(self.image_mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    @classmethod
    def from_config(cls, config: VisionConfig) -> CLIPImageProcessor:
        """Create processor from VisionConfig."""
        return cls(image_size=config.image_size)
