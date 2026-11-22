"""Image preprocessing for pretrained feature extractors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torchvision import transforms

from modules.vision.config import load_config

# ImageNet statistics used by every torchvision pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(config: Mapping[str, Any] | str | Path) -> transforms.Compose:
    """Build the image resize and normalization transform.

    ``resize`` may be an int (square output) or ``[height, width]``. Aligned CelebA
    crops are 218x178, so a square resize slightly stretches faces horizontally;
    this keeps the whole face instead of cropping it away. Defaults to 224.
    """
    cfg = load_config(config)
    resize = cfg.get("resize") or 224
    if isinstance(resize, int):
        size = (resize, resize)
    elif isinstance(resize, (list, tuple)) and len(resize) == 2:
        size = (int(resize[0]), int(resize[1]))
    else:
        raise ValueError("resize must be an int or a [height, width] pair.")
    if min(size) <= 0:
        raise ValueError("resize values must be positive.")

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
