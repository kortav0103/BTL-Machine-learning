"""Pretrained image feature extraction using PyTorch and Torchvision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from modules.vision.data.preprocessing import build_transform

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader


def _load_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Parse configuration from dict or JSON file path."""
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Expected config to be a dict or Path, got {type(config).__name__}")


class CelebADataset:
    """PyTorch Dataset loading images strictly by manifest row order."""

    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.paths = manifest["image_path"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image at {path}: {exc}") from exc

        if self.transform is not None:
            image = self.transform(image)
        return image, idx


def build_extractor(model_name: str, device: str = "cpu"):
    """Instantiate a pretrained feature extractor with classification head replaced by Identity.

    Supported models:
    - 'resnet50': 2048-dimensional global average pooled features
    - 'mobilenet_v3_large': 960-dimensional pooled features
    - 'resnet18': 512-dimensional residual features

    Returns
    -------
    tuple[torch.nn.Module, int]
        Model set to eval mode on specified device, and its feature dimension.
    """
    name = model_name.lower().strip()
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features  # 2048
        model.fc = nn.Identity()
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        # pooled representation before classifier has 960 channels
        feature_dim = model.classifier[0].in_features  # 960
        model.classifier = nn.Identity()
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features  # 512
        model.fc = nn.Identity()
    else:
        supported = ["resnet50", "mobilenet_v3_large", "resnet18"]
        raise ValueError(f"Unsupported extractor '{model_name}'. Supported models: {supported}")

    model = model.to(device)
    model.eval()
    return model, feature_dim


def extract_features(manifest: pd.DataFrame, config: dict[str, Any] | str | Path) -> np.ndarray:
    """Extract image features in manifest row order using a pretrained vision backbone.

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest containing 'image_path' and 'image_id' in desired row order.
    config : dict or Path/str
        Configuration containing 'extractor' (e.g. 'resnet50', 'mobilenet_v3_large'),
        'batch_size', 'device', 'num_workers', etc.

    Returns
    -------
    np.ndarray
        Extracted features of shape (len(manifest), feature_dim) with dtype float32.
    """
    if manifest.empty:
        raise ValueError("Cannot extract features from an empty manifest.")
    if "image_path" not in manifest.columns:
        raise ValueError("Manifest must contain an 'image_path' column.")

    cfg = _load_config(config)
    model_name = cfg.get("extractor", "resnet50")
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 2 if torch.cuda.is_available() else 0))

    # Determine device
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device_cfg)

    # Build model & transforms
    model, feature_dim = build_extractor(model_name, device=device)
    transform = build_transform(cfg)

    dataset = CelebADataset(manifest, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Strictly preserve manifest order
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    n_samples = len(manifest)
    features = np.empty((n_samples, feature_dim), dtype=np.float32)

    progress = tqdm(loader, desc=f"Extracting {model_name} features ({device.upper()})", total=len(loader))
    with torch.no_grad():
        for batch_imgs, batch_indices in progress:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_feats = model(batch_imgs)
            # Flatten in case of extra singleton spatial dimensions
            batch_feats = torch.flatten(batch_feats, start_dim=1)
            batch_feats_np = batch_feats.cpu().numpy().astype(np.float32)

            start_idx = batch_indices[0].item()
            end_idx = start_idx + len(batch_feats_np)
            features[start_idx:end_idx] = batch_feats_np

    return features
