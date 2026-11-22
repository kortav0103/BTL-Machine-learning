"""Feature serialization and associated sample metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.vision.config import load_config




def save_features(
    features: np.ndarray,
    manifest: pd.DataFrame,
    config: dict[str, Any] | str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Save features with ordered image IDs, labels, splits, and extraction settings.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (N, D).
    manifest : pd.DataFrame
        Dataset manifest containing 'image_id', 'partition', and target attribute columns.
    config : dict or Path/str
        Configuration dictionary containing extraction settings.
    output_path : str or Path, optional
        Custom destination file path. If None, saved under config['features_dir'].

    Returns
    -------
    Path
        Path to the saved .npz archive.
    """
    if len(features) != len(manifest):
        raise ValueError(
            f"Row count mismatch: features has {len(features)} rows, "
            f"but manifest has {len(manifest)} rows."
        )

    cfg = load_config(config)
    extractor = str(cfg.get("extractor", "features"))

    # Extract aligned metadata
    image_ids = manifest["image_id"].astype(str).to_numpy()
    splits = manifest["partition"].to_numpy()

    # Identify attribute columns
    non_target_cols = {"image_id", "image_path", "partition"}
    target_names = [c for c in manifest.columns if c not in non_target_cols]
    labels = manifest[target_names].to_numpy() if target_names else np.empty((len(manifest), 0))

    metadata = {
        "extractor": extractor,
        "feature_dim": int(features.shape[1]) if features.ndim > 1 else 1,
        "num_samples": int(len(features)),
        "target_names": target_names,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }

    if output_path is None:
        features_dir = Path(cfg.get("features_dir", "features"))
        output_file = features_dir / f"{extractor}_celeba.npz"
    else:
        output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        features=features.astype(np.float32),
        image_ids=image_ids,
        splits=splits,
        labels=labels,
        target_names=np.array(target_names, dtype=object),
        metadata=json.dumps(metadata, indent=2),
    )
    return output_file


def load_features(path: str | Path) -> dict[str, Any]:
    """Load features and metadata, validating row alignment.

    Parameters
    ----------
    path : str or Path
        Path to the serialized .npz feature file.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'features': np.ndarray of shape (N, D)
        - 'image_ids': np.ndarray of shape (N,)
        - 'splits': np.ndarray of shape (N,) with partition codes
        - 'labels': np.ndarray of shape (N, num_targets)
        - 'target_names': list[str] of target attribute names
        - 'metadata': dict containing extraction settings and info
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    with np.load(file_path, allow_pickle=True) as archive:
        required_keys = ["features", "image_ids", "splits", "labels"]
        for key in required_keys:
            if key not in archive:
                raise KeyError(f"Corrupted feature archive: missing key '{key}' in {file_path}")

        features = archive["features"]
        image_ids = archive["image_ids"]
        splits = archive["splits"]
        labels = archive["labels"]

        target_names = (
            archive["target_names"].tolist()
            if "target_names" in archive
            else []
        )

        metadata_raw = archive["metadata"].item() if "metadata" in archive else "{}"
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else dict(metadata_raw)

    # Validate row alignment across all components
    n_samples = len(features)
    if len(image_ids) != n_samples:
        raise ValueError(
            f"Row alignment error: features has {n_samples} rows, "
            f"but image_ids has {len(image_ids)} rows."
        )
    if len(splits) != n_samples:
        raise ValueError(
            f"Row alignment error: features has {n_samples} rows, "
            f"but splits has {len(splits)} rows."
        )
    if len(labels) != n_samples:
        raise ValueError(
            f"Row alignment error: features has {n_samples} rows, "
            f"but labels has {len(labels)} rows."
        )

    return {
        "features": features,
        "image_ids": image_ids,
        "splits": splits,
        "labels": labels,
        "target_names": target_names,
        "metadata": metadata,
    }
