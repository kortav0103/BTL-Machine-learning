"""CelebA dataset preparation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from modules.vision.config import load_config

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
ATTR_FILE = "list_attr_celeba.csv"
PARTITION_FILE = "list_eval_partition.csv"
IMAGE_DIR = "img_align_celeba"


def resolve_image_dir(data_dir: str | Path) -> Path:
    """Return the aligned-image folder, handling Kaggle's nested ``img_align_celeba/img_align_celeba``."""
    base = Path(data_dir) / IMAGE_DIR
    nested = base / IMAGE_DIR
    return nested if nested.is_dir() else base


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CelebA file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    df["image_id"] = df["image_id"].astype(str).str.strip()
    return df


def _sample_split(df: pd.DataFrame, size: int | None, seed: int) -> pd.DataFrame:
    if size is None or size >= len(df):
        return df
    return df.sample(n=int(size), random_state=seed)


def _split_sizes(sample_size: Any) -> dict[str, int | None]:
    """Normalize ``sample_size`` (None, int, or per-split mapping) into per-split sizes."""
    if sample_size is None:
        return dict.fromkeys(SPLIT_CODES)
    if isinstance(sample_size, int):
        return dict.fromkeys(SPLIT_CODES, sample_size)
    if isinstance(sample_size, Mapping):
        unknown = set(sample_size) - set(SPLIT_CODES)
        if unknown:
            raise ValueError(f"Unknown split names in sample_size: {sorted(unknown)}")
        return {name: sample_size.get(name) for name in SPLIT_CODES}
    raise TypeError("sample_size must be null, an integer, or a mapping of split -> size.")


def prepare_data(config: Mapping[str, Any] | str | Path) -> pd.DataFrame:
    """Prepare a manifest containing image paths, target labels, and official splits.

    Uses the official CelebA partition (0 train, 1 validation, 2 test), samples each
    split independently with ``seed``, and converts -1/1 attribute labels to 0/1.

    Returns
    -------
    pd.DataFrame
        Columns ``image_id``, ``image_path``, ``partition`` and one column per target,
        sorted by partition then image_id so row order is deterministic.
    """
    cfg = load_config(config)
    targets = cfg.get("targets")
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        raise ValueError("config['targets'] must name at least one CelebA attribute.")

    data_dir = Path(cfg.get("data_dir", "data"))
    seed = int(cfg.get("seed", 42))

    attrs = _read_table(data_dir / ATTR_FILE, ["image_id", *targets])
    parts = _read_table(data_dir / PARTITION_FILE, ["image_id", "partition"])
    df = parts[["image_id", "partition"]].merge(
        attrs[["image_id", *targets]], on="image_id", how="inner", validate="one_to_one"
    )
    if df.empty:
        raise ValueError("No images are shared between the attribute and partition tables.")

    for target in targets:
        values = set(df[target].unique().tolist())
        if not values <= {-1, 1}:
            raise ValueError(f"Attribute '{target}' must contain only -1/1, got {sorted(values)}.")
        df[target] = (df[target] == 1).astype(int)

    sizes = _split_sizes(cfg.get("sample_size"))
    pieces = [
        _sample_split(df[df["partition"] == code], sizes[name], seed)
        for name, code in SPLIT_CODES.items()
    ]
    manifest = pd.concat(pieces).sort_values(["partition", "image_id"]).reset_index(drop=True)

    image_dir = resolve_image_dir(data_dir)
    manifest.insert(1, "image_path", [str(image_dir / i) for i in manifest["image_id"]])
    missing = [p for p in manifest["image_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images in the manifest are missing, e.g. {missing[0]}"
        )
    return manifest
