"""Image EDA for CelebA. Dependencies: pandas, numpy, Pillow, matplotlib.

Scan original files without resizing. Pixel statistics use RGB values in
[0, 1] and are computed per image, so the summary weights each image equally.
Pass training filenames for training-set EDA; no labels or splits are inferred.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUMERIC_COLUMNS = [
    "width", "height", "aspect_ratio", "file_size_kb",
    "red_mean", "green_mean", "blue_mean", "brightness_mean", "brightness_std",
]
COLUMNS = ["image_id", "mode", *NUMERIC_COLUMNS, "error"]


def scan_images(
    image_dir: str | Path,
    image_ids=None,
    max_images: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inspect images and return one row per file, preserving failures.

    By default, recursively discover supported image files. Optional
    ``image_ids`` are relative paths (e.g. training filenames from metadata).
    Missing requested files are reported, not silently dropped. Duplicate
    requested paths and paths outside the directory are rejected.
    ``max_images`` selects a reproducible random sample before decoding;
    the returned table describes only that sample, not the whole dataset.
    Each image is decoded individually to keep memory use bounded.
    """
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if max_images is not None and (
        not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < 1
    ):
        raise ValueError("max_images must be a positive integer or None.")
    if image_ids is None:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        paths = []
        for image_id in image_ids:
            relative = Path(str(image_id))
            if relative.is_absolute() or ".." in relative.parts or str(relative) == ".":
                raise ValueError(f"Expected a relative image path: {image_id}")
            paths.append(root / relative)
        if len(paths) != len(set(paths)):
            raise ValueError("image_ids contains duplicate paths.")
        paths.sort()
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Image path is outside image_dir: {path}")
    if not paths:
        raise ValueError("No images found or requested.")
    if max_images is not None and len(paths) > max_images:
        indices = np.random.default_rng(random_state).choice(len(paths), max_images, replace=False)
        paths = [paths[i] for i in sorted(indices)]

    rows = []
    for path in paths:
        row = dict.fromkeys(COLUMNS)
        row["image_id"] = path.relative_to(root).as_posix()
        try:
            row["file_size_kb"] = path.stat().st_size / 1024
            with Image.open(path) as image:
                image.load()  # Decode completely to catch truncated/corrupt files.
                width, height = image.size
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                brightness = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                row.update(
                    mode=image.mode, width=width, height=height,
                    aspect_ratio=width / height,
                    red_mean=float(pixels[:, :, 0].mean()),
                    green_mean=float(pixels[:, :, 1].mean()),
                    blue_mean=float(pixels[:, :, 2].mean()),
                    brightness_mean=float(brightness.mean()),
                    brightness_std=float(brightness.std()),
                )
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    table = pd.DataFrame(rows, columns=COLUMNS)
    table[NUMERIC_COLUMNS] = table[NUMERIC_COLUMNS].astype(float)
    return table


def check_images(df: pd.DataFrame) -> dict[str, int]:
    """Count decoded images, read failures, color modes, and distinct sizes.

    Non-RGB files and varying sizes are review points, not necessarily errors.
    This does not detect duplicate image contents or person identities.
    """
    valid = df.loc[df["error"].isna()]
    return {
        "total_images": len(df),
        "readable_images": len(valid),
        "unreadable_images": int(df["error"].notna().sum()),
        "non_rgb_images": int(valid["mode"].ne("RGB").sum()),
        "distinct_dimensions": len(valid[["width", "height"]].drop_duplicates()),
    }


def summarize_images(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for readable images only."""
    return df.loc[df["error"].isna(), NUMERIC_COLUMNS].describe().T.rename_axis("metric")


def _save_figure(fig, save_path):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_image_distributions(df: pd.DataFrame, save_path: str | Path | None = None):
    """Plot dimensions, sizes, and brightness; return a Matplotlib Figure."""
    import matplotlib.pyplot as plt

    valid = df.loc[df["error"].isna()]
    if valid.empty:
        raise ValueError("No readable images to plot.")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = ["width", "height", "aspect_ratio", "file_size_kb", "brightness_mean", "brightness_std"]
    labels = ["Width (pixels)", "Height (pixels)", "Width / height", "File size (KiB)",
              "Mean luminance (0–1)", "Luminance standard deviation (0–1)"]
    for ax, metric, label in zip(axes.flat, metrics, labels):
        ax.hist(valid[metric], bins=30, color="#4C78A8", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of images")
    fig.suptitle(f"Image distributions (n={len(valid):,} readable images)")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_sample_images(
    df: pd.DataFrame,
    image_dir: str | Path,
    n_images: int = 12,
    random_state: int = 42,
    save_path: str | Path | None = None,
):
    """Show a reproducible gallery of readable images; return a Figure."""
    import matplotlib.pyplot as plt

    if not isinstance(n_images, int) or isinstance(n_images, bool) or n_images < 1:
        raise ValueError("n_images must be a positive integer.")
    valid = df.loc[df["error"].isna()]
    if valid.empty:
        raise ValueError("No readable images to plot.")
    sample = valid.sample(min(n_images, len(valid)), random_state=random_state)
    root = Path(image_dir).resolve()
    images = []
    for image_id in sample["image_id"]:
        path = (root / str(image_id)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path is outside image_dir: {image_id}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    ncols = min(4, len(images))
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows), squeeze=False)
    for ax, image_id, image in zip(axes.flat, sample["image_id"], images):
        ax.imshow(image)
        ax.set_title(f"{image_id}\n{image.width} × {image.height}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
