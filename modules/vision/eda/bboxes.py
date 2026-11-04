"""CelebA bounding box EDA. Input: CSV with image_id, x_1, y_1, width, height (pixels,
top-left corner plus size) referring to the original, unaligned images.

Dependencies: pandas, matplotlib, Pillow.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

BBOX_COORDINATE_COLUMNS = ["x_1", "y_1", "width", "height"]
REQUIRED_BBOX_COLUMNS = ["image_id", *BBOX_COORDINATE_COLUMNS]

# (column, axis label, color) for each histogram panel.
DISTRIBUTION_PANELS = [
    ("x_1", "X min (x_1)", "#4C78A8"),
    ("y_1", "Y min (y_1)", "#4C78A8"),
    ("width", "Box width (px)", "#54A24B"),
    ("height", "Box height (px)", "#54A24B"),
    ("aspect_ratio", "Aspect ratio (width / height)", "#F2CF5B"),
    ("area", "Box area (px²)", "#E45756"),
]


def _save_figure(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _image_ids(df):
    """image_id values with whitespace trimmed; blanks become <NA>."""
    return df["image_id"].astype("string").str.strip().replace("", pd.NA)


def _numeric_bboxes(df):
    """Coordinates as numbers; unparseable and infinite values become NaN."""
    numeric = df[BBOX_COORDINATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([float("inf"), float("-inf")], float("nan"))


def _with_derived(numeric):
    """Add aspect_ratio and area, NaN where width or height is not positive."""
    valid = numeric["width"].gt(0) & numeric["height"].gt(0)
    return numeric.assign(
        aspect_ratio=(numeric["width"] / numeric["height"]).where(valid),
        area=(numeric["width"] * numeric["height"]).where(valid),
    )


def _outside_image(x, y, w, h, image_w, image_h):
    """True where any part of the box lies outside the image (scalars or aligned Series)."""
    return (x < 0) | (y < 0) | (x + w > image_w) | (y + h > image_h)


def _load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def load_bboxes(path: str | Path) -> pd.DataFrame:
    """Load bounding boxes from a CSV; odd values are kept for check_bboxes to report."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bounding box file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[REQUIRED_BBOX_COLUMNS].copy()
    if df.empty:
        raise ValueError("The bounding box table contains no data rows.")
    return df.assign(image_id=_image_ids(df))


def check_bboxes(
    df: pd.DataFrame,
    partitions: pd.DataFrame | None = None,
    images_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Count box problems without altering rows.

    Cell counts: missing (blank) and invalid (not a finite number) coordinates. Row counts:
    negative x_1/y_1, non-positive width/height, and any of these combined. ``partitions``
    adds IDs found in only one table. ``images_df`` (image_id, width, height; e.g. from
    scan_images) adds boxes with any part outside their image and box IDs with no image.
    """
    ids = _image_ids(df)
    numeric = _numeric_bboxes(df)
    text = df[BBOX_COORDINATE_COLUMNS].astype("string").apply(lambda c: c.str.strip())
    missing = text.fillna("").eq("").astype(bool)
    invalid = numeric.isna() & ~missing
    negative = numeric[["x_1", "y_1"]].lt(0).any(axis=1)
    bad_size = numeric[["width", "height"]].le(0).any(axis=1)

    result = {
        "total_rows": len(df),
        "missing_image_ids": int(ids.isna().sum()),
        "duplicate_image_ids": int(ids.dropna().duplicated().sum()),
        "missing_coordinate_values": int(missing.to_numpy().sum()),
        "invalid_coordinate_values": int(invalid.to_numpy().sum()),
        "rows_with_negative_coordinates": int(negative.sum()),
        "rows_with_non_positive_dimensions": int(bad_size.sum()),
        "rows_with_bbox_issues": int((missing.any(axis=1) | invalid.any(axis=1) | negative | bad_size).sum()),
    }
    if partitions is not None:
        bbox_ids, part_ids = set(ids.dropna()), set(_image_ids(partitions).dropna())
        result["bbox_ids_without_partition"] = len(bbox_ids - part_ids)
        result["partition_ids_without_bbox"] = len(part_ids - bbox_ids)
    if images_df is not None:
        sizes = pd.DataFrame({
            "image_id": _image_ids(images_df),
            "image_w": pd.to_numeric(images_df["width"], errors="coerce"),
            "image_h": pd.to_numeric(images_df["height"], errors="coerce"),
        }).dropna(subset=["image_id"])
        boxes = numeric.assign(image_id=ids).dropna(subset=["image_id"])
        m = boxes.merge(sizes, on="image_id", validate="many_to_one")  # raises if images_df repeats an image
        result["bbox_ids_without_image"] = len(set(boxes["image_id"]) - set(sizes["image_id"]))
        result["bboxes_exceeding_image_bounds"] = int(_outside_image(m.x_1, m.y_1, m.width, m.height, m.image_w, m.image_h).sum())
    return result


def summarize_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """Describe x_1, y_1, width, height, aspect_ratio and area (the last two only for positive sizes)."""
    return _with_derived(_numeric_bboxes(df)).describe().T.rename_axis("metric")


def plot_bboxes_on_images(
    df: pd.DataFrame,
    image_dir: str | Path,
    n_images: int = 12,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay boxes on random images; boxes crossing the image edge get an [OOB] tag.

    Only rows with an ID, finite coordinates and positive size are eligible; duplicate IDs are rejected.
    """
    ids = _image_ids(df)
    if ids.dropna().duplicated().any():
        raise ValueError("Duplicate image IDs in the bounding box table; run check_bboxes first.")
    numeric = _numeric_bboxes(df)
    ok = ids.notna() & numeric.notna().all(axis=1) & numeric["width"].gt(0) & numeric["height"].gt(0)
    candidates = numeric[ok].assign(image_id=ids[ok])
    if n_images < 1 or candidates.empty:
        raise ValueError("Need n_images >= 1 and at least one valid bounding box.")
    sample = candidates.sample(n=min(n_images, len(candidates)), random_state=random_state)
    # Load every image first so a missing file doesn't leave a half-drawn figure.
    images = [_load_rgb(Path(image_dir) / i) for i in sample["image_id"]]

    ncols = min(4, len(sample))
    nrows = math.ceil(len(sample) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, b, image in zip(axes.flat, sample.itertuples(index=False), images):
        ax.imshow(image)
        ax.add_patch(patches.Rectangle((b.x_1, b.y_1), b.width, b.height,
                                       linewidth=2, edgecolor="#E45756", facecolor="none"))
        oob = " [OOB]" if _outside_image(b.x_1, b.y_1, b.width, b.height, *image.size) else ""
        ax.set_title(f"{b.image_id}\nBox: {b.width:g}×{b.height:g} | Img: {image.width}×{image.height}{oob}", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_bbox_distributions(df: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    """Histograms of box position, size, aspect ratio and area; the top 1% of each metric is hidden."""
    metrics = _with_derived(_numeric_bboxes(df)).dropna()
    if metrics.empty:
        raise ValueError("No valid bounding boxes to plot.")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (column, label, color) in zip(axes.flat, DISTRIBUTION_PANELS):
        values = metrics[column]
        ax.hist(values[values <= values.quantile(0.99)], bins=35, color=color, edgecolor="white")
        ax.set(xlabel=label, ylabel="Count")
    fig.suptitle(f"CelebA bounding box distributions (n={len(metrics):,}; top 1% of each metric hidden)", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig