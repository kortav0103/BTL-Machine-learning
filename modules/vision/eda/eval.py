"""Exploratory data analysis for CelebA bounding boxes and evaluation attribute labels.

Supports bounding box annotations (x_1, y_1, width, height) and binary facial
attribute labels (-1, 1). Reads CSV files or original CelebA TXT files.

Dependencies: pandas, numpy, matplotlib, Pillow.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BBOX_COORDINATE_COLUMNS = ["x_1", "y_1", "width", "height"]
REQUIRED_BBOX_COLUMNS = ["image_id", *BBOX_COORDINATE_COLUMNS]

CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def _save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def _image_ids(df: pd.DataFrame) -> pd.Series:
    """Normalize image_id values by trimming whitespace."""
    return df["image_id"].astype("string").str.strip().replace("", pd.NA)


# ==============================================================================
# Bounding Box EDA
# ==============================================================================

def _validate_bbox_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_BBOX_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Bounding box table is missing columns: {', '.join(missing)}")


def _numeric_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bounding box coordinates to numeric, invalid entries become NaN."""
    numeric = df[BBOX_COORDINATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([float("inf"), float("-inf")], float("nan"))


def load_bboxes(path: str | Path) -> pd.DataFrame:
    """Load CelebA bounding boxes from CSV or original TXT file.

    In CSV files, headers must include ``image_id, x_1, y_1, width, height``.
    In original TXT files, the first line is image count, the second line
    is column names, followed by image rows.

    Anomalies and invalid values are preserved for ``check_bboxes`` to report.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bounding box file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
    elif path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8-sig") as stream:
            count_line = stream.readline().strip()
            header = stream.readline().split()
        try:
            expected_count = int(count_line)
        except ValueError as exc:
            raise ValueError("The first line of a bbox TXT file must be the image count.") from exc

        if len(header) == 4 and set(header) == set(BBOX_COORDINATE_COLUMNS):
            df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, encoding="utf-8-sig")
            if df.shape[1] != 5:
                raise ValueError("Each TXT data row must contain an image filename and 4 coordinates.")
            df.columns = ["image_id", *header]
        elif len(header) == 5 and set(header) == set(REQUIRED_BBOX_COLUMNS):
            df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, encoding="utf-8-sig")
            df.columns = header
        else:
            raise ValueError(f"Unexpected header format in bbox TXT file: {header}")

        if len(df) != expected_count:
            raise ValueError(f"TXT declares {expected_count} images, but {len(df)} data rows were read.")
    else:
        raise ValueError("Bounding box files must have a .csv or .txt extension.")

    _validate_bbox_columns(df)
    if df.empty:
        raise ValueError("The bounding box table contains no data rows.")

    df = df[REQUIRED_BBOX_COLUMNS].copy()
    df["image_id"] = _image_ids(df)
    return df


def check_bboxes(
    df: pd.DataFrame,
    partitions: pd.DataFrame | None = None,
    images_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Validate bounding box metadata without modifying or dropping input rows.

    Checks for:
    - Missing or duplicate image IDs
    - Missing or non-numeric coordinate values
    - Negative coordinate values (x_1 < 0 or y_1 < 0)
    - Non-positive dimensions (width <= 0 or height <= 0)
    - Inconsistencies with partition metadata if provided
    - Bounding boxes exceeding image boundaries if images_df is provided
    """
    _validate_bbox_columns(df)
    ids = _image_ids(df)
    raw = df[BBOX_COORDINATE_COLUMNS].astype("string").apply(lambda col: col.str.strip())
    missing = raw.isna() | raw.eq("").fillna(False)
    numeric = _numeric_bboxes(df)
    invalid = numeric.isna() & ~missing

    coords = numeric[["x_1", "y_1"]]
    dims = numeric[["width", "height"]]
    negative_coords = coords.lt(0).any(axis=1)
    non_positive_dims = dims.le(0).any(axis=1)

    result = {
        "total_rows": len(df),
        "missing_image_ids": int(ids.isna().sum()),
        "duplicate_image_ids": int(ids.dropna().duplicated().sum()),
        "missing_coordinate_values": int(missing.to_numpy().sum()),
        "invalid_coordinate_values": int(invalid.to_numpy().sum()),
        "negative_coordinate_values": int(negative_coords.sum()),
        "non_positive_dimension_values": int(non_positive_dims.sum()),
        "rows_with_bbox_issues": int(
            (missing.any(axis=1) | invalid.any(axis=1) | negative_coords | non_positive_dims).sum()
        ),
    }

    if partitions is not None:
        if not {"image_id", "partition"}.issubset(partitions.columns):
            raise ValueError("The partition table must contain image_id and partition columns.")
        bbox_ids = set(ids.dropna())
        partition_ids = set(_image_ids(partitions).dropna())
        result["bbox_ids_without_partition"] = len(bbox_ids - partition_ids)
        result["partition_ids_without_bbox"] = len(partition_ids - bbox_ids)

    if images_df is not None:
        if not {"image_id", "width", "height"}.issubset(images_df.columns):
            raise ValueError("images_df must contain image_id, width, and height columns.")
        merged = pd.merge(
            df[["image_id", "x_1", "y_1", "width", "height"]],
            images_df[["image_id", "width", "height"]].rename(
                columns={"width": "img_width", "height": "img_height"}
            ),
            on="image_id",
            how="inner",
        )
        x2 = merged["x_1"] + merged["width"]
        y2 = merged["y_1"] + merged["height"]
        exceeds = (x2 > merged["img_width"]) | (y2 > merged["img_height"])
        result["bboxes_exceeding_image_bounds"] = int(exceeds.sum())

    return result


def summarize_bboxes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize bounding box dimensions, aspect ratios, and areas.

    Returns descriptive statistics for x_1, y_1, width, height,
    aspect_ratio (width / height), and area (width * height).
    """
    _validate_bbox_columns(df)
    numeric = _numeric_bboxes(df).copy()
    valid_dims = numeric["width"].gt(0) & numeric["height"].gt(0)
    numeric["aspect_ratio"] = np.where(
        valid_dims, numeric["width"] / numeric["height"], np.nan
    )
    numeric["area"] = np.where(
        valid_dims, numeric["width"] * numeric["height"], np.nan
    )

    summary = numeric.describe().T
    summary.index.name = "metric"
    return summary


def plot_bboxes_on_images(
    df: pd.DataFrame,
    image_dir: str | Path,
    n_images: int = 12,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Sample raw images, overlay bounding box annotations, and return a Figure.

    CelebA bounding boxes correspond to original, unaligned images (e.g. ``data/img_celeba``).
    """
    _validate_bbox_columns(df)
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not isinstance(n_images, int) or isinstance(n_images, bool) or n_images < 1:
        raise ValueError("n_images must be a positive integer.")

    ids = _image_ids(df)
    if ids.dropna().duplicated().any():
        raise ValueError("The bounding box table contains duplicate filenames; validate it first.")

    numeric = _numeric_bboxes(df)
    valid_mask = (
        ids.notna()
        & numeric.notna().all(axis=1)
        & numeric["width"].gt(0)
        & numeric["height"].gt(0)
    )
    candidates = numeric.loc[valid_mask].copy()
    candidates.insert(0, "image_id", ids.loc[valid_mask])

    if candidates.empty:
        raise ValueError("No valid bounding box entries found to plot.")

    sample = candidates.sample(n=min(n_images, len(candidates)), random_state=random_state)

    images = []
    for _, row in sample.iterrows():
        image_id = str(row["image_id"])
        path = image_dir / image_id
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    ncols = min(4, len(sample))
    nrows = math.ceil(len(sample) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for ax, (_, row), img in zip(axes.flat, sample.iterrows(), images):
        ax.imshow(img)
        w_img, h_img = img.size
        x, y, w, h = row["x_1"], row["y_1"], row["width"], row["height"]

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor="#E45756",
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect)

        is_oob = (x + w > w_img) or (y + h > h_img) or (x < 0) or (y < 0)
        title = f"{row['image_id']}\nBox: {int(w)}×{int(h)} | Img: {w_img}×{h_img}"
        if is_oob:
            title += " [OOB]"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in list(axes.flat)[len(sample):]:
        ax.axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_bbox_distributions(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot histograms of bounding box metrics and return a Figure."""
    _validate_bbox_columns(df)
    numeric = _numeric_bboxes(df).dropna()
    valid_dims = numeric["width"].gt(0) & numeric["height"].gt(0)
    numeric = numeric.loc[valid_dims].copy()
    if numeric.empty:
        raise ValueError("No valid numeric bounding box records to plot.")

    numeric["aspect_ratio"] = numeric["width"] / numeric["height"]
    numeric["area"] = numeric["width"] * numeric["height"]

    metrics = ["x_1", "y_1", "width", "height", "aspect_ratio", "area"]
    titles = [
        "X min (x_1)", "Y min (y_1)",
        "Box Width (px)", "Box Height (px)",
        "Aspect Ratio (W / H)", "BBox Area (px²)",
    ]
    colors = ["#4C78A8", "#4C78A8", "#54A24B", "#54A24B", "#F2CF5B", "#E45756"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
        series = numeric[metric]
        # Trim extreme 1% outliers for cleaner visualization of skewed distributions
        q99 = series.quantile(0.99)
        trimmed = series[series <= q99]
        ax.hist(trimmed, bins=35, color=color, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.suptitle(f"CelebA Bounding Box Distributions (n={len(numeric):,})", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ==============================================================================
# Attribute / Evaluation Labels EDA
# ==============================================================================

def load_attributes(path: str | Path) -> pd.DataFrame:
    """Load CelebA facial attribute labels from CSV or original TXT format.

    Attributes in CelebA are binary flags represented as 1 (positive) or -1 (negative).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Attribute file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
    elif path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8-sig") as stream:
            count_line = stream.readline().strip()
            header = stream.readline().split()
        try:
            expected_count = int(count_line)
        except ValueError as exc:
            raise ValueError("The first line of an attribute TXT file must be the image count.") from exc

        df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, encoding="utf-8-sig")
        if df.shape[1] == len(header) + 1:
            df.columns = ["image_id", *header]
        elif df.shape[1] == len(header):
            df.columns = header
        else:
            raise ValueError("Mismatch between declared header columns and data columns in TXT file.")

        if len(df) != expected_count:
            raise ValueError(f"TXT declares {expected_count} images, but {len(df)} data rows were read.")
    else:
        raise ValueError("Attribute files must have a .csv or .txt extension.")

    if "image_id" not in df.columns:
        raise ValueError("Attribute table must contain an image_id column.")
    if len(df.columns) <= 1:
        raise ValueError("Attribute table contains no attribute columns.")

    df["image_id"] = _image_ids(df)
    return df


# Alias for load_attributes matching eval terminology
load_eval_labels = load_attributes


def check_attributes(
    df: pd.DataFrame,
    partitions: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Validate facial attribute metadata.

    Counts:
    - Missing or duplicate image IDs
    - Missing attribute entries
    - Non-binary attribute values (values not in {-1, 1})
    - Overlap/consistency with partition table if provided
    """
    if "image_id" not in df.columns:
        raise ValueError("Attribute table must contain an image_id column.")

    attr_cols = [c for c in df.columns if c != "image_id"]
    if not attr_cols:
        raise ValueError("Attribute table has no attribute columns.")

    ids = _image_ids(df)
    raw_attrs = df[attr_cols].astype("string").apply(lambda col: col.str.strip())
    missing = raw_attrs.isna() | raw_attrs.eq("").fillna(False)

    numeric = df[attr_cols].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna() & ~missing
    non_binary = (~missing) & (~numeric.isin([-1, 1]))

    result = {
        "total_rows": len(df),
        "total_attributes": len(attr_cols),
        "missing_image_ids": int(ids.isna().sum()),
        "duplicate_image_ids": int(ids.dropna().duplicated().sum()),
        "missing_attribute_values": int(missing.to_numpy().sum()),
        "invalid_numeric_values": int(invalid_numeric.to_numpy().sum()),
        "non_binary_values": int(non_binary.to_numpy().sum()),
        "rows_with_attribute_issues": int(
            (missing.any(axis=1) | invalid_numeric.any(axis=1) | non_binary.any(axis=1)).sum()
        ),
    }

    if partitions is not None:
        if not {"image_id", "partition"}.issubset(partitions.columns):
            raise ValueError("The partition table must contain image_id and partition columns.")
        attr_ids = set(ids.dropna())
        partition_ids = set(_image_ids(partitions).dropna())
        result["attr_ids_without_partition"] = len(attr_ids - partition_ids)
        result["partition_ids_without_attr"] = len(partition_ids - attr_ids)

    return result


check_eval_labels = check_attributes


def summarize_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute label statistics and class balance for all attributes.

    Returns positive count, negative count, positive percentage,
    negative percentage, and imbalance ratio (majority / minority class).
    """
    if "image_id" not in df.columns:
        raise ValueError("Attribute table must contain an image_id column.")

    attr_cols = [c for c in df.columns if c != "image_id"]
    if not attr_cols:
        raise ValueError("Attribute table has no attribute columns.")

    numeric = df[attr_cols].apply(pd.to_numeric, errors="coerce")

    pos_counts = (numeric == 1).sum(axis=0)
    neg_counts = (numeric == -1).sum(axis=0)
    valid_counts = pos_counts + neg_counts

    pos_pct = np.where(valid_counts > 0, (pos_counts / valid_counts) * 100, 0.0)
    neg_pct = np.where(valid_counts > 0, (neg_counts / valid_counts) * 100, 0.0)

    # Imbalance ratio: majority count / minority count
    min_count = np.minimum(pos_counts, neg_counts)
    max_count = np.maximum(pos_counts, neg_counts)
    imbalance_ratio = np.where(min_count > 0, max_count / min_count, np.nan)

    summary = pd.DataFrame(
        {
            "positive_count": pos_counts,
            "negative_count": neg_counts,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "imbalance_ratio": imbalance_ratio,
        },
        index=attr_cols,
    )
    summary.index.name = "attribute"
    return summary.sort_values(by="positive_pct", ascending=True)


summarize_eval_labels = summarize_attributes


def summarize_attribute_partitions(
    df: pd.DataFrame,
    partitions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare positive attribute percentage across train, val, and test splits.

    Detects potential dataset shift or class distribution anomalies between partitions.
    """
    if not {"image_id", "partition"}.issubset(partitions.columns):
        raise ValueError("partitions must contain image_id and partition.")
    if "image_id" not in df.columns:
        raise ValueError("Attribute dataframe must contain image_id.")

    merged = pd.merge(df, partitions[["image_id", "partition"]], on="image_id", how="inner")
    attr_cols = [c for c in df.columns if c != "image_id"]

    partition_map = {0: "train_pos_pct", 1: "val_pos_pct", 2: "test_pos_pct"}
    split_summaries = {}

    for code, split_name in partition_map.items():
        subset = merged[merged["partition"] == code]
        if len(subset) == 0:
            split_summaries[split_name] = pd.Series(0.0, index=attr_cols)
        else:
            num = subset[attr_cols].apply(pd.to_numeric, errors="coerce")
            pos = (num == 1).sum(axis=0)
            total = (num.isin([1, -1])).sum(axis=0)
            split_summaries[split_name] = (pos / total) * 100

    result = pd.DataFrame(split_summaries, index=attr_cols)
    result["max_split_diff"] = result.max(axis=1) - result.min(axis=1)
    result.index.name = "attribute"
    return result.sort_values(by="max_split_diff", ascending=False)


def plot_attribute_imbalance(
    df: pd.DataFrame,
    top_n: int = 40,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot horizontal bar chart of attribute positive percentages.

    Demonstrates class imbalance across attributes, with a reference line at 50%
    (perfect class balance).
    """
    summary = summarize_attributes(df)
    if top_n and top_n < len(summary):
        # Pick extreme tails: most rare and most common
        half = top_n // 2
        summary = pd.concat([summary.iloc[:half], summary.iloc[-half:]])

    fig, ax = plt.subplots(figsize=(10, max(6, len(summary) * 0.28)))
    y_pos = np.arange(len(summary))

    bars = ax.barh(
        y_pos,
        summary["positive_pct"],
        color=np.where(summary["positive_pct"] < 20, "#E45756", "#4C78A8"),
        edgecolor="none",
        height=0.7,
    )

    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Balanced (50%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index, fontsize=9)
    ax.set_xlabel("Positive Frequency (%)")
    ax.set_title("CelebA Attribute Class Distribution (% Positive Labels)", fontsize=12)
    ax.set_xlim(0, 100)

    for bar, pct in zip(bars, summary["positive_pct"]):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_attribute_correlations(
    df: pd.DataFrame,
    attributes: list[str] | None = None,
    top_k: int = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compute and plot correlation matrix between attributes as a heatmap.

    If attributes is None, automatically selects key representative attributes.
    """
    attr_cols = [c for c in df.columns if c != "image_id"]
    if attributes is None:
        # Default representative subset of CelebA attributes
        defaults = [
            "Male", "No_Beard", "Wearing_Lipstick", "Heavy_Makeup",
            "Smiling", "Young", "Attractive", "Eyeglasses", "Bald",
            "Gray_Hair", "Mustache", "Goatee", "Sideburns", "Bangs", "Black_Hair",
        ]
        attributes = [a for a in defaults if a in attr_cols][:top_k]

    missing = [a for a in attributes if a not in attr_cols]
    if missing:
        raise ValueError(f"Attributes not in dataset: {', '.join(missing)}")

    numeric = df[attributes].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(attributes) * 0.7), max(7, len(attributes) * 0.65)))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(attributes)))
    ax.set_yticks(np.arange(len(attributes)))
    ax.set_xticklabels(attributes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(attributes, fontsize=9)

    for i in range(len(attributes)):
        for j in range(len(attributes)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title("CelebA Attribute Pearson Correlation Matrix", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_sample_attributes(
    df: pd.DataFrame,
    image_dir: str | Path,
    n_images: int = 8,
    random_state: int = 42,
    max_attrs_to_show: int = 6,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Display sample images along with their prominent positive attributes."""
    if "image_id" not in df.columns:
        raise ValueError("Attribute table must contain image_id.")

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    attr_cols = [c for c in df.columns if c != "image_id"]
    ids = _image_ids(df)

    eligible = df[ids.notna()].copy()
    sample = eligible.sample(n=min(n_images, len(eligible)), random_state=random_state)

    images = []
    for _, row in sample.iterrows():
        image_id = str(row["image_id"])
        path = image_dir / image_id
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as img:
            images.append(img.convert("RGB"))

    ncols = min(4, len(sample))
    nrows = math.ceil(len(sample) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.2 * nrows), squeeze=False)

    for ax, (_, row), img in zip(axes.flat, sample.iterrows(), images):
        ax.imshow(img)
        pos_attrs = [c for c in attr_cols if row[c] == 1]
        if len(pos_attrs) > max_attrs_to_show:
            displayed_attrs = pos_attrs[:max_attrs_to_show] + [f"(+{len(pos_attrs) - max_attrs_to_show} more)"]
        else:
            displayed_attrs = pos_attrs

        attr_text = "\n".join(displayed_attrs) if displayed_attrs else "None"
        ax.set_title(f"{row['image_id']}\n{attr_text}", fontsize=8)
        ax.axis("off")

    for ax in list(axes.flat)[len(sample):]:
        ax.axis("off")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
