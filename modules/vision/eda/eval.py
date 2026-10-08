"""CelebA attribute label EDA. Input: CSV with image_id plus one 1/-1 column per attribute.

Dependencies: pandas, numpy, matplotlib, Pillow.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

VALID_LABELS = (-1, 1)
PARTITION_COLUMNS = {0: "train_pos_pct", 1: "val_pos_pct", 2: "test_pos_pct"}
RARE_PCT = 20  # attributes with a lower positive rate are drawn red
CORRELATION_DEFAULTS = [
    "Male", "No_Beard", "Wearing_Lipstick", "Heavy_Makeup", "Smiling", "Young", "Attractive",
    "Eyeglasses", "Bald", "Gray_Hair", "Mustache", "Goatee", "Sideburns", "Bangs", "Black_Hair",
]


def _save_figure(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _image_ids(df):
    """image_id values with whitespace trimmed; blanks become <NA>."""
    return df["image_id"].astype("string").str.strip().replace("", pd.NA)


def _attribute_columns(df):
    """Validate the layout and return the attribute column names."""
    attrs = [c for c in df.columns if c != "image_id"]
    if "image_id" not in df.columns or not attrs:
        raise ValueError("Table needs an image_id column and at least one attribute column.")
    return attrs


def _labels(df, cols):
    """Attribute columns as numbers; unparseable entries become NaN."""
    return df[cols].apply(pd.to_numeric, errors="coerce")


def _load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def load_attributes(path: str | Path) -> pd.DataFrame:
    """Load attribute labels from a CSV; odd values are kept for check_attributes to report."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Attribute file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    _attribute_columns(df)
    if df.empty:
        raise ValueError("The attribute table contains no data rows.")
    return df.assign(image_id=_image_ids(df))


def check_attributes(df: pd.DataFrame, partitions: pd.DataFrame | None = None) -> dict[str, int]:
    """Count label problems without altering rows.

    Each bad cell lands in exactly one count: missing (blank), invalid_numeric (not a
    number) or non_binary (a number other than -1/1). With ``partitions``, also counts
    image IDs that appear in only one of the two tables.
    """
    cols = _attribute_columns(df)
    ids = _image_ids(df)
    labels = _labels(df, cols)
    text = df[cols].astype("string").apply(lambda c: c.str.strip())
    missing = text.fillna("").eq("").astype(bool)
    invalid = labels.isna() & ~missing
    non_binary = labels.notna() & ~labels.isin(VALID_LABELS)

    result = {
        "total_rows": len(df),
        "total_attributes": len(cols),
        "missing_image_ids": int(ids.isna().sum()),
        "duplicate_image_ids": int(ids.dropna().duplicated().sum()),
        "missing_attribute_values": int(missing.to_numpy().sum()),
        "invalid_numeric_values": int(invalid.to_numpy().sum()),
        "non_binary_values": int(non_binary.to_numpy().sum()),
        "rows_with_attribute_issues": int((missing | invalid | non_binary).any(axis=1).sum()),
    }
    if partitions is not None:
        attr_ids, part_ids = set(ids.dropna()), set(_image_ids(partitions).dropna())
        result["attr_ids_without_partition"] = len(attr_ids - part_ids)
        result["partition_ids_without_attr"] = len(part_ids - attr_ids)
    return result


def summarize_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-attribute counts, positive/negative % and imbalance ratio (majority / minority), rarest first.

    Percentages are NaN with no valid labels; the ratio is NaN when a class is absent.
    """
    labels = _labels(df, _attribute_columns(df))
    pos, neg = labels.eq(1).sum(), labels.eq(-1).sum()
    valid = (pos + neg).where(lambda s: s > 0)  # NaN instead of dividing by zero
    summary = pd.DataFrame({
        "positive_count": pos,
        "negative_count": neg,
        "positive_pct": pos / valid * 100,
        "negative_pct": neg / valid * 100,
        "imbalance_ratio": np.maximum(pos, neg) / np.minimum(pos, neg).where(lambda s: s > 0),
    })
    summary.index.name = "attribute"
    return summary.sort_values("positive_pct")


def summarize_attribute_partitions(df: pd.DataFrame, partitions: pd.DataFrame) -> pd.DataFrame:
    """Positive rate (%) per attribute in each split; a large ``max_split_diff`` hints at shift.

    Splits with no valid labels are NaN, not 0%. Raises MergeError if an image is repeated in ``partitions``.
    """
    cols = _attribute_columns(df)
    splits = pd.DataFrame({
        "image_id": _image_ids(partitions),
        "partition": pd.to_numeric(partitions["partition"], errors="coerce"),
    })
    merged = df.assign(image_id=_image_ids(df)).merge(splits, on="image_id", validate="many_to_one")
    labels = _labels(merged, cols)

    rates = {}
    for code, name in PARTITION_COLUMNS.items():
        part = labels[merged["partition"].eq(code)]
        rates[name] = part.eq(1).sum() / part.isin(VALID_LABELS).sum().where(lambda s: s > 0) * 100
    result = pd.DataFrame(rates)
    result["max_split_diff"] = result.max(axis=1) - result.min(axis=1)
    result.index.name = "attribute"
    return result.sort_values("max_split_diff", ascending=False)


def plot_attribute_imbalance(
    df: pd.DataFrame, top_n: int | None = 40, save_path: str | Path | None = None
) -> plt.Figure:
    """Bar chart of positive-label % per attribute (dashed line = balanced, red = rare).

    If ``top_n`` is below the attribute count, shows the rarest and most common half each.
    """
    summary = summarize_attributes(df).dropna(subset=["positive_pct"])
    if top_n and top_n < len(summary):
        summary = pd.concat([summary.head(top_n // 2), summary.tail(top_n - top_n // 2)])
    pct = summary["positive_pct"]

    fig, ax = plt.subplots(figsize=(10, max(6, len(summary) * 0.28)))
    bars = ax.barh(summary.index, pct, height=0.7,
                   color=["#E45756" if p < RARE_PCT else "#4C78A8" for p in pct])
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pct], padding=3, fontsize=8)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.7, label="Balanced (50%)")
    ax.set(xlim=(0, 100), xlabel="Positive labels (%)", title="CelebA attribute class balance")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_attribute_correlations(
    df: pd.DataFrame, attributes: list[str] | None = None, save_path: str | Path | None = None
) -> plt.Figure:
    """Heatmap of pairwise correlations (phi, for 1/-1 labels); defaults to CORRELATION_DEFAULTS."""
    cols = _attribute_columns(df)
    attributes = attributes or [a for a in CORRELATION_DEFAULTS if a in cols]
    unknown = [a for a in attributes if a not in cols]
    if unknown:
        raise ValueError(f"Attributes not in dataset: {', '.join(unknown)}")

    corr = _labels(df, attributes).corr()
    n = len(attributes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.65)))
    fig.colorbar(ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1), ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n), attributes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), attributes, fontsize=9)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.45 else "black")  # white stays readable on dark cells
    ax.set_title("CelebA attribute correlations", fontsize=12)
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
    """Random images titled with their positive attributes (any beyond the limit shown as a count)."""
    cols = _attribute_columns(df)
    ids = _image_ids(df)
    eligible = df.loc[ids.notna()].assign(image_id=ids.dropna())
    if n_images < 1 or eligible.empty:
        raise ValueError("Need n_images >= 1 and at least one row with an image_id.")
    sample = eligible.sample(n=min(n_images, len(eligible)), random_state=random_state)
    positive = _labels(sample, cols).eq(1).to_numpy()
    # Load every image first so a missing file doesn't leave a half-drawn figure.
    images = [_load_rgb(Path(image_dir) / i) for i in sample["image_id"]]

    ncols = min(4, len(sample))
    nrows = math.ceil(len(sample) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image_id, image, flags in zip(axes.flat, sample["image_id"], images, positive):
        names = [c for c, flag in zip(cols, flags) if flag]
        extra = len(names) - max_attrs_to_show
        shown = names[:max_attrs_to_show] + ([f"(+{extra} more)"] if extra > 0 else [])
        ax.imshow(image)
        ax.set_title(f"{image_id}\n" + ("\n".join(shown) or "None"), fontsize=8)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# Aliases for callers that use the "eval labels" naming.
load_eval_labels = load_attributes
check_eval_labels = check_attributes
summarize_eval_labels = summarize_attributes