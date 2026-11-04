"""Exploratory data analysis for CelebA train/validation/test partitions.

Supports CSV files with ``image_id`` and ``partition`` columns and
original TXT files without a header. Codes: 0 = train, 1 = validation, 2 = test.

Dependencies: pandas and matplotlib (required only for plotting).
"""

from pathlib import Path

import pandas as pd


PARTITION_NAMES = {0: "train", 1: "validation", 2: "test"}
REQUIRED_COLUMNS = ["image_id", "partition"]


def _validate_columns(df: pd.DataFrame) -> None:
    """Check that the table contains both required columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Partition table is missing columns: {', '.join(missing)}")


def _image_ids(df: pd.DataFrame) -> pd.Series:
    """Strip whitespace to identify missing or duplicate image filenames."""
    return df["image_id"].astype("string").str.strip().replace("", pd.NA)


def _partition_codes(df: pd.DataFrame) -> pd.Series:
    """Convert partition codes to numbers; unparseable values become NaN."""
    return pd.to_numeric(df["partition"], errors="coerce")


def load_partitions(path: str | Path) -> pd.DataFrame:
    """Load a partition table from a CSV or TXT file.

    CSV files must have an ``image_id,partition`` header. Original TXT
    files contain rows such as ``000001.jpg 0`` without a header.
    Unusual codes are preserved for ``check_partitions`` to report;
    this function does not automatically remove rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Partition file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
    elif path.suffix.lower() == ".txt":
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            encoding="utf-8-sig",
        )
        if df.shape[1] != 2:
            raise ValueError("Partition TXT files must have exactly 2 columns and no header.")
        df.columns = REQUIRED_COLUMNS
    else:
        raise ValueError("Partition files must have a .csv or .txt extension.")

    _validate_columns(df)
    if df.empty:
        raise ValueError("The partition table contains no data rows.")

    df = df[REQUIRED_COLUMNS].copy()
    df["image_id"] = _image_ids(df)
    return df


def check_partitions(df: pd.DataFrame) -> dict[str, int]:
    """Count metadata issues without modifying the input data.

    ``duplicate_image_ids`` counts extra rows after the first occurrence,
    excluding missing filenames. Overlap metrics count filenames that
    appear in both splits. These checks compare filenames only, not
    image contents or the identities of the people in the images.
    """
    _validate_columns(df)
    ids = _image_ids(df)
    codes = _partition_codes(df)
    raw_codes = df["partition"].astype("string").str.strip()
    missing_codes = raw_codes.isna() | raw_codes.eq("").fillna(False)
    invalid_codes = ~missing_codes & ~codes.isin(PARTITION_NAMES)

    split_ids = {
        name: set(ids[codes.eq(code)].dropna())
        for code, name in PARTITION_NAMES.items()
    }

    return {
        "total_rows": len(df),
        "missing_image_ids": int(ids.isna().sum()),
        "missing_partitions": int(missing_codes.sum()),
        "invalid_partitions": int(invalid_codes.sum()),
        "duplicate_image_ids": int(ids.dropna().duplicated().sum()),
        "train_validation_overlap": len(split_ids["train"] & split_ids["validation"]),
        "train_test_overlap": len(split_ids["train"] & split_ids["test"]),
        "validation_test_overlap": len(split_ids["validation"] & split_ids["test"]),
    }


def summarize_partitions(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``count`` and ``percent`` for all three splits, including empty ones.

    Each metadata row is counted once; duplicates and missing filenames
    are not removed. Percentages use the total number of rows as the
    denominator. Missing or invalid codes cause the percentages to sum
    to less than 100%. Interpret this table alongside the validation results.
    """
    _validate_columns(df)
    codes = _partition_codes(df)
    counts = codes.value_counts().reindex(list(PARTITION_NAMES), fill_value=0)
    summary = pd.DataFrame({"count": counts.astype("int64")})
    summary["percent"] = counts / len(df) * 100 if len(df) else 0.0
    summary.index = pd.Index(list(PARTITION_NAMES.values()), name="split")
    return summary


def plot_partition_counts(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    ax=None,
):
    """Plot the number of metadata rows per split and return Matplotlib Axes.

    Pass ``save_path`` to save a PNG. In a notebook, call ``plt.show()``
    after this function to display the chart.
    """
    import matplotlib.pyplot as plt

    summary = summarize_partitions(df)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        summary.index,
        summary["count"],
        color=["#4C78A8", "#F2CF5B", "#E45756"],
    )
    ax.set_title("CelebA: Train / Validation / Test")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of metadata rows")
    ax.bar_label(bars, labels=[f"{count:,}" for count in summary["count"]], padding=3)
    ax.margins(y=0.15)
    ax.figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
