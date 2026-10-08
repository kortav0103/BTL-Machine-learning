"""Exploratory data analysis for the five CelebA facial landmarks.

Reads CSV files or original TXT files containing an image count,
coordinate headers, and data rows. Dependencies: pandas, matplotlib,
and Pillow. Plot images at their original size using matching annotations.
"""

import math
from pathlib import Path

import pandas as pd


LANDMARK_NAMES = ("lefteye", "righteye", "nose", "leftmouth", "rightmouth")
COORDINATE_COLUMNS = [
    f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")
]
REQUIRED_COLUMNS = ["image_id", *COORDINATE_COLUMNS]
LANDMARK_COLORS = ("#E45756", "#4C78A8", "#F2CF5B", "#54A24B", "#B279A2")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Landmark table is missing columns: {', '.join(missing)}")


def _image_ids(df: pd.DataFrame) -> pd.Series:
    return df["image_id"].astype("string").str.strip().replace("", pd.NA)


def _numeric_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert coordinates to numbers and exclude non-finite values from statistics."""
    numeric = df[COORDINATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([float("inf"), float("-inf")], float("nan"))


def _save_figure(fig, save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def load_landmarks(path: str | Path) -> pd.DataFrame:
    """Load a CSV with headers or a TXT in the original CelebA format.

    In TXT files, line 1 is the image count, line 2 lists the 10 coordinate
    columns without ``image_id``, and each subsequent row contains an
    image filename and 10 values. CSV files contain ``image_id`` and the
    10 coordinate columns. Unusual values are preserved for validation
    by ``check_landmarks``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Landmark file not found: {path}")

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
            raise ValueError("The first line of a landmark TXT file must be the image count.") from exc
        if len(header) != 10 or set(header) != set(COORDINATE_COLUMNS):
            raise ValueError("The second line must contain all 10 CelebA coordinate column names.")
        df = pd.read_csv(
            path, sep=r"\s+", skiprows=2, header=None, encoding="utf-8-sig"
        )
        if df.shape[1] != 11:
            raise ValueError("Each TXT data row must contain an image filename and 10 coordinates.")
        df.columns = ["image_id", *header]
        if len(df) != expected_count:
            raise ValueError(
                f"TXT declares {expected_count} images, but {len(df)} data rows were read."
            )
    else:
        raise ValueError("Landmark files must have a .csv or .txt extension.")

    _validate_columns(df)
    if df.empty:
        raise ValueError("The landmark table contains no data rows.")
    df = df[REQUIRED_COLUMNS].copy()
    df["image_id"] = _image_ids(df)
    return df


def check_landmarks(
    df: pd.DataFrame,
    partitions: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Validate metadata without modifying or removing input rows.

    Missing coordinate values and values that are not finite numbers
    are counted separately. Negative coordinates are flagged for review,
    without assuming they are errors. ``duplicate_image_ids`` counts
    extra rows after the first occurrence, excluding missing filenames.
    When partitions are provided, compare unique filenames in both tables.
    Upper coordinate bounds require the actual image dimensions.
    """
    _validate_columns(df)
    ids = _image_ids(df)
    raw = df[COORDINATE_COLUMNS].astype("string").apply(lambda col: col.str.strip())
    missing = raw.isna() | raw.eq("").fillna(False)
    numeric = _numeric_coordinates(df)
    invalid = numeric.isna() & ~missing

    result = {
        "total_rows": len(df),
        "missing_image_ids": int(ids.isna().sum()),
        "duplicate_image_ids": int(ids.dropna().duplicated().sum()),
        "missing_coordinate_values": int(missing.to_numpy().sum()),
        "invalid_coordinate_values": int(invalid.to_numpy().sum()),
        "rows_with_coordinate_issues": int((missing | invalid).any(axis=1).sum()),
        "negative_coordinate_values": int(numeric.lt(0).to_numpy().sum()),
    }
    if partitions is not None:
        if not {"image_id", "partition"}.issubset(partitions.columns):
            raise ValueError("The partition table must contain image_id and partition columns.")
        landmark_ids = set(ids.dropna())
        partition_ids = set(_image_ids(partitions).dropna())
        result["landmark_ids_without_partition"] = len(landmark_ids - partition_ids)
        result["partition_ids_without_landmarks"] = len(partition_ids - landmark_ids)
    return result


def summarize_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize count, mean, std, min, quantiles, and max for all 10 columns.

    Only finite numeric values in each column are included. Interpret
    these statistics alongside ``check_landmarks`` to account for missing
    or invalid values.
    """
    _validate_columns(df)
    summary = _numeric_coordinates(df).describe().T
    summary.index.name = "coordinate"
    return summary


def plot_landmarks_on_images(
    df: pd.DataFrame,
    image_dir: str | Path,
    n_images: int = 12,
    random_state: int = 42,
    save_path: str | Path | None = None,
):
    """Sample images, overlay five points at the original size, and return a Figure.

    For training-set EDA, pass a table filtered by training filenames.
    Only rows with a filename and all 10 finite coordinates are eligible.
    Validate the table first to identify rows excluded from the plot.
    Resolve duplicate filenames before calling this function. Image files
    must be located directly inside ``image_dir``.

    Out-of-bounds points are counted in each title and may indicate
    mismatched annotation versions. This function does not crop or resize
    images or adjust coordinates.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    _validate_columns(df)
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not isinstance(n_images, int) or n_images < 1:
        raise ValueError("n_images must be a positive integer.")
    ids = _image_ids(df)
    if ids.dropna().duplicated().any():
        raise ValueError("The landmark table contains duplicate filenames; validate it first.")

    numeric = _numeric_coordinates(df)
    eligible = numeric.notna().all(axis=1) & ids.notna()
    candidates = numeric.loc[eligible].copy()
    candidates.insert(0, "image_id", ids.loc[eligible])
    if candidates.empty:
        raise ValueError("No rows have both an image filename and 10 valid coordinates to plot.")
    sample = candidates.sample(n=min(n_images, len(candidates)), random_state=random_state)

    # Read images before creating a Figure to avoid leaving a partial plot on failure.
    images = []
    for _, row in sample.iterrows():
        image_id = str(row["image_id"])
        if Path(image_id).name != image_id:
            raise ValueError(f"image_id must be an image filename without a directory path: {image_id}")
        path = image_dir / image_id
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    ncols = min(4, len(sample))
    nrows = math.ceil(len(sample) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    handles = []
    for ax, (_, row), image in zip(axes.flat, sample.iterrows(), images):
        width, height = image.size
        ax.imshow(image)
        outside = 0
        handles = []
        for name, color in zip(LANDMARK_NAMES, LANDMARK_COLORS):
            x, y = row[f"{name}_x"], row[f"{name}_y"]
            outside += int(not (0 <= x < width and 0 <= y < height))
            handles.append(ax.scatter(x, y, color=color, s=35, edgecolors="black", linewidths=0.4))
        title = str(row["image_id"])
        if outside:
            title += f"\nOutside image: {outside}/5 points"
        ax.set_title(title)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.axis("off")
    for ax in list(axes.flat)[len(sample):]:
        ax.axis("off")
    fig.legend(handles, LANDMARK_NAMES, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save_figure(fig, save_path)
    return fig


def plot_landmark_distribution(
    df: pd.DataFrame,
    landmark: str = "nose",
    bins: int = 40,
    save_path: str | Path | None = None,
):
    """Plot x and y histograms for one landmark and return a Figure.

    Only rows with finite x and y values are included. For training-set
    EDA, pass a filtered training table. Coordinates are measured in pixels;
    direct comparisons require matching image sizes and coordinate systems.
    """
    import matplotlib.pyplot as plt

    _validate_columns(df)
    if landmark not in LANDMARK_NAMES:
        raise ValueError(f"landmark must be one of: {', '.join(LANDMARK_NAMES)}")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    coordinates = _numeric_coordinates(df)[[f"{landmark}_x", f"{landmark}_y"]].dropna()
    if coordinates.empty:
        raise ValueError(f"No valid coordinate pairs are available for {landmark}.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, axis, color in zip(axes, ("x", "y"), LANDMARK_COLORS):
        ax.hist(coordinates[f"{landmark}_{axis}"], bins=bins, color=color, edgecolor="white")
        ax.set_title(f"{landmark}: {axis} distribution (n={len(coordinates):,})")
        ax.set_xlabel(f"{axis} coordinate (pixels)")
        ax.set_ylabel("Number of metadata rows")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
