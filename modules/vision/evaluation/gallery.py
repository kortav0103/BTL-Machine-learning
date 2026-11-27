"""Prediction gallery: example test faces per outcome for the best run of each target.

Exporting (on the machine that has the images) copies small thumbnails into
``results/gallery/<target>/`` with a ``gallery.csv`` index, so the gallery can be
re-rendered anywhere from ``results/`` alone.

Rows are true positives, true negatives, false positives and false negatives.
Correct predictions are sampled at random; errors are the most confident ones
(highest score among false positives, lowest among false negatives).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from modules.vision.plotting.style import apply_style

GALLERY_DIR = "gallery"
INDEX_FILE = "gallery.csv"
THUMB_SIZE = (89, 109)  # half of the aligned 178x218 crop
OUTCOMES = ("TP", "TN", "FP", "FN")
OUTCOME_LABELS = {"TP": "true pos", "TN": "true neg", "FP": "false pos", "FN": "false neg"}


def _outcome(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    labels = np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 0), (y_true == 0) & (y_pred == 1)],
        ["TP", "TN", "FP"],
        default="FN",
    )
    return pd.Series(labels, index=y_true.index)


def select_examples(predictions: pd.DataFrame, n_per_outcome: int = 6, seed: int = 42) -> pd.DataFrame:
    """Pick up to ``n_per_outcome`` test predictions for each outcome."""
    preds = predictions.assign(outcome=_outcome(predictions["y_true"], predictions["y_pred"]))
    picks = []
    for outcome in OUTCOMES:
        group = preds[preds["outcome"] == outcome]
        if group.empty:
            continue
        n = min(n_per_outcome, len(group))
        if outcome == "FP":
            group = group.sort_values("score", ascending=False).head(n)
        elif outcome == "FN":
            group = group.sort_values("score", ascending=True).head(n)
        else:
            group = group.sample(n=n, random_state=seed)
        picks.append(group)
    return pd.concat(picks, ignore_index=True) if picks else preds.head(0)


def _best_runs(results_dir: Path) -> list[dict]:
    from modules.vision.evaluation.comparison import best_runs, collect_runs

    table = collect_runs(results_dir)
    return [] if table.empty else best_runs(table).to_dict("records")


def export_gallery(
    results_dir: str | Path,
    manifest: pd.DataFrame,
    n_per_outcome: int = 6,
    seed: int = 42,
) -> list[Path]:
    """Copy thumbnails of selected test images for each target's best run.

    Returns the written ``gallery.csv`` paths.
    """
    results_dir = Path(results_dir)
    paths = manifest.set_index("image_id")["image_path"]
    written = []
    for run in _best_runs(results_dir):
        run_dir = results_dir / run["target"] / run["extractor"] / run["classifier"]
        examples = select_examples(pd.read_csv(run_dir / "predictions.csv"), n_per_outcome, seed)
        out_dir = results_dir / GALLERY_DIR / run["target"]
        out_dir.mkdir(parents=True, exist_ok=True)
        thumbs = []
        for row in examples.itertuples(index=False):
            thumb = f"{row.outcome}_{Path(row.image_id).stem}.jpg"
            with Image.open(paths[row.image_id]) as image:
                image.convert("RGB").resize(THUMB_SIZE).save(out_dir / thumb, quality=90)
            thumbs.append(thumb)
        index = examples.assign(
            thumbnail=thumbs, target=run["target"], extractor=run["extractor"], classifier=run["classifier"]
        )
        index.to_csv(out_dir / INDEX_FILE, index=False)
        written.append(out_dir / INDEX_FILE)
    return written


def plot_gallery(gallery_dir: str | Path, max_cols: int = 6) -> plt.Figure:
    """Grid of thumbnails for one target: one row per outcome, score under each face."""
    from modules.vision.evaluation.comparison import SHORT_EXTRACTORS, SHORT_NAMES

    gallery_dir = Path(gallery_dir)
    index = pd.read_csv(gallery_dir / INDEX_FILE)
    style = apply_style("heatmap")
    tick = style["typography"]["tick_label_size"]
    rows = [o for o in OUTCOMES if (index["outcome"] == o).any()]
    width = style["figure"]["two_column_width"]
    height = width / max_cols * (THUMB_SIZE[1] / THUMB_SIZE[0]) * len(rows) * 1.15 + 0.5
    fig, axes = plt.subplots(len(rows), max_cols, figsize=(width, height), squeeze=False)

    for r, outcome in enumerate(rows):
        group = index[index["outcome"] == outcome].head(max_cols)
        for c, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(group):
                row = group.iloc[c]
                with Image.open(gallery_dir / row["thumbnail"]) as image:
                    ax.imshow(image)
                ax.set_xlabel(f"{row['score']:+.2f}", fontsize=tick, labelpad=1)
        axes[r][0].set_ylabel(OUTCOME_LABELS[outcome], fontsize=style["typography"]["axis_label_size"])

    first = index.iloc[0]
    fig.suptitle(
        f"{first['target'].replace('_', ' ')}: "
        f"{SHORT_EXTRACTORS.get(first['extractor'], first['extractor'])} + "
        f"{SHORT_NAMES.get(first['classifier'], first['classifier'])}  (score under each face)",
        fontsize=style["typography"]["title_size"],
    )
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    return fig
