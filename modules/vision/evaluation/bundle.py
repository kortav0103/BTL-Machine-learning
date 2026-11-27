"""Portable experiment artifacts for runs with access to the dataset."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.vision.eda.images import plot_image_distributions, scan_images, summarize_images
from modules.vision.evaluation.comparison import collect_runs


def export_eda(results: Path, manifest: pd.DataFrame, targets: list[str], seed: int) -> None:
    """Describe the run's splits and up to 500 deterministically sampled training images."""
    out = results / "eda"
    out.mkdir()
    counts = (
        manifest.melt(id_vars="partition", value_vars=targets, var_name="target", value_name="label")
        .groupby(["partition", "target", "label"])
        .size()
        .unstack("label", fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: "negative", 1: "positive"})
    )
    counts.to_csv(out / "class_counts.csv")
    training = manifest.loc[manifest["partition"] == 0]
    image_dir = Path(training["image_path"].iloc[0]).parent
    scanned = scan_images(image_dir, training["image_id"], max_images=500, random_state=seed)
    scanned.to_csv(out / "training_image_sample.csv", index=False)
    summarize_images(scanned).to_csv(out / "image_summary.csv")
    scanned["mode"].value_counts(dropna=False).rename("count").to_csv(out / "color_modes.csv")
    plt.close(plot_image_distributions(scanned, save_path=out / "image_distributions.png"))


def export_features(
    results: Path, extractor: str, features: np.ndarray, timing: dict[str, float]
) -> np.ndarray:
    """Save a matrix aligned to the bundle manifest and reopen it for training."""
    out = results / "features"
    out.mkdir(exist_ok=True)
    path = out / f"{extractor}.npy"
    np.save(path, features, allow_pickle=False)
    metadata = {
        "extractor": extractor,
        "rows": len(features),
        "dimensions": features.shape[1],
        "dtype": str(features.dtype),
        "file_bytes": path.stat().st_size,
        "extraction_seconds": timing.get("extraction_seconds"),
        "row_metadata": "../manifest.csv",
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def export_comparison_crops(results: Path, manifest: pd.DataFrame) -> None:
    """Save identical test crops across configurations, favoring informative disagreements."""
    paths = manifest.set_index("image_id")["image_path"]
    learned = collect_runs(results).query("classifier != 'majority'")
    for target, runs in learned.groupby("target"):
        out = results / "comparison_crops" / target
        out.mkdir(parents=True)
        records = []
        for run in runs.itertuples(index=False):
            predictions = pd.read_csv(results / target / run.extractor / run.classifier / "predictions.csv")
            records.append(predictions.assign(extractor=run.extractor, classifier=run.classifier))
        predictions = pd.concat(records, ignore_index=True)
        predictions["correct"] = predictions["y_true"] == predictions["y_pred"]
        correct = predictions.pivot(
            index="image_id", columns=["extractor", "classifier"], values="correct"
        )
        if correct.isna().any().any():
            raise ValueError(f"Test image IDs differ across configurations for {target}")

        by_extractor = correct.T.groupby(level="extractor").any().T
        by_classifier = correct.T.groupby(level="classifier").any().T
        categories = {
            "backbone_difference": by_extractor.any(axis=1) & ~by_extractor.all(axis=1),
            "classifier_difference": by_classifier.any(axis=1) & ~by_classifier.all(axis=1),
            "all_wrong": ~correct.any(axis=1),
            "all_correct": correct.all(axis=1),
            "disagreement": correct.any(axis=1) & ~correct.all(axis=1),
        }
        selected: dict[str, str] = {}
        for category, mask in categories.items():
            candidates = correct.index[mask].difference(selected)
            for image_id in candidates[:6]:
                copyfile(paths[image_id], out / image_id)
                selected[image_id] = category
        index = predictions.loc[predictions["image_id"].isin(selected)].copy()
        index["category"] = index["image_id"].map(selected)
        index.sort_values(["category", "image_id", "extractor", "classifier"]).to_csv(
            out / "gallery.csv", index=False
        )
