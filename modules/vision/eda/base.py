"""Run the full CelebA EDA and save its checks, summaries, and figures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from modules.vision.config import load_config
from modules.vision.data.celeba import resolve_image_dir
from modules.vision.eda import bboxes, eval as attributes, images, labels, landmarks


def run_eda(
    config: Mapping[str, Any] | str | Path,
    output_dir: str | Path = "reports/eda",
    max_images: int = 2000,
) -> dict[str, dict[str, int]]:
    """Run every EDA module on the CelebA files in ``config['data_dir']``.

    Saves summary CSVs and PNG figures to ``output_dir`` plus ``checks.json``
    with the integrity counts, and returns those counts. ``max_images`` caps
    the image scan, which is the slowest step.
    """
    cfg = load_config(config)
    data_dir = Path(cfg.get("data_dir", "data"))
    seed = int(cfg.get("seed", 42))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    align_dir = resolve_image_dir(data_dir)

    parts = labels.load_partitions(data_dir / "list_eval_partition.csv")
    attrs = attributes.load_attributes(data_dir / "list_attr_celeba.csv")
    marks = landmarks.load_landmarks(data_dir / "list_landmarks_align_celeba.csv")
    boxes = bboxes.load_bboxes(data_dir / "list_bbox_celeba.csv")
    scanned = images.scan_images(align_dir, max_images=max_images, random_state=seed)

    checks = {
        "partitions": labels.check_partitions(parts),
        "attributes": attributes.check_attributes(attrs, partitions=parts),
        "landmarks": landmarks.check_landmarks(marks, partitions=parts),
        "bboxes": bboxes.check_bboxes(boxes, partitions=parts),
        "images": images.check_images(scanned),
    }
    with open(out / "checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, default=int)

    labels.summarize_partitions(parts).to_csv(out / "partitions_summary.csv")
    attributes.summarize_attributes(attrs).to_csv(out / "attributes_summary.csv")
    attributes.summarize_attribute_partitions(attrs, parts).to_csv(out / "attributes_by_split.csv")
    landmarks.summarize_landmarks(marks).to_csv(out / "landmarks_summary.csv")
    bboxes.summarize_bboxes(boxes).to_csv(out / "bboxes_summary.csv")
    images.summarize_images(scanned).to_csv(out / "images_summary.csv")

    labels.plot_partition_counts(parts, save_path=out / "partition_counts.png")
    attributes.plot_attribute_imbalance(attrs, save_path=out / "attribute_imbalance.png")
    attributes.plot_attribute_correlations(attrs, save_path=out / "attribute_correlations.png")
    landmarks.plot_landmark_distribution(marks, save_path=out / "landmark_nose.png")
    bboxes.plot_bbox_distributions(boxes, save_path=out / "bbox_distributions.png")
    images.plot_image_distributions(scanned, save_path=out / "image_distributions.png")
    plt.close("all")
    return checks
