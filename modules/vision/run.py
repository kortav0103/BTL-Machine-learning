"""End-to-end CelebA vision experiment runner.

Pipeline: prepare manifest -> extract features per backbone (cached) ->
train each classifier per target with validation selection -> evaluate on test ->
save the fitted classifier -> export a prediction gallery -> compare all runs.

Usage::

    python -m modules.vision.run --config configs/vision/celeba.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from modules.vision.config import load_config
from modules.vision.data.celeba import prepare_data
from modules.vision.evaluation.bundle import export_comparison_crops, export_eda, export_features
from modules.vision.evaluation.classification import evaluate_classifier, score_split
from modules.vision.evaluation.comparison import compare_results
from modules.vision.evaluation.gallery import export_gallery
from modules.vision.features.storage import load_features, save_features
from modules.vision.training.classification import train_classifier


MODEL_FILE = "model.joblib"


def _as_list(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not value:
        raise ValueError(f"config['{name}'] must list at least one entry.")
    return list(value)


def get_features(
    manifest: pd.DataFrame,
    cfg: Mapping[str, Any],
    extractor: str,
    overwrite: bool = False,
    timing: dict[str, float] | None = None,
) -> np.ndarray:
    """Load cached features for ``extractor`` if they match ``manifest``, else extract and save them."""
    path = Path(cfg["features_dir"]) / f"{extractor}_celeba.npz"
    if path.is_file() and not overwrite:
        cached = load_features(path)
        if np.array_equal(cached["image_ids"].astype(str), manifest["image_id"].astype(str).to_numpy()):
            print(f"[features] using cache {path}")
            return cached["features"]
        print(f"[features] cache {path} does not match the manifest; re-extracting")

    from modules.vision.features.extraction import extract_features  # torch import is slow

    started = perf_counter()
    features = extract_features(manifest, {**cfg, "extractor": extractor})
    if timing is not None:
        timing["extraction_seconds"] = perf_counter() - started
    save_features(features, manifest, {**cfg, "extractor": extractor}, output_path=path)
    print(f"[features] saved {path} {features.shape}")
    return features


def run_experiment(
    features: np.ndarray,
    manifest: pd.DataFrame,
    cfg: Mapping[str, Any],
    extractor: str,
    classifier: str,
    target: str,
) -> dict[str, Any]:
    """Train, select on validation, evaluate on test, and save ``run.json`` for one configuration."""
    run_dir = Path(cfg["results_dir"]) / target / extractor / classifier
    run_cfg = {**cfg, "target": target, "classifier": classifier, "run_dir": run_dir}

    trained = train_classifier(features, manifest, run_cfg)
    evaluated = evaluate_classifier(trained["model"], features, manifest, run_cfg)
    joblib.dump(trained["model"], run_dir / MODEL_FILE)  # scaler + classifier, a few KB

    if cfg.get("export_bundle"):
        validation = score_split(trained["model"], features, manifest, run_cfg, "validation")
        validation.to_csv(run_dir / "validation_predictions.csv", index=False)

    summary = {
        "target": target,
        "extractor": extractor,
        "classifier": trained["classifier"],
        "selection_metric": trained["selection_metric"],
        "best_params": trained["best_params"],
        "val_metrics": trained["val_metrics"],
        "test_metrics": evaluated["metrics"],
        "trials": trained["trials"],
        "sizes": {"train": trained["train_size"], "validation": trained["val_size"], "test": evaluated["test_size"]},
        "seed": cfg.get("seed"),
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_pipeline(config: Mapping[str, Any] | str | Path, overwrite_features: bool = False) -> pd.DataFrame:
    """Run every (extractor, classifier, target) combination and return the comparison table."""
    cfg = load_config(config)
    cfg.setdefault("features_dir", "features")
    cfg.setdefault("results_dir", "results")
    targets = _as_list(cfg.get("targets"), "targets")
    extractors = _as_list(cfg.get("extractors") or cfg.get("extractor"), "extractors")
    classifiers = _as_list(cfg.get("classifiers") or cfg.get("classifier"), "classifiers")

    bundle = cfg.get("export_bundle", False)
    results_dir = Path(cfg["results_dir"])
    if bundle and results_dir.exists():
        raise FileExistsError(f"Choose a fresh results directory: {results_dir}")

    manifest = prepare_data(cfg)
    if reference_path := cfg.get("reference_manifest"):
        columns = ["image_id", "partition", *targets]
        reference = pd.read_csv(reference_path, usecols=columns, dtype={"image_id": str})
        pd.testing.assert_frame_equal(manifest[columns], reference[columns])
    counts = manifest["partition"].value_counts().sort_index().to_dict()
    print(f"[data] {len(manifest)} images, per split {counts}, targets {targets}")

    results_dir.mkdir(parents=True, exist_ok=not bundle)
    manifest.to_csv(results_dir / "manifest.csv", index=False)

    if bundle:
        (results_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        export_eda(results_dir, manifest, targets, seed=int(cfg.get("seed", 42)))

    for extractor in extractors:
        timing: dict[str, float] = {}
        features = get_features(manifest, cfg, extractor, overwrite=overwrite_features, timing=timing)
        if bundle:
            features = export_features(results_dir, extractor, features, timing)
        for target in targets:
            for classifier in classifiers:
                summary = run_experiment(features, manifest, cfg, extractor, classifier, target)
                print(
                    f"[run] {target:<12} {extractor:<20} {classifier:<20} "
                    f"val_f1={summary['val_metrics']['f1']:.4f} test_f1={summary['test_metrics']['f1']:.4f}"
                )

    if bundle:
        export_comparison_crops(results_dir, manifest)
    export_gallery(results_dir, manifest)
    print(f"[gallery] wrote {results_dir / 'gallery'}")
    table = compare_results(results_dir, results_dir)
    print(f"[compare] wrote {results_dir / 'comparison.csv'}")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/vision/celeba.json", help="Path to the JSON config.")
    parser.add_argument("--overwrite-features", action="store_true", help="Re-extract even if a cache exists.")
    parser.add_argument("--device", help="Override config['device'], e.g. cuda or cpu.")
    parser.add_argument("--num-workers", type=int, help="Override config['num_workers'].")
    parser.add_argument("--results-dir", help="Override the result directory.")
    parser.add_argument("--export-bundle", action="store_true", help="Export submission and analysis artifacts.")
    parser.add_argument("--reference-manifest", help="Require identical sample IDs, splits and labels.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.export_bundle:
        cfg["export_bundle"] = True
    if args.reference_manifest is not None:
        cfg["reference_manifest"] = args.reference_manifest
    run_pipeline(cfg, overwrite_features=args.overwrite_features)


if __name__ == "__main__":
    main()
