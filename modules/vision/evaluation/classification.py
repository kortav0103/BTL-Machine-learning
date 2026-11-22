"""Classifier evaluation and prediction analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from modules.vision.training.classification import (
    _as_feature_matrix,
    _normalize_binary_labels,
    _resolve_split_series,
    _resolve_target_column,
)


def _scores(model: Any, x: np.ndarray) -> np.ndarray | None:
    """Continuous positive-class scores for ROC-AUC, or None if the model has none."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None) -> dict[str, Any]:
    """Binary classification metrics, including the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = None
    if scores is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, scores))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "positive_rate": float(np.mean(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_classifier(
    model: Any,
    features: Any,
    manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the selected model on test data and save metrics and predictions.

    Only ``test`` rows are scored. If ``config['run_dir']`` is set, writes
    ``metrics.json`` and ``predictions.csv`` there.

    Returns
    -------
    dict
        ``metrics`` (test metrics), ``test_size`` and ``predictions`` (DataFrame).
    """
    x_all = _as_feature_matrix(features, expected_rows=len(manifest))
    splits = _resolve_split_series(manifest, config)
    target = _resolve_target_column(manifest, config)

    test_mask = splits.eq("test").to_numpy()
    if not test_mask.any():
        raise ValueError("The 'test' split contains no samples.")

    x_test = x_all[test_mask]
    y_test = _normalize_binary_labels(manifest.loc[test_mask, target], target)
    y_pred = np.asarray(model.predict(x_test)).astype(int)
    scores = _scores(model, x_test)
    metrics = compute_metrics(y_test, y_pred, scores)

    predictions = pd.DataFrame(
        {
            "image_id": manifest.loc[test_mask, "image_id"].to_numpy(),
            "y_true": y_test,
            "y_pred": y_pred,
            "score": scores if scores is not None else np.nan,
        }
    )

    run_dir = config.get("run_dir")
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(run_dir / "predictions.csv", index=False)
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"target": target, "test_size": int(test_mask.sum()), "test_metrics": metrics}, f, indent=2)

    return {"metrics": metrics, "test_size": int(test_mask.sum()), "predictions": predictions}
