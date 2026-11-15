"""Classifier training and validation-based model selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid

from modules.vision.models.classification import (
    _parse_classifier_config,
    build_classifier,
)


_SPLIT_ALIASES = {
    0: "train",
    "0": "train",
    "train": "train",
    1: "validation",
    "1": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    2: "test",
    "2": "test",
    "test": "test",
}

_DEFAULT_C_GRID = (0.1, 1.0, 10.0)
_SUPPORTED_METRICS = ("f1", "accuracy", "precision", "recall", "roc_auc")


def _as_feature_matrix(features: Any, expected_rows: int) -> np.ndarray:
    """Validate and convert features into a finite 2D numpy float array."""
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D array.")
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Row count mismatch: features has {matrix.shape[0]} rows, manifest has {expected_rows}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("features contains NaN or infinite values.")
    return matrix


def _resolve_split_series(manifest: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    """Normalize split labels from manifest into 'train', 'validation', or 'test'."""
    split_col = config.get("split_column")
    if split_col is None:
        for candidate in ("split", "partition"):
            if candidate in manifest.columns:
                split_col = candidate
                break
    if split_col is None or split_col not in manifest.columns:
        raise ValueError("manifest must contain a 'split' or 'partition' column.")

    def _map_split(value: Any) -> str | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return _SPLIT_ALIASES.get(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return _SPLIT_ALIASES.get(int(value))
        return _SPLIT_ALIASES.get(str(value).strip().lower())

    normalized = manifest[split_col].map(_map_split)
    if normalized.isna().any():
        raise ValueError(f"manifest['{split_col}'] contains missing or unknown split values.")
    return normalized


def _resolve_target_column(manifest: pd.DataFrame, config: Mapping[str, Any]) -> str:
    """Resolve the single binary target column name from config or manifest."""
    target = config.get("target")
    if target is None:
        targets = config.get("targets")
        if isinstance(targets, str):
            target = targets
        elif isinstance(targets, Sequence) and len(targets) == 1:
            target = targets[0]
        elif isinstance(targets, Sequence) and len(targets) > 1:
            raise ValueError(
                "Binary training expects a single target; set config['target'] when multiple targets exist."
            )
    if target is None:
        for candidate in ("label", "target"):
            if candidate in manifest.columns:
                target = candidate
                break
    if not target or target not in manifest.columns:
        raise ValueError("Target column not found in manifest; specify config['target'].")
    return str(target)


def _normalize_binary_labels(series: pd.Series, target_col: str) -> np.ndarray:
    """Convert binary labels ({0, 1} or {-1, 1}) into integer {0, 1} array."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing or non-numeric values.")
    unique_values = set(numeric.unique().tolist())
    if unique_values.issubset({-1, 1}):
        return (numeric.to_numpy() == 1).astype(int)
    if unique_values.issubset({0, 1}):
        return numeric.to_numpy().astype(int)
    raise ValueError(
        f"Target column '{target_col}' must be binary ({{0, 1}} or {{-1, 1}}), got {sorted(unique_values)}."
    )


def _candidate_param_sets(
    canonical_name: str,
    base_params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the list of hyperparameter dictionaries to evaluate on validation."""
    raw_clf = config.get("classifier")
    clf_grid = raw_clf.get("param_grid") if isinstance(raw_clf, Mapping) else None
    grid = clf_grid if clf_grid is not None else config.get("param_grid")

    if grid is not None:
        if isinstance(grid, Mapping):
            candidates = list(ParameterGrid(grid))
        elif isinstance(grid, Sequence) and not isinstance(grid, (str, bytes)):
            candidates = [dict(item) for item in grid]
        else:
            raise TypeError("param_grid must be a mapping or a sequence of parameter dicts.")
        if not candidates:
            raise ValueError("param_grid must not be empty.")
        return [{**base_params, **item} for item in candidates]

    if canonical_name in ("logistic_regression", "linear_svc") and "C" not in base_params:
        return [{**base_params, "C": c_value} for c_value in _DEFAULT_C_GRID]

    return [dict(base_params)]


def _with_classifier_params(
    config: Mapping[str, Any],
    canonical_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a shallow config copy with explicit classifier name and params."""
    updated = deepcopy(dict(config))
    raw_clf = updated.get("classifier")
    if isinstance(raw_clf, Mapping):
        clf_cfg = dict(raw_clf)
        clf_cfg["name"] = canonical_name
        clf_cfg["params"] = dict(params)
        updated["classifier"] = clf_cfg
    else:
        updated["classifier"] = canonical_name
        updated["classifier_params"] = dict(params)
    return updated


def _compute_validation_metrics(model: Any, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float | None]:
    """Compute binary classification metrics on validation data."""
    y_pred = model.predict(x_val)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if len(np.unique(y_val)) == 2:
        scores = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_val)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                scores = proba[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_val)
        if scores is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_val, scores))

    return metrics


def train_classifier(
    features: Any,
    manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit on training features and select model settings using validation data.

    Only ``train`` rows are used to fit models and only ``validation`` rows are
    used for hyperparameter selection; ``test`` rows are never accessed.
    When ``param_grid`` is omitted for ``logistic_regression`` or ``linear_svc``
    (and ``C`` is not fixed), a default grid ``C in (0.1, 1.0, 10.0)`` is tuned.
    """
    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise ValueError("manifest must be a non-empty pandas DataFrame.")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")

    x_all = _as_feature_matrix(features, expected_rows=len(manifest))
    splits = _resolve_split_series(manifest, config)
    target_col = _resolve_target_column(manifest, config)

    train_mask = splits.eq("train").to_numpy()
    val_mask = splits.eq("validation").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Both 'train' and 'validation' splits must contain at least one sample.")

    # Normalize only train and validation targets so test labels are never touched.
    y_train = _normalize_binary_labels(manifest.loc[train_mask, target_col], target_col)
    y_val = _normalize_binary_labels(manifest.loc[val_mask, target_col], target_col)

    x_train = x_all[train_mask]
    x_val = x_all[val_mask]

    canonical_name, base_params, _, _ = _parse_classifier_config(config)
    if canonical_name != "majority" and len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both classes (0 and 1).")

    selection_metric = str(config.get("selection_metric", "f1")).strip().lower()
    if selection_metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported selection_metric '{selection_metric}'. Choose from: {', '.join(_SUPPORTED_METRICS)}"
        )

    param_sets = _candidate_param_sets(canonical_name, base_params, config)

    best_model = None
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, float | None] = {}
    best_score = -1.0
    trials: list[dict[str, Any]] = []

    for params in param_sets:
        trial_config = _with_classifier_params(config, canonical_name, params)
        model = build_classifier(trial_config)
        model.fit(x_train, y_train)
        val_metrics = _compute_validation_metrics(model, x_val, y_val)

        score = val_metrics[selection_metric]
        if score is None:
            raise ValueError(
                f"Metric '{selection_metric}' is undefined on the validation split (e.g., single class)."
            )

        trials.append({"params": dict(params), "val_metrics": val_metrics})
        if score > best_score:
            best_score = score
            best_model = model
            best_params = dict(params)
            best_metrics = val_metrics

    return {
        "model": best_model,
        "classifier": canonical_name,
        "target": target_col,
        "selection_metric": selection_metric,
        "best_params": best_params,
        "best_score": float(best_score),
        "val_metrics": best_metrics,
        "trials": trials,
        "train_size": int(train_mask.sum()),
        "val_size": int(val_mask.sum()),
    }
