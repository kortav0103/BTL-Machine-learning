"""Unit tests for vision classification models and validation training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from modules.vision.models.classification import build_classifier
from modules.vision.training.classification import train_classifier


def _make_mock_dataset(
    n_samples: int = 240,
    celeba_labels: bool = True,
    string_splits: bool = False,
):
    features, labels = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        weights=[0.65, 0.35],
        random_state=42,
    )
    if celeba_labels:
        labels = np.where(labels == 1, 1, -1)

    n_train = int(n_samples * 0.6)
    n_val = int(n_samples * 0.2)
    n_test = n_samples - n_train - n_val

    if string_splits:
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        split_col = "split"
    else:
        splits = [0] * n_train + [1] * n_val + [2] * n_test
        split_col = "partition"

    manifest = pd.DataFrame(
        {
            "image_id": [f"{i:06d}.jpg" for i in range(n_samples)],
            split_col: splits,
            "Smiling": labels,
        }
    )
    return features, manifest


def test_build_classifier_supported_models_and_aliases():
    majority = build_classifier("majority")
    assert isinstance(majority.named_steps["classifier"], DummyClassifier)

    logreg = build_classifier({"classifier": "logreg", "seed": 7})
    assert "scaler" in logreg.named_steps
    assert isinstance(logreg.named_steps["classifier"], LogisticRegression)
    assert logreg.named_steps["classifier"].random_state == 7

    svc = build_classifier(
        {"classifier": {"name": "svm", "params": {"C": 0.5}, "scale": False}, "seed": 11}
    )
    assert "scaler" not in svc.named_steps
    assert isinstance(svc.named_steps["classifier"], LinearSVC)
    assert svc.named_steps["classifier"].C == 0.5


def test_build_classifier_rejects_missing_or_unknown_name():
    with pytest.raises(ValueError):
        build_classifier({"classifier": None})
    with pytest.raises(ValueError):
        build_classifier({"classifier": "xgboost"})


@pytest.mark.parametrize("classifier_name", ["majority", "logistic_regression", "linear_svc"])
def test_train_classifier_runs_end_to_end_and_beats_baseline(classifier_name):
    features, manifest = _make_mock_dataset(celeba_labels=True, string_splits=False)
    result = train_classifier(
        features,
        manifest,
        {"classifier": classifier_name, "target": "Smiling", "seed": 42},
    )

    assert result["classifier"] == classifier_name
    assert result["target"] == "Smiling"
    assert result["selection_metric"] == "f1"
    assert result["train_size"] == 144
    assert result["val_size"] == 48
    assert result["model"] is not None
    assert 0.0 <= result["best_score"] <= 1.0
    if classifier_name in ("logistic_regression", "linear_svc"):
        assert len(result["trials"]) == 3
        assert result["best_score"] > 0.5


def test_train_classifier_supports_param_grid_and_string_splits():
    features, manifest = _make_mock_dataset(celeba_labels=False, string_splits=True)
    result = train_classifier(
        features,
        manifest,
        {
            "classifier": {
                "name": "logistic_regression",
                "param_grid": {"C": [0.05, 0.5]},
            },
            "target": "Smiling",
            "selection_metric": "accuracy",
            "seed": 42,
        },
    )
    assert result["selection_metric"] == "accuracy"
    assert len(result["trials"]) == 2
    assert result["best_params"]["C"] in (0.05, 0.5)


def test_train_classifier_does_not_leak_test_split():
    features, manifest = _make_mock_dataset(celeba_labels=True, string_splits=False)
    base_res = train_classifier(
        features,
        manifest,
        {"classifier": "logistic_regression", "target": "Smiling", "seed": 42},
    )

    # Corrupt test labels; training and validation must remain completely unaffected.
    corrupted_manifest = manifest.copy()
    corrupted_manifest.loc[corrupted_manifest["partition"] == 2, "Smiling"] = 999
    corrupted_res = train_classifier(
        features,
        corrupted_manifest,
        {"classifier": "logistic_regression", "target": "Smiling", "seed": 42},
    )

    assert base_res["best_params"] == corrupted_res["best_params"]
    assert base_res["val_metrics"] == corrupted_res["val_metrics"]
    np.testing.assert_array_equal(
        base_res["model"].predict(features[:20]),
        corrupted_res["model"].predict(features[:20]),
    )


def test_train_classifier_validation_errors():
    features, manifest = _make_mock_dataset()

    with pytest.raises(ValueError, match="Row count mismatch"):
        train_classifier(features[:-1], manifest, {"classifier": "majority", "target": "Smiling"})

    bad_features = features.copy()
    bad_features[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        train_classifier(bad_features, manifest, {"classifier": "majority", "target": "Smiling"})

    no_val_manifest = manifest.loc[manifest["partition"] != 1].reset_index(drop=True)
    no_val_features = features[manifest["partition"].to_numpy() != 1]
    with pytest.raises(ValueError, match="validation"):
        train_classifier(
            no_val_features,
            no_val_manifest,
            {"classifier": "majority", "target": "Smiling"},
        )

    with pytest.raises(ValueError, match="selection_metric"):
        train_classifier(
            features,
            manifest,
            {"classifier": "majority", "target": "Smiling", "selection_metric": "mse"},
        )
