"""Traditional image feature classifiers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


_CLASSIFIER_ALIASES = {
    "majority": "majority",
    "dummy": "majority",
    "baseline": "majority",
    "logistic_regression": "logistic_regression",
    "logreg": "logistic_regression",
    "linear_svc": "linear_svc",
    "svm": "linear_svc",
}

SUPPORTED_CLASSIFIERS = ("majority", "logistic_regression", "linear_svc")


def _parse_classifier_config(config: Mapping[str, Any] | str) -> tuple[str, dict[str, Any], int | None, bool]:
    """Normalize string or dict config into (canonical_name, params, seed, scale_features)."""
    if isinstance(config, str):
        raw_name = config
        params: dict[str, Any] = {}
        seed: int | None = 42
        scale_features = True
    elif isinstance(config, Mapping):
        seed = config.get("seed", 42)
        raw_clf = config.get("classifier")
        if isinstance(raw_clf, Mapping):
            raw_name = raw_clf.get("name") or raw_clf.get("type")
            params = dict(raw_clf.get("params") or {})
            scale_features = bool(raw_clf.get("scale", True))
        else:
            raw_name = raw_clf
            params = dict(config.get("classifier_params") or {})
            scale_features = bool(config.get("scale_features", True))
    else:
        raise TypeError("config must be a mapping or classifier name string.")

    if not raw_name or not isinstance(raw_name, str):
        raise ValueError(
            f"Classifier name is required. Supported classifiers: {', '.join(SUPPORTED_CLASSIFIERS)}"
        )

    canonical_name = _CLASSIFIER_ALIASES.get(raw_name.strip().lower())
    if canonical_name is None:
        raise ValueError(
            f"Unsupported classifier '{raw_name}'. Supported classifiers: {', '.join(SUPPORTED_CLASSIFIERS)}"
        )
    return canonical_name, params, seed, scale_features


def build_classifier(config: Mapping[str, Any] | str) -> Pipeline:
    """Construct an unfitted classifier and its preprocessing pipeline.

    Supported classifiers:
    - ``majority``: ``DummyClassifier(strategy="most_frequent")`` baseline.
    - ``logistic_regression``: ``StandardScaler`` + ``LogisticRegression``.
    - ``linear_svc``: ``StandardScaler`` + ``LinearSVC``.
    """
    name, params, seed, scale_features = _parse_classifier_config(config)

    if name == "majority":
        estimator = DummyClassifier(
            strategy=params.pop("strategy", "most_frequent"),
            random_state=params.pop("random_state", seed),
            **params,
        )
        return Pipeline([("classifier", estimator)])

    builders = {
        "logistic_regression": lambda p: LogisticRegression(
            max_iter=p.pop("max_iter", 1000),
            random_state=p.pop("random_state", seed),
            **p,
        ),
        "linear_svc": lambda p: LinearSVC(
            max_iter=p.pop("max_iter", 10000),
            random_state=p.pop("random_state", seed),
            **p,
        ),
    }

    estimator = builders[name](dict(params))
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)
