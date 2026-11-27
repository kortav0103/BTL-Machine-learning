"""End-to-end tests for data preparation, preprocessing, evaluation, comparison and the runner.

A tiny synthetic CelebA folder is built on disk and the pretrained backbone is
replaced by a small untrained network, so no weights are downloaded.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from modules.vision.data.celeba import prepare_data
from modules.vision.data.preprocessing import build_transform
from modules.vision.evaluation.classification import evaluate_classifier
from modules.vision.evaluation.comparison import compare_results
from modules.vision.features import extraction
from modules.vision.features.storage import load_features
from modules.vision.run import run_pipeline
from modules.vision.training.classification import train_classifier

N_IMAGES = 60


def _make_celeba(root):
    """Write a 60-image fake CelebA: Smiling images are bright, others dark."""
    rng = np.random.default_rng(0)
    img_dir = root / "img_align_celeba" / "img_align_celeba"  # Kaggle nesting
    img_dir.mkdir(parents=True)
    ids = [f"{i:06d}.jpg" for i in range(1, N_IMAGES + 1)]
    smiling = np.where(np.arange(N_IMAGES) % 2 == 0, 1, -1)
    for image_id, label in zip(ids, smiling):
        base = 200 if label == 1 else 50
        pixels = np.clip(base + rng.normal(0, 10, (218, 178, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(img_dir / image_id)
    pd.DataFrame({"image_id": ids, "Smiling": smiling, "Male": -smiling}).to_csv(
        root / "list_attr_celeba.csv", index=False
    )
    partition = [0] * 36 + [1] * 12 + [2] * 12
    pd.DataFrame({"image_id": ids, "partition": partition}).to_csv(root / "list_eval_partition.csv", index=False)


def _tiny_extractor(model_name, device="cpu"):
    torch.manual_seed(0)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten())  # 3*4*4 = 48 colour features
    return model.to(device).eval(), 48


@pytest.fixture
def config(tmp_path, monkeypatch):
    _make_celeba(tmp_path / "data")
    monkeypatch.setattr(extraction, "build_extractor", _tiny_extractor)
    return {
        "targets": ["Smiling", "Male"],
        "sample_size": {"train": 30, "validation": None, "test": None},
        "seed": 42,
        "resize": 32,
        "extractors": ["tiny"],
        "classifiers": ["majority", "logistic_regression"],
        "batch_size": 16,
        "num_workers": 0,
        "device": "cpu",
        "data_dir": str(tmp_path / "data"),
        "features_dir": str(tmp_path / "features"),
        "results_dir": str(tmp_path / "results"),
    }


def test_prepare_data_samples_splits_and_maps_labels(config):
    manifest = prepare_data(config)
    assert list(manifest.columns) == ["image_id", "image_path", "partition", "Smiling", "Male"]
    assert manifest["partition"].value_counts().to_dict() == {0: 30, 1: 12, 2: 12}
    assert set(manifest["Smiling"]) == {0, 1}
    assert (manifest["Smiling"] + manifest["Male"] == 1).all()
    assert manifest.equals(prepare_data(config)), "sampling must be deterministic"


def test_prepare_data_rejects_unknown_target(config):
    with pytest.raises(ValueError, match="missing columns"):
        prepare_data({**config, "targets": ["Not_An_Attribute"]})


def test_build_transform_shapes_and_validation():
    image = Image.new("RGB", (178, 218), color=(128, 128, 128))
    assert build_transform({"resize": 64})(image).shape == (3, 64, 64)
    assert build_transform({"resize": [48, 40]})(image).shape == (3, 48, 40)
    assert build_transform({})(image).shape == (3, 224, 224)
    with pytest.raises(ValueError):
        build_transform({"resize": "big"})


def test_evaluate_classifier_uses_only_test_rows(config, tmp_path):
    manifest = prepare_data(config)
    features = extraction.extract_features(manifest, {**config, "extractor": "tiny"})
    run_cfg = {**config, "target": "Smiling", "classifier": "logistic_regression", "run_dir": tmp_path / "run"}
    model = train_classifier(features, manifest, run_cfg)["model"]
    result = evaluate_classifier(model, features, manifest, run_cfg)

    assert result["test_size"] == 12
    assert len(result["predictions"]) == 12
    assert set(result["predictions"]["image_id"]) == set(manifest.loc[manifest["partition"] == 2, "image_id"])
    assert result["metrics"]["accuracy"] == 1.0  # bright vs dark is trivially separable
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved["test_metrics"]["confusion_matrix"]["tp"] == result["metrics"]["confusion_matrix"]["tp"]


def test_compare_results_requires_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare_results(tmp_path, tmp_path / "out")


def test_run_pipeline_end_to_end(config, tmp_path):
    table = run_pipeline(config)

    assert len(table) == 4  # 1 extractor x 2 classifiers x 2 targets
    results = tmp_path / "results"
    for name in ("manifest.csv", "comparison.csv", "comparison.md", "figures/comparison_test_f1.pdf", "figures/confusion_best.png"):
        assert (results / name).is_file()
    run = json.loads((results / "Smiling" / "tiny" / "logistic_regression" / "run.json").read_text())
    assert run["sizes"] == {"train": 30, "validation": 12, "test": 12}
    assert run["test_metrics"]["f1"] == 1.0

    logreg = table[table["classifier"] == "logistic_regression"]["test_f1"]
    majority = table[table["classifier"] == "majority"]["test_f1"]
    assert logreg.min() > majority.max()

    stored = load_features(tmp_path / "features" / "tiny_celeba.npz")
    assert stored["features"].shape == (54, 48)
    assert stored["target_names"] == ["Smiling", "Male"]


def test_run_pipeline_reuses_feature_cache(config, monkeypatch):
    run_pipeline(config)

    def _fail(*args, **kwargs):
        raise AssertionError("features should come from the cache")

    monkeypatch.setattr(extraction, "extract_features", _fail)
    run_pipeline(config)


def test_download_celeba_converts_parquet(tmp_path, monkeypatch):
    import io
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, "scripts")
    import download_celeba

    shard_dir = tmp_path / "hf" / download_celeba.SUBSET
    shard_dir.mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (178, 218)).save(buffer, "PNG")
    image = {"bytes": buffer.getvalue(), "path": "x.png"}
    for split, n in (("train", 4), ("valid", 2), ("test", 2)):
        frame = pd.DataFrame(
            {"image": [image] * n, "celeb_id": range(n), "5_o_Clock_Shadow": [True, False] * (n // 2),
             "Smiling": [False, True] * (n // 2)}
        )
        pq.write_table(pa.Table.from_pandas(frame), shard_dir / f"{split}-00000-of-00001.parquet")
    monkeypatch.setattr(download_celeba, "snapshot_download", lambda *a, **k: str(tmp_path / "hf"))

    data_dir = tmp_path / "data"
    download_celeba.download(data_dir)

    attrs = pd.read_csv(data_dir / "list_attr_celeba.csv")
    assert list(attrs.columns) == ["image_id", "5_o_Clock_Shadow", "Smiling"]
    assert attrs["5_o_Clock_Shadow"].tolist()[:2] == [1, -1]
    assert pd.read_csv(data_dir / "list_eval_partition.csv")["partition"].tolist() == [0, 0, 0, 0, 1, 1, 2, 2]
    manifest = prepare_data({"targets": ["Smiling"], "data_dir": str(data_dir)})
    assert len(manifest) == 8


def test_run_saves_weights_gallery_and_supports_inference(config, tmp_path):
    from modules.vision.evaluation.gallery import OUTCOMES, select_examples
    from modules.vision.predict import predict_images

    run_pipeline(config)
    results = tmp_path / "results"
    run_dir = results / "Smiling" / "tiny" / "logistic_regression"
    assert (run_dir / "model.joblib").is_file()

    index = pd.read_csv(results / "gallery" / "Smiling" / "gallery.csv")
    assert set(index["outcome"]) <= set(OUTCOMES)
    assert all((results / "gallery" / "Smiling" / t).is_file() for t in index["thumbnail"])
    assert (results / "figures" / "gallery_Smiling.png").is_file()

    images = sorted((tmp_path / "data" / "img_align_celeba" / "img_align_celeba").glob("*.jpg"))[:4]
    table = predict_images(run_dir, images, device="cpu", resize=32)
    assert table["Smiling"].tolist() == [1, 0, 1, 0]  # even-numbered fakes are bright (smiling)


def test_select_examples_takes_most_confident_errors():
    from modules.vision.evaluation.gallery import select_examples

    preds = pd.DataFrame(
        {"image_id": list("abcdef"), "y_true": [0, 0, 0, 1, 1, 1], "y_pred": [1, 1, 0, 0, 0, 1],
         "score": [0.6, 0.9, 0.1, 0.4, 0.05, 0.8]}
    )
    picked = select_examples(preds, n_per_outcome=1)
    assert picked.set_index("outcome")["image_id"].to_dict() == {"TP": "f", "TN": "c", "FP": "b", "FN": "e"}


def test_export_bundle_preserves_split_alignment_and_gallery_crops(config, tmp_path):
    manifest = prepare_data(config)
    reference_path = tmp_path / "reference.csv"
    manifest.to_csv(reference_path, index=False)
    bundle_cfg = {
        **config,
        "export_bundle": True,
        "reference_manifest": str(reference_path),
        "extractors": ["tiny", "tiny_alt"],
        "classifiers": ["majority", "logistic_regression", "linear_svc"],
    }

    assert len(run_pipeline(bundle_cfg)) == 12
    results = tmp_path / "results"
    for extractor in bundle_cfg["extractors"]:
        features = np.load(results / "features" / f"{extractor}.npy", allow_pickle=False)
        stored = load_features(tmp_path / "features" / f"{extractor}_celeba.npz")
        np.testing.assert_array_equal(features, stored["features"])
    assert (results / "eda" / "image_distributions.png").is_file()
    assert pd.read_csv(results / "eda" / "class_counts.csv")[["negative", "positive"]].sum().sum() == 108

    for run in results.glob("*/*/*/run.json"):
        validation = pd.read_csv(run.with_name("validation_predictions.csv"))
        predictions = pd.read_csv(run.with_name("predictions.csv"))
        assert validation["image_id"].tolist() == manifest.loc[manifest["partition"] == 1, "image_id"].tolist()
        assert predictions["image_id"].tolist() == manifest.loc[manifest["partition"] == 2, "image_id"].tolist()
        assert run.with_name("model.joblib").is_file()
    crops = results / "comparison_crops" / "Smiling"
    index = pd.read_csv(crops / "gallery.csv")
    assert not index.empty
    assert index.groupby("image_id").size().eq(4).all()
    for image_id in index["image_id"].unique():
        original = manifest.set_index("image_id").loc[image_id, "image_path"]
        assert (crops / image_id).read_bytes() == (tmp_path / original).read_bytes()

    with pytest.raises(FileExistsError, match="fresh results directory"):
        run_pipeline(bundle_cfg)
