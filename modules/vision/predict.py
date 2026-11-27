"""Predict a CelebA attribute for new face images with a trained run.

A run folder (``results/<target>/<extractor>/<classifier>``) holds ``model.joblib``
and ``run.json``. The pretrained backbone is rebuilt from torchvision, which
downloads its ImageNet weights on first use (cached in ``~/.cache/torch``).
Faces should be aligned crops like CelebA's for meaningful predictions.

Usage::

    python -m modules.vision.predict results/Smiling/mobilenet_v3_large/logistic_regression face1.jpg face2.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.vision.evaluation.classification import _scores
from modules.vision.run import MODEL_FILE


def predict_images(
    run_dir: str | Path,
    image_paths: Sequence[str | Path],
    device: str = "auto",
    resize: int | Sequence[int] = 224,
) -> pd.DataFrame:
    """Return one row per image with the predicted label (0/1) and the model score."""
    from modules.vision.features.extraction import extract_features  # torch import is slow

    run_dir = Path(run_dir)
    model_path = run_dir / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} not found; re-run the pipeline to save model weights.")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    paths = [str(Path(p)) for p in image_paths]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Images not found: {missing}")

    manifest = pd.DataFrame({"image_id": [Path(p).name for p in paths], "image_path": paths})
    features = extract_features(
        manifest,
        {"extractor": run["extractor"], "device": device, "resize": resize, "batch_size": 32, "num_workers": 0},
    )
    model = joblib.load(model_path)
    labels = np.asarray(model.predict(features)).astype(int)
    scores = _scores(model, features)
    return pd.DataFrame(
        {
            "image": paths,
            run["target"]: labels,
            "score": scores if scores is not None else np.nan,
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/<target>/<extractor>/<classifier>")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Optional CSV path for the predictions.")
    args = parser.parse_args(argv)
    table = predict_images(args.run_dir, args.images, device=args.device)
    print(table.to_string(index=False))
    if args.output is not None:
        table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
