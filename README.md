# BTL-Machine-learning

Binary facial-attribute classification on [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
with frozen pretrained CNN features and classical classifiers.

For each target attribute (default: `Smiling`, `Male`, `Eyeglasses`), image features from two pretrained
backbones (ResNet-18 and MobileNetV3-Large) are fed to a majority baseline, logistic regression and a linear SVM.
Hyperparameters are tuned on the official validation split; the test split is used only for the final scores.

## Setup

```bash
uv sync            # or: python3 -m venv .venv && pip install -r requirements.txt
```

Either download the Kaggle dataset and unzip it into `data/`, or run
`uv run python scripts/download_celeba.py` to build the same layout from the Hugging Face
mirror `flwrlabs/celeba` (official splits and attributes; no landmarks/bboxes, and image
ids are renumbered). The Kaggle layout is:

```
data/
  list_attr_celeba.csv
  list_eval_partition.csv
  list_landmarks_align_celeba.csv
  list_bbox_celeba.csv
  img_align_celeba/img_align_celeba/000001.jpg ...
```

## Running

Full pipeline (data → features → training → test evaluation → comparison):

```bash
python -m modules.vision.run --config configs/vision/celeba.json
```

Features are cached in `features/<extractor>_celeba.npz` and reused while the sampled manifest is unchanged;
pass `--overwrite-features` to re-extract. The notebook `notebooks/run_vision.ipynb` walks through the EDA,
runs the same pipeline and plots the results.

## Running on Vast (`vast-run`)

`launch.toml` defines the task `celeba`. vast-run checks out `feat/runner` from GitHub,
runs `uv sync --locked`, then `scripts/vast_run.sh`, which downloads CelebA from the
Hugging Face mirror, runs the pipeline on the GPU and leaves everything in `results/`;
vast-run brings `results/` back and destroys the instance.

```bash
vast-run run celeba --dry-run   # check the task without renting
vast-run run celeba             # pick an offer, run, fetch results/
uv run python -m modules.vision.plot results   # re-render figures locally in Courier New
```

The branch must be pushed first, since the instance clones it from GitHub.

## Inference

Each run saves its fitted classifier as `model.joblib` (scaler + linear model, a few KB),
so predictions on new aligned face crops need no retraining. The backbone's ImageNet
weights download from torchvision on first use and are cached in `~/.cache/torch`.

```bash
uv run python -m modules.vision.predict results/Smiling/mobilenet_v3_large/logistic_regression face1.jpg face2.png
```

## Figures

Figures use the lab manuscript style from `configs/styles/` (`bar.toml`, `heatmap.toml`,
loaded by `modules/vision/plotting/style.py`): Courier New, open thin axes, fixed column
widths, exported as PDF, SVG and PNG to `results/figures/`. If Courier New is missing
(e.g. on Vast) a monospace fallback is used; put `Courier New.ttf` in `configs/styles/`
or re-render locally.

Tests:

```bash
python -m pytest tests -q
```

## Configuration (`configs/vision/celeba.json`)

| Key | Meaning |
|---|---|
| `targets` | CelebA attributes to classify, one binary task each |
| `sample_size` | Images sampled per split (`train`, `validation`, `test`); `null` uses the whole split |
| `seed` | Seed for sampling and classifiers |
| `resize` | Input size for the backbones (int or `[height, width]`) |
| `extractors` | Backbones: `resnet18`, `resnet50`, `mobilenet_v3_large` |
| `classifiers` | `majority`, `logistic_regression`, `linear_svc` |
| `selection_metric` | Validation metric for picking hyperparameters (`f1`, `accuracy`, `roc_auc`, ...) |
| `batch_size`, `num_workers`, `device` | Feature-extraction settings; `device: "auto"` uses CUDA when available |
| `data_dir`, `features_dir`, `results_dir` | Input and output folders |

## Outputs

```
results/
  manifest.csv                     sampled images, splits and 0/1 labels
  comparison.csv / comparison.md   one row per (target, extractor, classifier)
  figures/                         comparison_test_f1, comparison_test_roc_auc, confusion_best,
                                   gallery_<target> (.pdf/.svg/.png)
  gallery/<target>/                thumbnails + gallery.csv: TP/TN and the most confident FP/FN
                                   test faces of the best run, so the gallery re-renders without data/
  <target>/<extractor>/<classifier>/
    run.json                       chosen params, validation trials, val + test metrics
    metrics.json                   test metrics and confusion matrix
    predictions.csv                per-image test predictions and scores
    model.joblib                   fitted classifier for modules.vision.predict
reports/eda/                       EDA summaries, figures and integrity checks (run_eda)
```

## Code layout

```
modules/vision/
  config.py                 JSON config loading
  data/celeba.py            prepare_data: manifest with paths, official splits, 0/1 labels
  data/preprocessing.py     build_transform: resize + ImageNet normalisation
  eda/                      partition, image, landmark, bbox and attribute EDA; base.run_eda runs all
  features/extraction.py    pretrained backbone feature extraction
  features/storage.py       save/load features with ids, splits and labels
  models/classification.py  classifier pipelines
  training/classification.py  training with validation-based selection
  evaluation/classification.py  test-set metrics and predictions
  evaluation/comparison.py  comparison table and plots across runs
  plotting/style.py         lab figure style
  run.py                    end-to-end runner
  evaluation/gallery.py     prediction gallery export and plot
  plot.py                   re-render figures from a results folder
  predict.py                inference on new images with a saved run
scripts/
  download_celeba.py        CelebA from the Hugging Face mirror into data/
  vast_run.sh               remote entry point used by launch.toml
```
