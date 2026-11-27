# Weekly meeting minutes

<!-- FORMAT — copy this block after each weekly meeting, newest at the bottom.
     Graded on being dated, consistent with the commit history, and showing a real task split.
     Give every decision an ID (D-NN) and expand it in DECISIONS.md.

## YYYY-MM-DD (week-NN) — <time>, <place or platform, e.g. B1-304 / Google Meet>
- Attendance: <StudentID, StudentID, ...>   Absent: <StudentID reason, or "none">
- Review of last week: <what landed, what slipped>
- Decisions:
  - D-NN: <one line> (see DECISIONS.md)
- Task split for the coming week (who does what across all four modules):
  - <StudentID Name> -> <req X.Y short description>
- Next meeting: YYYY-MM-DD
-->

## 2026-09-26 (week-05) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Decisions: 
  - D-01: Choose dataset and task.
  - D-02: Build the GITHUB repo structure.
- Task split for the coming week (who does what across all four modules):
  - 2352470 -> Structure REPO; EDA (Images)
  - 2352551 -> Choose Dataset: CelebA; EDA Labels (Partitions, Landmarks)
  - 2353219 -> Write Docs; EDA Labels (BBOX, EVAL)
- Next meeting: 2026-10-03
- Dataset link: [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## 2026-10-03 (week-06) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: 2352470 scaffolded the repository structure (`18aa6ac`); 2352551 completed `eda/labels.py` and `eda/landmarks.py` (`a4eefe9`). Image, bounding-box, and attribute EDA modules remain in progress.
- Decisions:
  - D-03: Preserve the official CelebA evaluation partitions (`0 = train`, `1 = validation`, `2 = test`) without random re-splitting to avoid identity leakage across splits.
  - D-04: Separate bounding-box and 40-attribute EDA into dedicated modules (`bboxes.py`, `eval.py`) and assemble an interactive walkthrough in `notebooks/run_vision.ipynb`.
- Task split for the coming week:
  - 2352470 -> Implement `eda/images.py` (resolution, color mode, file-size, and luminance checks).
  - 2353219 -> Implement `eda/eval.py`, `eda/bboxes.py`, and initial EDA cells in `notebooks/run_vision.ipynb`.
  - 2352551 -> Verify partition and landmark integrity checks across all 202,599 rows.
- Next meeting: 2026-10-10

## 2026-10-10 (week-07) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: All five EDA modules landed (`ecd9244`, `4ba7492`, `d29d2a3`) and executed on the full 202,599-image Kaggle CelebA dataset in `notebooks/run_vision.ipynb` (0 missing partitions/attributes/landmarks, 1 degenerate zero-size raw bbox, tight aligned landmark dispersion, strong attribute imbalance, and a 3.92-point split shift on `Male`).
- Decisions:
  - D-05: Operate directly on pre-aligned `img_align_celeba` crops (`178x218` RGB) without adding a separate face-detection stage.
  - D-06: Select three independent binary target attributes spanning complementary class-balance regimes: `Smiling` (near-balanced), `Male` (mildly imbalanced), and `Eyeglasses` (strongly imbalanced, ~6.5% positive).
- Task split for the coming week:
  - 2352470 -> Draft JSON configuration schema (`configs/vision/celeba.json`) and unified `eda/base.py`.
  - 2352551 -> Research linear-probe classifiers (`DummyClassifier`, `LogisticRegression`, `LinearSVC`) and leakage-free validation protocol.
  - 2353219 -> Evaluate pretrained `torchvision` CNN feature extractors (`ResNet-18`, `MobileNetV3-Large`, `ResNet-50`).
- Next meeting: 2026-10-17

## 2026-10-17 (week-08) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: Reviewed full-dataset EDA conclusions, candidate CNN backbones, and compute budget for reproducible experimentation across 3 targets × 2 backbones × 3 classifiers.
- Decisions:
  - D-07: Adopt frozen ImageNet-pretrained CNN feature extraction (`ResNet-18` 512-d and `MobileNetV3-Large` 960-d) paired with classical linear models (`Majority`, `LogisticRegression`, `LinearSVC`).
  - D-08: Sample each official split independently with `seed = 42` to construct a deterministic 28,000-image subset (`20,000 train / 4,000 validation / 4,000 test`).
- Task split for the coming week:
  - 2352470 -> Define module interfaces across `data/`, `features/`, `models/`, `training/`, and `evaluation/`.
  - 2352551 -> Specify evaluation metrics (`Accuracy`, `Balanced Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`) and `val_f1` selection criterion.
  - 2353219 -> Design row-aligned `.npz` feature caching format with `image_ids`, `splits`, `labels`, and JSON metadata.
- Next meeting: 2026-10-24

## 2026-10-24 (week-09) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: Finalized preprocessing transforms, feature serialization schema, and leakage-prevention contracts across pipeline modules.
- Decisions:
  - D-09: Use `Resize((224, 224)) + ToTensor() + ImageNet Normalize` instead of center-cropping to retain peripheral cues (eyeglass temples, chin contours, hairlines).
  - D-10: Encapsulate `StandardScaler` inside `sklearn.pipeline.Pipeline` and isolate the test split completely during training and hyperparameter selection.
- Task split for the coming week:
  - 2352470 -> Prepare pipeline module stubs and open PR #1 (`eda` -> `main`).
  - 2352551 -> Finalize design of `models/classification.py` and `training/classification.py`.
  - 2353219 -> Finalize design of `features/extraction.py` and `features/storage.py`.
- Next meeting: 2026-10-31

## 2026-10-31 (week-10) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: Reviewed `eda` branch readiness for merge; confirmed blank module layout under `modules/vision/` and branch-per-feature workflow (`feat/classification`, `feat/runner`).
- Decisions:
  - D-11: Merge `eda` branch into `main` first (PR #1) and commit skeleton modules plus `configs/vision/celeba.json` so feature extraction and classification can proceed in parallel.
- Task split for the coming week:
  - 2352470 -> Commit skeleton modules and `celeba.json`, and merge PR #1.
  - 2353219 -> Begin implementing `features/extraction.py` and `features/storage.py`.
  - 2352551 -> Begin implementing classifiers and validation grid search.
- Next meeting: 2026-11-03

## 2026-11-03 (week-11) - 21:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: EDA modules and demo exist. Blank pipeline modules are added. Training and results remain unfinished.
- Decisions:
  - D-12: Use Jupyter notebooks as entry points. Defer notebook creation.
  - D-13: Agree on target attributes, experiment settings, and feature format before implementation.
- Task split for the coming week:
  - 2353219 -> Extract features from two pretrained extractors with image IDs, labels, and splits.
  - 2352551 -> Implement two classifiers, a majority baseline, validation.
  - 2352470 -> Approve settings and feature format. Review modules, split handling, and initial results. validation, tuning and plotting.
- Next meeting: 2026-11-10

## 2026-11-10 (week-12) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: PR #1 (`eda`) merged (`e186ed9`, 2026-11-04) and pipeline stubs + `celeba.json` landed (`1f92dea`). Implementation of feature extraction and classification modules actively underway.
- Decisions:
  - D-14: Enforce automated `pytest` unit tests including an explicit test-split leakage test (`test_train_classifier_does_not_leak_test_split` corrupting test labels to `999`).
  - D-15: Support `resnet50` in `extraction.py` as a configurable option while benchmarking `resnet18` and `mobilenet_v3_large` as the primary extractors.
- Task split for the coming week:
  - 2352551 -> Complete `models/classification.py`, `training/classification.py`, and `tests/test_vision_classification.py`.
  - 2353219 -> Complete `features/extraction.py` and `features/storage.py` with row-order alignment checks.
  - 2352470 -> Implement `data/celeba.py`, `data/preprocessing.py`, and evaluation/comparison modules.
- Next meeting: 2026-11-17

## 2026-11-17 (week-13) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: 2352551 landed `feat/classification` (`9ccbef7`, 2026-11-15: classifiers, validation grid, leakage unit tests); 2353219 landed feature extraction and `.npz` serialization (`f87ba12`, 2026-11-16).
- Decisions:
  - D-16: Use the public Hugging Face mirror `flwrlabs/celeba` via `scripts/download_celeba.py` for reproducible, credential-free remote GPU runs on Vast.ai (`scripts/vast_run.sh`).
  - D-17: Expand the regularization grid downward to `C in {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0}` and set `LinearSVC(max_iter=10000)` to prevent overfitting in 960-d feature space and ensure solver convergence.
- Task split for the coming week:
  - 2352470 -> Merge PR #2 & PR #3, build `run.py`, `evaluation/{classification,comparison,gallery,bundle}.py`, `plot.py`, `predict.py`, `tests/test_vision_pipeline.py`, and run full GPU benchmark.
  - 2352551 -> Review test metrics and draft report outline and methodology.
  - 2353219 -> Verify feature cache dimensions/metadata and export EDA charts for the report.
- Next meeting: 2026-11-24

## 2026-11-24 (week-14) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: PR #2 (`67c12d8`) and PR #3 (`bcde293`) merged to `main` on 2026-11-22; 2352470 completed runner, plotting, galleries, expanded C-grid (`b56c929`), and initial 18-configuration GPU runs (`d1c4ca8`–`da79300`).
- Decisions:
  - D-18: Execute the final portable artifact export (`--export-bundle` with `--reference-manifest` verification, `validation_predictions.csv`, `comparison_crops/`, `.npy` feature matrices, and compressed MobileNetV3 archive) on NVIDIA RTX A4000.
  - D-19: Draft the full HCMUT LaTeX report (`report/main.tex`) grounded strictly in verified `origin/main` outputs and qualitative gallery error inspection.
- Task split for the coming week:
  - 2352470 -> Run final RTX A4000 bundle export, commit compressed MobileNetV3 feature archive, and merge PR #4.
  - 2352551 -> Draft full LaTeX report Sections 1–6 and appendices, and audit all metrics against `comparison.csv` and `run.json`.
  - 2353219 -> Review EDA and feature-extraction sections, inspect gallery error cases, and verify repository documentation.
- Next meeting: 2026-11-27

## 2026-11-27 (week-14, Final Pre-Submission Review) - 20:00, Google Meet
- Attendance: 2352470, 2352551, 2353219   Absent: none
- Review of last week: Final RTX A4000 bundle run (`0bb3bc0`), MobileNetV3 archive (`5ce5561`), and updated CelebA results (`5fb8616`) merged to `origin/main` via PR #4 (`6ddb47e`). Best validation-selected test F1 achieved: `Eyeglasses` `0.8065`, `Male` `0.9187`, `Smiling` `0.7962`. Completed academic and technical audit of `report/main.pdf`.
- Decisions:
  - D-20: Apply all audit fixes to `report/main.tex` and `report/references.bib` (mandatory Work Distribution & Contribution table, figure caption counts 34/14, 8-stage pipeline wording, ROC-AUC bound `<= 0.0033`, accurate artifact-logging and Colab documentation, fixed `--results-dir` reproduction command, and formatting polish) before compiling the final submission PDF.
- Task split for final submission:
  - 2352470 -> Verify final repository state, `README.md` completeness, and Colab packaging.
  - 2352551 -> Apply final LaTeX fixes, recompile, and visually verify `report/main.pdf`.
  - 2353219 -> Attach teamwork evidence screenshots and verify final submission `.zip` structure.
- Next meeting: None (final submission).
