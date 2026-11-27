#!/usr/bin/env bash
# Run the full CelebA pipeline on a Vast instance (called by `vast-run run celeba`).
# vast-run has already checked out the repo and run `uv sync --locked`.
# Outputs go to results/, which vast-run archives and brings back.
set -euo pipefail

config=${1:-configs/vision/celeba.json}
reference=${2:-results/run2-cgrid-wide/manifest.csv}
run_id="celeba-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)-$$"
output="results/$run_id"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

printf '== GPU ==\n'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

printf '== Downloading CelebA ==\n'
uv run --locked python scripts/download_celeba.py --data-dir data

printf '== Running pipeline with %s ==\n' "$config"
uv run --locked python -m modules.vision.run \
  --config "$config" --device cuda --num-workers 8 \
  --results-dir "$output" --export-bundle --reference-manifest "$reference"

git rev-parse HEAD > "$output/git_commit.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$output/gpu.csv"
printf '== Done: %s ==\n' "$output"
cat "$output/comparison.md"
