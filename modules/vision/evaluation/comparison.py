"""Comparison of feature extractors and classifier configurations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.vision.plotting.style import apply_style, save_figure, sequential_cmap, style_bar_axes

RUN_FILE = "run.json"
SHORT_NAMES = {"majority": "Majority", "logistic_regression": "LogReg", "linear_svc": "LinSVM"}
SHORT_EXTRACTORS = {"resnet18": "ResNet18", "resnet50": "ResNet50", "mobilenet_v3_large": "MobileNetV3"}
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")


def collect_runs(results_dir: str | Path) -> pd.DataFrame:
    """Flatten every ``run.json`` under ``results_dir`` into one row per run."""
    rows = []
    for path in sorted(Path(results_dir).rglob(RUN_FILE)):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        row = {
            "target": run["target"],
            "extractor": run["extractor"],
            "classifier": run["classifier"],
            "best_params": json.dumps(run.get("best_params", {}), sort_keys=True),
            "val_f1": run["val_metrics"].get("f1"),
        }
        row.update({f"test_{m}": run["test_metrics"].get(m) for m in METRICS})
        rows.append(row)
    columns = ["target", "extractor", "classifier", "best_params", "val_f1", *(f"test_{m}" for m in METRICS)]
    return pd.DataFrame(rows, columns=columns)


def plot_comparison(table: pd.DataFrame, metric: str = "test_f1") -> plt.Figure:
    """Grouped bars of ``metric`` per classifier and extractor, one panel per target."""
    style = apply_style("bar")
    palette, bar = style["palette"], style["bar"]
    targets = sorted(table["target"].unique())
    width = style["figure"]["two_column_width"] if len(targets) > 1 else style["figure"]["one_column_width"]
    fig, axes = plt.subplots(1, len(targets), figsize=(width, style["figure"]["height"]), squeeze=False, sharey=True)
    for ax, target in zip(axes[0], targets):
        pivot = table[table["target"] == target].pivot(index="classifier", columns="extractor", values=metric)
        x = np.arange(len(pivot.index))
        step = bar["height"] / max(len(pivot.columns), 1)
        for i, extractor in enumerate(pivot.columns):
            values = pivot[extractor].to_numpy(dtype=float)
            bars = ax.bar(x + i * step, values, step, label=extractor, zorder=bar["zorder"],
                          color=palette.get(extractor, palette["other"]))
            ax.bar_label(bars, fmt="%.2f", padding=1.5, rotation=90,
                         fontsize=style["typography"]["value_label_size"] - 1)
        ax.set_xticks(x + step * (len(pivot.columns) - 1) / 2, [SHORT_NAMES.get(c, c) for c in pivot.index])
        ax.set_title(target.replace("_", " "))
        ax.set_ylim(0, 1.15)
        style_bar_axes(ax, style)
    axes[0][0].set_ylabel(metric.replace("test_", "test ").replace("_", "-"))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    return fig


def plot_confusion_matrices(runs: list[tuple[str, pd.DataFrame]]) -> plt.Figure:
    """Row-normalised confusion matrices, one panel per ``(title, predictions)`` pair."""
    style = apply_style("heatmap")
    cmap, cells = sequential_cmap(style), style["cells"]
    width = style["figure"]["two_column_width"] if len(runs) > 1 else style["figure"]["one_column_width"]
    fig, axes = plt.subplots(1, len(runs), figsize=(width, style["figure"]["height"]), squeeze=False)
    for ax, (title, preds) in zip(axes[0], runs):
        counts = pd.crosstab(preds["y_true"], preds["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        rates = counts.to_numpy() / np.maximum(counts.to_numpy().sum(axis=1, keepdims=True), 1)
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for (i, j), rate in np.ndenumerate(rates):
            ax.text(j, i, f"{rate:.2f}\n({counts.iat[i, j]})", ha="center", va="center",
                    color="white" if rate > 0.6 else "black", fontsize=style["typography"]["tick_label_size"])
        ax.set_xticks([0, 1], ["neg", "pos"])
        ax.set_yticks([0, 1], ["neg", "pos"])
        ax.set_xticks(np.arange(-0.5, 2), minor=True)
        ax.set_yticks(np.arange(-0.5, 2), minor=True)
        ax.grid(which="minor", color=cells["edgecolor"], linewidth=cells["edge_linewidth"] * 4)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("predicted")
        ax.set_title(title)
    axes[0][0].set_ylabel("true")
    fig.tight_layout()
    return fig


def best_runs(table: pd.DataFrame) -> pd.DataFrame:
    """Best configuration per target, chosen by validation F1 (never by test score)."""
    return table.loc[table.groupby("target")["val_f1"].idxmax()].reset_index(drop=True)


def render_figures(results_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Render comparison and best-run confusion figures from saved results."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "figures"
    table = collect_runs(results_dir)
    if table.empty:
        raise FileNotFoundError(f"No {RUN_FILE} files found under {results_dir}.")
    written = []
    for metric in ("test_f1", "test_roc_auc"):
        fig = plot_comparison(table, metric)
        written += save_figure(fig, output_dir / f"comparison_{metric}", apply_style("bar"))
        plt.close(fig)
    runs = [
        (f"{r.target.replace('_', ' ')}\n{SHORT_EXTRACTORS.get(r.extractor, r.extractor)} + "
         f"{SHORT_NAMES.get(r.classifier, r.classifier)}",
         pd.read_csv(results_dir / r.target / r.extractor / r.classifier / "predictions.csv"))
        for r in best_runs(table).itertuples()
    ]
    fig = plot_confusion_matrices(runs)
    written += save_figure(fig, output_dir / "confusion_best", apply_style("heatmap"))
    plt.close(fig)

    from modules.vision.evaluation.gallery import GALLERY_DIR, INDEX_FILE, plot_gallery

    for index in sorted((results_dir / GALLERY_DIR).glob(f"*/{INDEX_FILE}")):
        fig = plot_gallery(index.parent)
        written += save_figure(fig, output_dir / f"gallery_{index.parent.name}", apply_style("heatmap"))
        plt.close(fig)
    return written


def compare_results(results_dir: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Save comparison tables and figures for compatible experiment results.

    Writes ``comparison.csv``, ``comparison.md`` and ``figures/`` (PDF, SVG, PNG)
    into ``output_dir`` and returns the comparison table.
    """
    table = collect_runs(results_dir)
    if table.empty:
        raise FileNotFoundError(f"No {RUN_FILE} files found under {results_dir}.")
    table = table.sort_values(["target", "extractor", "classifier"]).reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "comparison.csv", index=False)

    shown = table.drop(columns="best_params").round(4)
    header = "| " + " | ".join(shown.columns) + " |"
    divider = "|" + "---|" * len(shown.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    (output_dir / "comparison.md").write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")

    render_figures(results_dir, output_dir / "figures")
    return table
