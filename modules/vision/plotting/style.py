"""Figure style shared with the lab manuscripts (configs/styles/*.toml).

Adapted from detachable-representation ``scripts/plots/style.py``: Courier New,
thin open axes, light grid, fixed column widths, and multi-format export.
If Courier New is not installed (e.g. on a Vast instance), a monospace fallback
is used so remote runs never fail on fonts; render final figures locally.
"""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import findfont, fontManager
from matplotlib.ticker import AutoMinorLocator

STYLE_DIR = Path(__file__).resolve().parents[3] / "configs" / "styles"
SYSTEM_FONTS = (
    Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
    Path("/Library/Fonts/Courier New.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf"),
)


def _register_font(family: str) -> str:
    """Return ``family`` if it can be found, otherwise a monospace fallback."""
    for candidate in (STYLE_DIR / f"{family}.ttf", *SYSTEM_FONTS):
        if candidate.is_file() and family.replace(" ", "") in candidate.stem.replace(" ", "").replace("_", ""):
            fontManager.addfont(str(candidate))
            break
    try:
        findfont(family, fallback_to_default=False)
        return family
    except ValueError:
        warnings.warn(f"Font '{family}' not found; falling back to DejaVu Sans Mono.", stacklevel=3)
        return "DejaVu Sans Mono"


def apply_style(kind: str = "bar") -> dict[str, Any]:
    """Load one theme (``bar`` or ``heatmap``), set rcParams, and return its settings."""
    if kind not in ("bar", "heatmap"):
        raise ValueError("style must be 'bar' or 'heatmap'")
    with open(STYLE_DIR / f"{kind}.toml", "rb") as f:
        style = tomllib.load(f)
    font, axes, figure = style["typography"], style["axes"], style["figure"]
    family = _register_font(font["family"])

    mpl.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "dejavusans",
            "font.size": font["base_size"],
            "axes.labelsize": font["axis_label_size"],
            "axes.titlesize": font["title_size"],
            "xtick.labelsize": font["tick_label_size"],
            "ytick.labelsize": font["tick_label_size"],
            "legend.fontsize": font["tick_label_size"],
            "legend.frameon": False,
            "axes.linewidth": axes["linewidth"],
            "axes.spines.top": axes["full_frame"],
            "axes.spines.right": axes["full_frame"],
            "xtick.direction": axes["tick_direction"],
            "ytick.direction": axes["tick_direction"],
            "xtick.major.width": axes["tick_width"],
            "ytick.major.width": axes["tick_width"],
            "axes.grid": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": figure["facecolor"],
            "savefig.facecolor": figure["facecolor"],
            "savefig.dpi": figure["export_dpi"],
        }
    )
    return style


def style_bar_axes(ax, style: dict[str, Any]) -> None:
    """Light horizontal grid behind the bars, minor ticks on the value axis."""
    grid = style["bar"]["grid"]
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", color=grid["color"], linewidth=grid["linewidth"])
    ax.spines["left"].set_zorder(style["axes"]["left_spine_zorder"])


def sequential_cmap(style: dict[str, Any]) -> LinearSegmentedColormap:
    """Light-to-dark magnitude colormap from the heatmap theme."""
    return LinearSegmentedColormap.from_list("manuscript_magnitude", style["colormap"]["colors"])


def save_figure(fig, path: str | Path, style: dict[str, Any]) -> list[Path]:
    """Export to every configured format at the declared size (no tight cropping)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in style["figure"]["export_formats"]:
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=style["figure"]["export_dpi"], facecolor=style["figure"]["facecolor"], bbox_inches=None)
        written.append(out)
    return written
