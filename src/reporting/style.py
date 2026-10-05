"""
Shared chart style.

Palette: the validated reference palette of the dataviz skill. Categorical
slots are used in fixed order (blue, orange, aqua); magnitude uses a single-hue
blue ramp; polarity uses blue and red around a gray midpoint. Text always
wears ink colors, never a series color, and series are direct-labeled.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898883"
GRID = "#e6e5e1"
NEUTRAL = "#c9c8c2"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"

# Ordinal ramp for ordered groups (lightest step still clears 2:1 on the surface)
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Sequential ramp for heatmaps, light to dark
BLUE_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    """Set matplotlib defaults: light surface, recessive grid, no chart junk."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.dpi": 100,
            "savefig.dpi": 160,
        }
    )


def headline(fig, title: str, subtitle: Optional[str] = None) -> None:
    """Left-aligned finding-style title with a quieter subtitle."""
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=10, color=INK_SECONDARY)


def footnote(fig, text: str) -> None:
    """Small source / method note at the bottom left."""
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color=INK_MUTED)


def save(fig, output_dir: str, name: str) -> str:
    """Save a figure as PNG and close it. Returns the file path."""
    path = Path(output_dir) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)
