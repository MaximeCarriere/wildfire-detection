"""Shared figure style — one place, so every plot in the repo reads as one system.

Palette is the validated categorical default (slots 1–3, the subset that clears
the colour-vision-deficiency separation floors under all pairings). Colour is
never the only carrier of meaning: every series is also directly labelled or
distinguished by position, so the figures survive greyscale printing and CVD.

Design constraints, chosen for a reader who is not a computer-vision engineer:
one idea per panel, the headline stated in the title, units on every axis, and
the key number annotated on the mark rather than left to the reader to look up.
"""
from __future__ import annotations

# Categorical slots (validated default palette, light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#d03b3b"        # status:critical — reserved for a failed configuration

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#dcdcd8"
SURFACE = "#fcfcfb"


def apply() -> None:
    """Global matplotlib defaults. Call once before building any figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 11,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 160,
    })


def subtitle(fig, text: str, y: float = 0.945) -> None:
    """One plain-language sentence under the title — what the reader should take
    away, in words, before they parse a single axis."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=10.5, color=INK_2)


def annotate(ax, x, y, text, *, dx=0, dy=8, color=INK, weight="bold", size=10):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha="center", fontsize=size, color=color, fontweight=weight)


def tidy(ax, *, ygrid=True) -> None:
    ax.grid(axis="y" if ygrid else "x", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
