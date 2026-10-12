"""Shared chart style for notebooks and figures, so every figure in the project looks the same."""

from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order (colour-blind safe)
MARKERS = ["o", "s", "^", "D"]                            # second channel, so colour never works alone


def apply_style() -> None:
    """Light surface, quiet axes, thin lines."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    })


def project_root(start: Path | None = None) -> Path:
    """The folder containing pyproject.toml, searching upwards from `start` (default: current folder)."""
    start = Path.cwd() if start is None else Path(start)
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("Could not find pyproject.toml above " + str(start))


def figures_dir() -> Path:
    folder = project_root() / "docs" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
