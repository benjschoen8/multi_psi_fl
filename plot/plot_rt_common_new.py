"""Shared style for the *_new.py plot scripts (fixed categorical order; PSI always slot 1)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
          "#17becf", "#8c564b", "#7f7f7f", "#bcbd22", "#9467bd", "#000000"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42,
})


def parse_named(items):
    """['Name=path[:xcol]', ...] -> [(name, path, xcol|None)] (order kept = color order)."""
    out = []
    for s in items or []:
        name, rest = s.split("=", 1)
        path, _, xcol = rest.partition("::")
        out.append((name, path, xcol or None))
    return out


def colors(names, psi_names):
    """PSI entries take the first slots (in given order), others follow. Stable per entity;
    every line gets its own color (PSI lines are additionally dashed by the callers)."""
    order = [n for n in names if n in psi_names] + [n for n in names if n not in psi_names]
    return {n: SERIES[k % len(SERIES)] for k, n in enumerate(order)}
