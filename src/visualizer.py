"""Server-side chart rendering for report export.

The web UI draws charts in the browser from JSON specs (see ``src/charts.py``).
Matplotlib is only needed when exporting a self-contained HTML/PDF report, so
this module is imported lazily by the report generator.

``generate_visualizations`` is kept as a bounded, curated replacement for the
old cartesian generator: it renders at most ``max_charts`` figures instead of
one per column and every column pair.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.charts import default_charts
from src.formatting import format_number
from src.profile import build_profile

INK = "#0f172a"
MUTED = "#64748b"
BRAND = "#4f46e5"
ACCENT = "#0d9488"
GRID = "#e2e8f0"


def _style(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)


def _short(value, _position=None) -> str:
    return format_number(value)


def _render_histogram(ax, spec: dict) -> None:
    bins = spec["data"]["bins"]
    if not bins:
        return
    widths = [max(item["end"] - item["start"], 1e-9) for item in bins]
    ax.bar(
        [item["start"] for item in bins],
        [item["count"] for item in bins],
        width=widths,
        align="edge",
        color=BRAND,
        alpha=0.85,
        linewidth=0,
    )
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    ax.xaxis.set_major_formatter(FuncFormatter(_short))
    _style(ax)


def _render_bar(ax, spec: dict) -> None:
    categories = list(reversed(spec["data"]["categories"]))
    labels = [item["label"] for item in categories]
    counts = [item["count"] for item in categories]
    ax.barh(labels, counts, color=BRAND, alpha=0.85, height=0.7)
    ax.set_xlabel("Rows")
    ax.xaxis.set_major_formatter(FuncFormatter(_short))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)


def _render_missing(ax, spec: dict) -> None:
    columns = list(reversed(spec["data"]["columns"]))
    labels = [item["column"] for item in columns]
    percentages = [item["pct"] for item in columns]
    ax.barh(labels, percentages, color="#f59e0b", alpha=0.9, height=0.7)
    ax.set_xlabel("Missing %")
    ax.set_xlim(0, max(max(percentages, default=0) * 1.2, 1))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)


def _render_heatmap(ax, spec: dict) -> None:
    values = np.array(
        [[np.nan if value is None else value for value in row] for row in spec["data"]["values"]],
        dtype=float,
    )
    labels = spec["data"]["columns"]
    if values.size == 0:
        return
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(False)
    ax.tick_params(colors=MUTED)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if len(labels) <= 10:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    continue
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#0f172a" if abs(value) < 0.55 else "#ffffff",
                )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _render_line(ax, spec: dict) -> None:
    points = spec["data"]["points"]
    values = [point[1] for point in points]
    positions = list(range(len(points)))
    ax.plot(positions, values, color=BRAND, linewidth=2)
    ax.fill_between(positions, values, color=BRAND, alpha=0.1)
    step = max(len(points) // 6, 1)
    ax.set_xticks(positions[::step])
    ax.set_xticklabels([points[index][0] for index in range(0, len(points), step)], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(spec["y_label"])
    ax.yaxis.set_major_formatter(FuncFormatter(_short))
    _style(ax)


def _render_scatter(ax, spec: dict) -> None:
    points = np.array(spec["data"]["points"], dtype=float)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.45, color=BRAND, linewidths=0)
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    ax.xaxis.set_major_formatter(FuncFormatter(_short))
    ax.yaxis.set_major_formatter(FuncFormatter(_short))
    _style(ax)


def _render_box(ax, spec: dict) -> None:
    groups = spec["data"]["groups"]
    if not groups:
        return
    stats = [
        {
            "label": group["label"],
            "med": group["median"],
            "q1": group["q1"],
            "q3": group["q3"],
            "whislo": group["whisker_low"],
            "whishi": group["whisker_high"],
            "fliers": [],
        }
        for group in groups
    ]
    ax.bxp(stats, showfliers=False, patch_artist=True,
           boxprops={"facecolor": "#c7d2fe", "edgecolor": BRAND},
           medianprops={"color": ACCENT, "linewidth": 2},
           whiskerprops={"color": MUTED},
           capprops={"color": MUTED})
    ax.set_ylabel(spec["y_label"])
    ax.set_xlabel(spec["x_label"])
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    _style(ax)


RENDERERS = {
    "histogram": _render_histogram,
    "bar": _render_bar,
    "missing": _render_missing,
    "heatmap": _render_heatmap,
    "line": _render_line,
    "scatter": _render_scatter,
    "box": _render_box,
}


def render_spec(spec: dict):
    """Render one chart spec into a Matplotlib figure."""
    renderer = RENDERERS.get(spec.get("type"))
    if renderer is None:
        raise ValueError(f"Unsupported chart type for reports: {spec.get('type')}")

    figure, axes = plt.subplots(figsize=(6.6, 3.9), dpi=110)
    renderer(axes, spec)
    axes.set_title(spec.get("title", ""), loc="left", pad=10)
    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    return buffer.getvalue()


def close_figures(figures) -> None:
    for figure in figures:
        plt.close(figure)


def generate_visualizations(
    dataframe: pd.DataFrame,
    *,
    max_charts: int = 5,
) -> dict:
    """Curated figures for reports (identifier-free, bounded).

    Kept as a compatible entry point; the old behaviour of rendering every
    column and every numeric pair is intentionally gone.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dataframe.empty:
        raise ValueError("DataFrame is empty.")

    profile = build_profile(dataframe)
    specs = default_charts(dataframe, profile)[:max_charts]
    return {spec["id"]: render_spec(spec) for spec in specs}
