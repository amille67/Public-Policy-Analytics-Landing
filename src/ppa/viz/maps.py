"""Map plotting utilities for geospatial chapter outputs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colorblind-safe palette presets
# ---------------------------------------------------------------------------
# Sequential palettes suitable for quintile choropleth maps.
# All palettes are perceptually uniform and safe for the most common forms
# of colour vision deficiency (deuteranopia/protanopia).
SEQUENTIAL_PALETTES: dict[str, str] = {
    "viridis": "viridis",  # default — perceptually uniform, colourblind-safe
    "plasma": "plasma",
    "cividis": "cividis",  # designed for CVD viewers
    "YlOrRd": "YlOrRd",  # traditional sequential (not CVD-optimal)
}

# Categorical palette for two-group fairness plots (colourblind-safe pair).
# Wong (2011) palette — works for deuteranopia, protanopia, and tritanopia.
CATEGORICAL_TWO: list[str] = ["#0072B2", "#D55E00"]  # blue, vermillion

# Full 8-colour Wong palette for multi-group maps.
CATEGORICAL_WONG: list[str] = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Default polygon styling for choropleth overlays — thin dark border aids
# cartographic legibility by separating adjacent polygons of similar hue.
DEFAULT_EDGECOLOR: str = "#444444"
DEFAULT_LINEWIDTH: float = 0.4


def choropleth_map(
    gdf: Any,
    column: str,
    *,
    title: str = "",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 8),
    edgecolor: str = DEFAULT_EDGECOLOR,
    linewidth: float = DEFAULT_LINEWIDTH,
    legend_kwds: dict | None = None,
    overlay_gdfs: list[Any] | None = None,
    overlay_colors: list[str] | None = None,
    theme: dict | None = None,
) -> Any:
    """Create a choropleth map of a GeoDataFrame column.

    Args:
        gdf: GeoDataFrame with polygon geometry.
        column: Column name to map (numeric).
        title: Plot title.
        cmap: Matplotlib colormap name. Default ``"viridis"`` (colourblind-safe).
            Use ``SEQUENTIAL_PALETTES`` keys for preset options.
        figsize: Figure dimensions (width, height) in inches.
        edgecolor: Polygon border colour. Default ``"#444444"`` (dark grey).
            Set to ``"none"`` to disable borders.
        linewidth: Polygon border width in points. Default 0.4.
        legend_kwds: Extra keyword arguments forwarded to geopandas legend/colorbar.
            Useful for controlling placement in small multiples, e.g.
            ``{"shrink": 0.5, "orientation": "horizontal"}``.
        overlay_gdfs: Additional GeoDataFrames to plot on top.
        overlay_colors: Colors for each overlay layer.
        theme: rcParams dict from map_theme() to apply.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    _legend_kwds = legend_kwds or {}

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        gdf.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            legend=True,
            edgecolor=edgecolor,
            linewidth=linewidth,
            legend_kwds=_legend_kwds,
        )

        if overlay_gdfs:
            colors = overlay_colors or ["blue"] * len(overlay_gdfs)
            for ogdf, color in zip(overlay_gdfs, colors):
                ogdf.plot(ax=ax, color=color, markersize=3, alpha=0.7)

        ax.set_title(title)
        ax.set_axis_off()

    return fig


def scatter_plot(
    x: Any,
    y: Any,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "",
    figsize: tuple[float, float] = (8, 6),
    theme: dict | None = None,
) -> Any:
    """Create a scatter plot.

    Args:
        x: X-axis data.
        y: Y-axis data.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        figsize: Figure size in inches.
        theme: rcParams dict from plot_theme() to apply.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, y, alpha=0.5, s=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    return fig
