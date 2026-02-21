"""Map plotting utilities for geospatial chapter outputs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def choropleth_map(
    gdf: Any,
    column: str,
    *,
    title: str = "",
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = (10, 8),
    overlay_gdfs: list[Any] | None = None,
    overlay_colors: list[str] | None = None,
    theme: dict | None = None,
) -> Any:
    """Create a choropleth map of a GeoDataFrame column.

    Args:
        gdf: GeoDataFrame with polygon geometry.
        column: Column name to map (numeric).
        title: Plot title.
        cmap: Matplotlib colormap name.
        figsize: Figure dimensions (width, height) in inches.
        overlay_gdfs: Additional GeoDataFrames to plot on top.
        overlay_colors: Colors for each overlay layer.
        theme: rcParams dict from map_theme() to apply.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        gdf.plot(column=column, ax=ax, cmap=cmap, legend=True)

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
