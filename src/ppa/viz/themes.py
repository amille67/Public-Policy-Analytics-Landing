"""Matplotlib theme dictionaries mirroring R's plotTheme and mapTheme helpers."""

from __future__ import annotations

from typing import Any


def plot_theme(*, base_size: int = 12, title_size: int = 16) -> dict[str, object]:
    """Return matplotlib rcParams overrides for non-map plots.

    Python equivalent of R ``plotTheme(base_size, title_size)``.
    The returned dict can be applied via ``matplotlib.rcParams.update(theme)``
    or used as a context manager via ``matplotlib.rc_context(theme)``.

    Titles are plain black (not bold) to match R's ggplot theme defaults.
    Use ``apply_subtitle`` and ``apply_caption`` for figure-level text.

    This function is **pure**: it does not mutate global rcParams.

    Args:
        base_size: Base font size (must be > 0). Default 12.
        title_size: Plot title font size (must be > 0). Default 16.

    Returns:
        Dict of matplotlib rcParams overrides approximating the ggplot
        plotTheme used in the R source.
    """
    if base_size <= 0:
        raise ValueError(f"base_size must be > 0, got {base_size}")
    if title_size <= 0:
        raise ValueError(f"title_size must be > 0, got {title_size}")

    return {
        # Text
        "text.color": "black",
        "font.size": base_size,
        # Title — plain black to match R's plotTheme (not bold)
        "axes.titlesize": title_size,
        "axes.titlecolor": "black",
        # Ticks removed
        "xtick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.width": 0,
        "ytick.major.width": 0,
        # Axis labels
        "axes.labelsize": base_size,
        "xtick.labelsize": max(base_size - 2, 6),
        "ytick.labelsize": max(base_size - 2, 6),
        # Background
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        # Grid
        "axes.grid": True,
        "grid.color": "grey",
        "grid.linewidth": 0.1,
        "axes.grid.which": "major",
        # Border (spines)
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.linewidth": 2.0,
        # Legend
        "legend.fontsize": base_size,
        "legend.title_fontsize": base_size,
        "legend.frameon": False,
    }


def map_theme(*, base_size: int = 12, title_size: int = 16) -> dict[str, object]:
    """Return matplotlib rcParams overrides for map plots.

    Python equivalent of R ``mapTheme(base_size, title_size)``.
    Removes axis titles, tick labels, and gridlines — suitable for
    geographic plots where spatial context replaces axes.

    Titles are plain black (not bold) to match R's ggplot theme defaults.
    Use ``apply_subtitle`` and ``apply_caption`` for figure-level text.

    This function is **pure**: it does not mutate global rcParams.

    Args:
        base_size: Base font size (must be > 0). Default 12.
        title_size: Plot title font size (must be > 0). Default 16.

    Returns:
        Dict of matplotlib rcParams overrides approximating the ggplot
        mapTheme used in the R source.
    """
    if base_size <= 0:
        raise ValueError(f"base_size must be > 0, got {base_size}")
    if title_size <= 0:
        raise ValueError(f"title_size must be > 0, got {title_size}")

    return {
        # Text
        "text.color": "black",
        "font.size": base_size,
        # Title — plain black to match R's mapTheme (not bold)
        "axes.titlesize": title_size,
        "axes.titlecolor": "black",
        # No axis ticks or labels for maps
        "xtick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.width": 0,
        "ytick.major.width": 0,
        "xtick.labelbottom": False,
        "ytick.labelleft": False,
        # No axis labels
        "axes.labelsize": 0,
        # No grid
        "axes.grid": False,
        # Background
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        # Border
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.linewidth": 2.0,
        # Legend
        "legend.fontsize": base_size,
        "legend.frameon": False,
    }


def apply_subtitle(fig: Any, text: str, *, fontsize: int | None = None) -> None:
    """Add an italic subtitle to a figure, mimicking R's plotTheme subtitle style.

    Places the subtitle just below the suptitle (y=0.92) in italic, centred.
    Call after ``fig.suptitle(...)`` so placement is predictable.

    Args:
        fig: matplotlib Figure to annotate.
        text: Subtitle string.
        fontsize: Optional font size override. Defaults to figure's base font size.
    """
    kwargs: dict[str, Any] = {"style": "italic", "ha": "center", "va": "top"}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    fig.text(0.5, 0.92, text, **kwargs)


def apply_caption(fig: Any, text: str, *, fontsize: int = 9) -> None:
    """Add a small italic caption at the bottom-left of a figure.

    Mimics R's plotTheme/mapTheme caption styling (small, bottom-left, italic).

    Args:
        fig: matplotlib Figure to annotate.
        text: Caption string.
        fontsize: Font size. Default 9.
    """
    fig.text(
        0.01, 0.01, text, fontsize=fontsize, ha="left", va="bottom", style="italic"
    )
