"""Non-map plotting utilities for chapter outputs."""

from __future__ import annotations

from typing import Any


def pred_vs_actual(
    y_true: Any,
    y_pred: Any,
    *,
    title: str = "Predicted vs Actual",
    figsize: tuple[float, float] = (8, 6),
    theme: dict | None = None,
) -> Any:
    """Scatter plot of predicted vs actual values with identity line.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        title: Plot title.
        figsize: Figure size in inches.
        theme: rcParams dict from plot_theme().

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true_arr, y_pred_arr, alpha=0.4, s=15)
        lims = [
            min(y_true_arr.min(), y_pred_arr.min()),
            max(y_true_arr.max(), y_pred_arr.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.legend()

    return fig


def utility_by_threshold(
    thresholds: Any,
    utility: Any,
    *,
    title: str = "Utility by Threshold",
    figsize: tuple[float, float] = (9, 5),
    theme: dict | None = None,
) -> Any:
    """Line plot of utility vs threshold.

    Args:
        thresholds: Threshold values (x-axis).
        utility: Utility values (y-axis).
        title: Plot title.
        figsize: Figure size.
        theme: rcParams dict from plot_theme().

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresholds, utility)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Utility")
        ax.set_title(title)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    return fig


def fpr_fnr_tradeoff(
    df: Any,
    *,
    group_col: str = "race",
    fpr_col: str = "False_Positive_Rate",
    fnr_col: str = "False_Negative_Rate",
    title: str = "FPR vs FNR by Group",
    figsize: tuple[float, float] = (9, 6),
    theme: dict | None = None,
) -> Any:
    """Scatter plot of FPR vs FNR colored by group.

    Args:
        df: DataFrame from iterate_fairness output.
        group_col: Column for grouping / coloring.
        fpr_col: False positive rate column.
        fnr_col: False negative rate column.
        title: Plot title.
        figsize: Figure size.
        theme: rcParams dict from plot_theme().

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        for grp, sub in df.groupby(group_col):
            ax.scatter(sub[fpr_col], sub[fnr_col], label=str(grp), alpha=0.5, s=15)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("False Negative Rate")
        ax.set_title(title)
        ax.legend()

    return fig
