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


def threshold_rate_curves(
    df: Any,
    *,
    threshold_col: str = "Threshold",
    rate_tp_col: str = "Rate_TP",
    rate_fp_col: str = "Rate_FP",
    accuracy_col: str = "Accuracy",
    title: str = "Classifier Performance by Threshold",
    figsize: tuple[float, float] = (9, 5),
    theme: dict | None = None,
) -> Any:
    """Line plot of TP rate, FP rate, and Accuracy vs decision threshold.

    Visualises the ROC-adjacent tradeoff curves produced by
    ``iterate_thresholds``: how sensitivity (Rate_TP), specificity complement
    (Rate_FP), and overall accuracy change as the classification threshold
    moves from 0 to 1.

    Args:
        df: DataFrame from ``iterate_thresholds`` (ungrouped output).
        threshold_col: Column with threshold values (x-axis). Default ``"Threshold"``.
        rate_tp_col: True-positive-rate column. Default ``"Rate_TP"``.
        rate_fp_col: False-positive-rate column. Default ``"Rate_FP"``.
        accuracy_col: Accuracy column. Default ``"Accuracy"``.
        title: Plot title.
        figsize: Figure size in inches.
        theme: rcParams dict from plot_theme().

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            df[threshold_col],
            df[rate_tp_col],
            label="True Positive Rate",
            linewidth=1.5,
        )
        ax.plot(
            df[threshold_col],
            df[rate_fp_col],
            label="False Positive Rate",
            linewidth=1.5,
            linestyle="--",
        )
        ax.plot(
            df[threshold_col],
            df[accuracy_col],
            label="Accuracy",
            linewidth=1.5,
            linestyle=":",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axvline(
            0.5, color="gray", linestyle="--", linewidth=0.7, label="Threshold = 0.5"
        )
        ax.legend()

    return fig


def fairness_frontier(
    df: Any,
    *,
    group_col: str = "race",
    group_a: str = "African-American",
    group_b: str = "Caucasian",
    fpr_col: str = "False_Positive_Rate",
    fnr_col: str = "False_Negative_Rate",
    title: str = "Fairness Frontier: FPR Gap vs FNR Gap",
    figsize: tuple[float, float] = (8, 7),
    theme: dict | None = None,
) -> Any:
    """Scatter plot of between-group FPR gap vs FNR gap across threshold combos.

    Each point represents one (tA, tB) threshold combination from
    ``iterate_fairness``. The axes show the signed gap between group A and
    group B for the false-positive rate (x) and false-negative rate (y).

    Points near the origin represent threshold combinations where both groups
    experience similar error rates — the ``fairness frontier``. Points in the
    upper-right indicate group A bears higher errors in both dimensions.

    Args:
        df: DataFrame from ``iterate_fairness``.
        group_col: Column identifying demographic group.
        group_a: Label for the primary group (plotted on numerator side of gap).
        group_b: Label for the reference group.
        fpr_col: False positive rate column.
        fnr_col: False negative rate column.
        title: Plot title.
        figsize: Figure size in inches.
        theme: rcParams dict from plot_theme().

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(df)  # ensure pandas; no-op if already DataFrame

    a = df[df[group_col] == group_a].reset_index(drop=True)
    b = df[df[group_col] == group_b].reset_index(drop=True)

    # Align by threshold string — both groups have the same set of threshold
    # combinations in iterate_fairness output (same ordering)
    fpr_gap = a[fpr_col].values - b[fpr_col].values
    fnr_gap = a[fnr_col].values - b[fnr_col].values

    with plt.rc_context(theme or {}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(fpr_gap, fnr_gap, alpha=0.3, s=12, color="#0072B2")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"FPR gap  ({group_a} - {group_b})")
        ax.set_ylabel(f"FNR gap  ({group_a} - {group_b})")
        ax.set_title(title)
        # Mark the origin (perfect parity) prominently
        ax.scatter([0], [0], color="red", s=60, zorder=5, label="Perfect parity")
        ax.legend()

    return fig
