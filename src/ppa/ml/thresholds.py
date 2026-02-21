"""Threshold sweep for binary classifier evaluation (Python equivalent of iterateThresholds)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def iterate_thresholds(
    data: pd.DataFrame,
    observed_class: str,
    predicted_probs: str,
    group: str | None = None,
    *,
    step: float = 0.01,
) -> pd.DataFrame:
    """Sweep probability thresholds and compute confusion matrix metrics.

    Python equivalent of R ``iterateThresholds(data, observedClass, predictedProbs, group)``.
    Iterates thresholds from ``step`` to 1.00 inclusive. Uses **strict greater-than**
    (``>``) for threshold comparison, matching R behavior.

    Args:
        data: DataFrame with observed class and predicted probability columns.
        observed_class: Column name containing binary observed labels (0/1 or False/True).
        predicted_probs: Column name containing predicted probabilities in [0, 1].
        group: Optional column name for grouped metrics (e.g., race). If provided,
            confusion metrics are computed per group per threshold.
        step: Threshold step size. Default 0.01 produces 100 thresholds (0.01..1.00).

    Returns:
        DataFrame with columns:
            ``Count_TN, Count_TP, Count_FN, Count_FP,``
            ``Rate_TP, Rate_FP, Rate_FN, Rate_TN,``
            ``Accuracy, Threshold``
            (plus the group column name if ``group`` is provided).
    """
    obs = data[observed_class].astype(int)
    probs = data[predicted_probs].astype(float)
    groups = data[group] if group is not None else None

    thresholds = np.round(np.arange(step, 1.0 + step / 2, step), 2)
    all_rows: list[dict] = []

    for x in thresholds:
        thresh = round(float(x), 2)
        pred = (probs > thresh).astype(int)

        if groups is not None:
            for g_val, g_idx in data.groupby(group).groups.items():
                row = _confusion_row(obs.loc[g_idx], pred.loc[g_idx], thresh)
                row[group] = g_val
                all_rows.append(row)
        else:
            row = _confusion_row(obs, pred, thresh)
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    return result


def _confusion_row(obs: pd.Series, pred: pd.Series, threshold: float) -> dict:
    """Compute one threshold row of confusion metrics."""
    tn = int(((pred == 0) & (obs == 0)).sum())
    tp = int(((pred == 1) & (obs == 1)).sum())
    fn = int(((pred == 0) & (obs == 1)).sum())
    fp = int(((pred == 1) & (obs == 0)).sum())
    total = len(obs)

    rate_tp = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    rate_fp = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    rate_fn = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    rate_tn = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / total if total > 0 else np.nan

    return {
        "Count_TN": tn,
        "Count_TP": tp,
        "Count_FN": fn,
        "Count_FP": fp,
        "Rate_TP": rate_tp,
        "Rate_FP": rate_fp,
        "Rate_FN": rate_fn,
        "Rate_TN": rate_tn,
        "Accuracy": accuracy,
        "Threshold": threshold,
    }
