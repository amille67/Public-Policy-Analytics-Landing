"""Metrics computation and JSON-serializable output."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for regression predictions.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dict with keys: ``mae``, ``rmse``, ``r2``.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))

    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(y_true: Any, y_proba: Any) -> dict[str, Any]:
    """Compute ROC-AUC and PR-AUC for binary classification.

    Args:
        y_true: Binary true labels (0/1).
        y_proba: Predicted probabilities for positive class.

    Returns:
        Dict with keys: ``roc_auc``, ``pr_auc``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_proba_arr))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def metrics_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
) -> dict[str, dict[str, float]]:
    """Compute regression metrics per group.

    Args:
        df: DataFrame with true and predicted columns.
        y_true_col: Column name for true values.
        y_pred_col: Column name for predicted values.
        group_col: Column name for grouping variable.

    Returns:
        Dict mapping group value -> metrics dict.
    """
    result: dict[str, dict[str, float]] = {}
    for g, sub in df.groupby(group_col):
        result[str(g)] = regression_metrics(sub[y_true_col], sub[y_pred_col])
    return result
