"""Two-group fairness threshold grid (Python equivalent of iterateFairness)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def iterate_fairness(
    data: pd.DataFrame,
    regression: Any,
    threshold_by: float,
    *,
    observed_col: str = "Recidivated",
    group_col: str = "race",
    group_a: str = "African-American",
    group_b: str = "Caucasian",
    positive_label: str = "Recidivate",
    negative_label: str = "notRecidivate",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute a fairness grid by sweeping thresholds for two demographic groups.

    Python equivalent of R ``iterateFairness(data, regression, threshold.by)``.
    Uses **greater-than-or-equal** (``>=``) for threshold comparison, matching R.

    The function sweeps all combinations of (tA, tB) thresholds for the two
    groups and computes per-group confusion metrics.

    Args:
        data: DataFrame containing group_col, observed_col, and feature columns.
        regression: Fitted model with probability output.
            - If has ``predict_proba``: uses ``predict_proba(X)[:, 1]``.
            - Else: uses ``predict(X)`` treating output as probability.
        threshold_by: Step size for threshold grid (e.g., 0.1 → 10x10 = 100 combos).
            Must be > 0.
        observed_col: Column with string outcome labels.
        group_col: Column identifying demographic group.
        group_a: Label for group A (e.g., "African-American").
        group_b: Label for group B (e.g., "Caucasian").
        positive_label: String for positive outcome (e.g., "Recidivate").
        negative_label: String for negative outcome (e.g., "notRecidivate").
        feature_cols: Columns to pass to model. When ``None``, features are
            inferred by excluding ``observed_col`` and ``group_col`` from
            ``data``. If the model exposes ``feature_names_in_`` (sklearn),
            that takes precedence to match trained column order exactly.

    Returns:
        DataFrame with columns:
            ``race`` (or group_col name), ``True_Negative, True_Positive,``
            ``False_Negative, False_Positive, False_Positive_Rate,``
            ``False_Negative_Rate, Accuracy, threshold``

    Raises:
        ValueError: If ``threshold_by <= 0``, required groups are not present,
            observed labels are invalid, no feature columns can be inferred,
            or model output is out of [0, 1].
    """
    # Bug 3 fix: validate threshold_by > 0 to prevent silent empty/invalid grids
    if threshold_by <= 0:
        raise ValueError(f"threshold_by must be > 0, got {threshold_by!r}")

    # Validate groups
    for g in [group_a, group_b]:
        if g not in data[group_col].values:
            raise ValueError(
                f"Required group '{g}' not found in column '{group_col}'. "
                f"Available: {data[group_col].unique().tolist()}"
            )

    # Validate observed labels
    valid_labels = {positive_label, negative_label}
    actual_labels = set(data[observed_col].unique())
    unknown = actual_labels - valid_labels
    if unknown:
        raise ValueError(
            f"Observed column '{observed_col}' contains unexpected labels: {unknown}. "
            f"Expected: {valid_labels}"
        )

    # Bug 1 fix: safe feature selection — never pass outcome/group cols to model
    if feature_cols is not None:
        X = data[feature_cols]
    elif hasattr(regression, "feature_names_in_"):
        # sklearn-compatible model: use the exact trained feature names to
        # prevent column order/name mismatches and label leakage
        X = data[list(regression.feature_names_in_)]
    else:
        # Infer features: exclude outcome and group columns to prevent leakage
        exclude = {observed_col, group_col}
        inferred = [c for c in data.columns if c not in exclude]
        if not inferred:
            raise ValueError(
                f"No feature columns remain after excluding '{observed_col}' and "
                f"'{group_col}'. Pass feature_cols explicitly."
            )
        logger.debug("iterate_fairness: inferred feature_cols=%s", inferred)
        X = data[inferred]

    if hasattr(regression, "predict_proba"):
        probs = regression.predict_proba(X)[:, 1]
    else:
        probs = regression.predict(X)
        if hasattr(probs, "values"):
            probs = probs.values

    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(
            "Model predictions are outside [0, 1]; cannot use as probabilities"
        )

    # Build threshold grid: match R's seq(0.1, 1, threshold_by) — stops at <= 1.0
    thresh_range = np.arange(0.1, 1.0 + threshold_by / 100, threshold_by)
    thresh_range = thresh_range[thresh_range <= 1.0 + 1e-9]
    thresh_range = np.round(thresh_range, 10)
    all_combos = [(ta, tb) for ta in thresh_range for tb in thresh_range]

    observed = data[observed_col].values
    groups = data[group_col].values

    all_rows: list[dict] = []

    for ta, tb in all_combos:
        # Assign predicted labels using >= rule (matches R)
        predicted = np.where(
            (groups == group_a) & (probs >= ta),
            positive_label,
            np.where(
                (groups == group_b) & (probs >= tb),
                positive_label,
                negative_label,
            ),
        )

        threshold_str = f"{round(float(ta), 10)}, {round(float(tb), 10)}"

        for g_val, g_mask in [
            (group_a, groups == group_a),
            (group_b, groups == group_b),
        ]:
            obs_g = observed[g_mask]
            pred_g = predicted[g_mask]

            tn = int(((pred_g == negative_label) & (obs_g == negative_label)).sum())
            tp = int(((pred_g == positive_label) & (obs_g == positive_label)).sum())
            fn = int(((pred_g == negative_label) & (obs_g == positive_label)).sum())
            fp = int(((pred_g == positive_label) & (obs_g == negative_label)).sum())
            total = len(obs_g)

            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
            accuracy = (tp + tn) / total if total > 0 else np.nan

            all_rows.append(
                {
                    group_col: g_val,
                    "True_Negative": tn,
                    "True_Positive": tp,
                    "False_Negative": fn,
                    "False_Positive": fp,
                    "False_Positive_Rate": fpr,
                    "False_Negative_Rate": fnr,
                    "Accuracy": accuracy,
                    "threshold": threshold_str,
                }
            )

    return pd.DataFrame(all_rows)
