"""Leave-one-group-out Poisson cross-validation (Python equivalent of crossValidate)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cross_validate_poisson_by_group(
    dataset: Any,
    id_col: str,
    dependent_variable: str,
    ind_variables: list[str],
) -> Any:
    """Leave-one-group-out cross-validation using Poisson GLM.

    Python equivalent of R ``crossValidate(dataset, id, dependentVariable, indVariables)``.
    For each unique group in ``id_col``, trains a Poisson GLM on all other groups
    and predicts on the held-out group. Predictions are on the response scale
    (i.e., ``predict(type='response')``).

    Args:
        dataset: GeoDataFrame with columns: id_col, dependent_variable,
            ind_variables, and geometry.
        id_col: Column name for leave-one-group-out grouping.
        dependent_variable: Name of count response column (must be non-negative).
        ind_variables: List of independent variable column names (numeric, no nulls).

    Returns:
        GeoDataFrame identical to input plus a ``Prediction`` column (float)
        containing Poisson mean predictions. Row order matches input. CRS preserved.

    Raises:
        ValueError: If dependent_variable contains negative values, nulls in
            predictors, or fewer than 2 unique groups.
    """
    import statsmodels.api as sm

    # Validate inputs
    y = dataset[dependent_variable]
    if (y < 0).any():
        raise ValueError(
            f"Column '{dependent_variable}' contains negative values; "
            "Poisson requires non-negative counts."
        )
    if y.isna().any():
        raise ValueError(f"Column '{dependent_variable}' contains null values.")

    for col in ind_variables:
        if dataset[col].isna().any():
            raise ValueError(f"Predictor column '{col}' contains null values.")

    groups = dataset[id_col].unique()
    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 unique groups in '{id_col}', got {len(groups)}"
        )

    predictions = pd.Series(np.nan, index=dataset.index)

    for group_id in groups:
        logger.info("CV fold: holding out group %s", group_id)
        train_mask = dataset[id_col] != group_id
        test_mask = dataset[id_col] == group_id

        train_df = dataset[train_mask]
        test_df = dataset[test_mask]

        X_train = train_df[ind_variables].astype(float)
        y_train = train_df[dependent_variable].astype(float)
        X_test = test_df[ind_variables].astype(float)

        X_train_sm = sm.add_constant(X_train, has_constant="add")
        X_test_sm = sm.add_constant(X_test, has_constant="add")

        try:
            model = sm.GLM(y_train, X_train_sm, family=sm.families.Poisson())
            result = model.fit(disp=False)
            preds = result.predict(X_test_sm)
        except Exception as exc:
            raise ValueError(
                f"GLM failed to converge for held-out group '{group_id}': {exc}"
            ) from exc

        predictions.loc[test_df.index] = preds.values

    out = dataset.copy()
    out["Prediction"] = predictions
    return out
