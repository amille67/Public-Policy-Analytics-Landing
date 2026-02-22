"""Standardized model training, saving, and loading wrappers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def fit_linear_regression(
    X: Any,
    y: Any,
    *,
    log_transform: bool = False,
    robust_se: str = "HC1",
) -> Any:
    """Fit an OLS regression model using statsmodels.

    Args:
        X: Feature matrix (array-like or DataFrame). Intercept is added automatically.
        y: Target vector.
        log_transform: If True, log-transform y before fitting.
        robust_se: Covariance type for robust standard errors (e.g., "HC1").

    Returns:
        Fitted statsmodels RegressionResults.
    """
    import numpy as np
    import statsmodels.api as sm

    import_y = y.values if hasattr(y, "values") else y
    if log_transform:
        import_y = np.log(import_y)

    X_sm = sm.add_constant(X, has_constant="add")
    model = sm.OLS(import_y, X_sm)
    result = model.fit(cov_type=robust_se)
    logger.info("OLS fitted: R2=%.4f, n=%d", result.rsquared, result.nobs)
    return result


def fit_random_forest(
    X: Any,
    y: Any,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    seed: int = 42,
) -> Any:
    """Fit a RandomForestRegressor.

    Args:
        X: Feature matrix.
        y: Target vector.
        n_estimators: Number of trees.
        max_depth: Max tree depth.
        seed: Random state for reproducibility.

    Returns:
        Fitted RandomForestRegressor.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    logger.info("RandomForest fitted: n_estimators=%d", n_estimators)
    return model


def fit_logistic_regression(
    X: Any,
    y: Any,
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 42,
) -> Any:
    """Fit a LogisticRegression classifier.

    Args:
        X: Feature matrix.
        y: Binary target vector.
        C: Inverse regularization strength.
        max_iter: Maximum number of iterations.
        seed: Random state.

    Returns:
        Fitted LogisticRegression.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", random_state=seed
    )
    model.fit(X, y)
    logger.info("LogisticRegression fitted: C=%.3f", C)
    return model


def fit_gradient_boosting(
    X: Any,
    y: Any,
    *,
    n_estimators: int = 100,
    max_depth: int = 3,
    seed: int = 42,
) -> Any:
    """Fit a GradientBoostingRegressor.

    Args:
        X: Feature matrix.
        y: Target vector.
        n_estimators: Number of boosting stages.
        max_depth: Max tree depth.
        seed: Random state.

    Returns:
        Fitted GradientBoostingRegressor.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    model.fit(X, y)
    logger.info("GradientBoosting fitted: n_estimators=%d", n_estimators)
    return model


def save_model(model: Any, path: Path) -> None:
    """Save a fitted model to disk using joblib.

    Args:
        model: Fitted model object.
        path: Output path (will create parent dirs).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved: %s", path)


def load_model(path: Path) -> Any:
    """Load a model from disk.

    Args:
        path: Path to the joblib-serialized model.

    Returns:
        Deserialized model object.
    """
    model = joblib.load(path)
    logger.info("Model loaded: %s", path)
    return model
