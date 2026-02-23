"""Quantile utilities: q5 binning and qBr break labels.

Python equivalents of the R helpers ``q5`` (ntile-based binning) and
``qBr`` (quantile break label formatting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def q5(values: ArrayLike) -> pd.Categorical:
    """Bin numeric values into 5 quantile tiles.

    Python equivalent of R ``q5 <- function(variable) as.factor(ntile(variable, 5))``.
    Uses rank-based tiling for stable, deterministic results (ties broken by
    first occurrence, matching dplyr::ntile behavior).

    Args:
        values: Array-like numeric values (list, np.ndarray, or pd.Series).
            May contain NaN; NaN positions are preserved.

    Returns:
        pd.Categorical with ordered integer categories [1, 2, 3, 4, 5].
        NaN values in input produce NaN in output.

    Raises:
        TypeError: If values cannot be cast to float.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    n_total = len(s)
    result = np.full(n_total, np.nan)

    non_null_mask = s.notna()
    n_nonnull = non_null_mask.sum()

    if n_nonnull == 0:
        return pd.Categorical(
            [np.nan] * n_total, categories=[1, 2, 3, 4, 5], ordered=True
        )

    vals = s[non_null_mask].values

    if n_nonnull < 5:
        # Use qcut with duplicates drop and remap to 1..k
        try:
            _, bins = pd.cut(vals, bins=min(5, n_nonnull), retbins=True)  # type: ignore[call-overload]
            codes_raw = pd.cut(vals, bins=bins, labels=False, include_lowest=True)  # type: ignore[call-overload]
            codes = np.asarray(codes_raw, dtype=float)
            tiles = (codes + 1).astype(float)
        except Exception:
            # fallback to rank
            ranks = np.asarray(pd.Series(vals).rank(method="first"), dtype=float)
            tiles = np.floor((ranks - 1) * 5 / n_nonnull).astype(int) + 1
            tiles = np.clip(tiles, 1, 5).astype(float)
    else:
        # Rank-based tiling: equivalent to dplyr::ntile
        ranks = np.asarray(pd.Series(vals).rank(method="first"), dtype=float)
        tiles = np.floor((ranks - 1) * 5 / n_nonnull).astype(int) + 1
        tiles = np.clip(tiles, 1, 5).astype(float)

    result[np.asarray(non_null_mask.values, dtype=bool)] = tiles
    cat = pd.Categorical(result, categories=[1, 2, 3, 4, 5], ordered=True)
    return cat


def qbr(
    df: pd.DataFrame,
    variable: str,
    rnd: bool | None = None,
) -> list[str]:
    """Compute quantile break labels for a numeric column.

    Python equivalent of R ``qBr(df, variable, rnd)``. Returns 5 strings
    representing quantile values at probabilities [0.01, 0.2, 0.4, 0.6, 0.8].

    Args:
        df: DataFrame containing the variable column.
        variable: Column name to compute quantiles for (numeric).
        rnd:
            - ``None`` (default): Round values to 0 decimals before computing
              quantiles (mirrors R's missing-argument branch).
            - ``False``: Use raw values; format with 3 decimal places.
            - ``True``: **Not supported** — R's ``qBr`` does not define behavior
              for ``rnd=TRUE`` (it returns ``NULL``). Raises ``ValueError``.

    Returns:
        List of 5 strings. Returns ``["nan"] * 5`` if all values are null.

    Raises:
        ValueError: If ``rnd=True`` is passed.
    """
    # Bug 2 fix: match R semantics — rnd=TRUE is undefined in R's qBr and
    # effectively returns NULL; raise explicitly rather than silently misbehave.
    if rnd is True:
        raise ValueError(
            "rnd=True is not supported. R's qBr does not define behavior for "
            "rnd=TRUE (it returns NULL). Pass rnd=False for raw-value quantiles "
            "or rnd=None (default) for rounded quantiles."
        )

    probs = [0.01, 0.2, 0.4, 0.6, 0.8]

    x = pd.to_numeric(df[variable], errors="coerce")
    non_null = x.dropna()

    if len(non_null) == 0:
        return ["nan"] * 5

    if rnd is None:
        # R default: quantile(round(x, 0), probs, na.rm=T)
        rounded = non_null.round(0)
        quantiles = rounded.quantile(probs)
        return [f"{v:g}" for v in quantiles]
    else:
        # rnd is False: quantile(x, probs, na.rm=T), format with 3 decimals
        quantiles = non_null.quantile(probs)
        return [f"{v:.3f}" for v in quantiles]


def q5_labels(
    df: pd.DataFrame,
    variable: str,
    *,
    rnd: bool | None = None,
) -> tuple[pd.Categorical, list[str]]:
    """Return q5 bin assignments and matching legend labels for a DataFrame column.

    Combines ``q5`` and ``qbr`` to produce both the category vector and the five
    break-value strings needed to label a quintile map legend in one call.

    Args:
        df: DataFrame containing the variable.
        variable: Column name to bin (numeric).
        rnd: Passed to ``qbr``. ``None`` (default) rounds before quantiles;
            ``False`` uses raw values with 3-decimal formatting.

    Returns:
        Tuple ``(categories, labels)`` where ``categories`` is a
        ``pd.Categorical`` with ordered integer bins [1-5] and ``labels`` is a
        list of 5 strings suitable for legend tick annotations.

    Example::

        cats, labels = q5_labels(gdf, "median_price")
        gdf["price_q5"] = cats
        # use labels as legend tick text
    """
    cats = q5(df[variable])
    labels = qbr(df, variable, rnd=rnd)
    return cats, labels
