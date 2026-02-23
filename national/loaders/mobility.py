"""Mobility space-time panel helpers for tract-level interval aggregation."""

from __future__ import annotations

import pandas as pd


def build_spacetime_panel(
    pings: pd.DataFrame,
    *,
    geoid_col: str = "GEOID",
    timestamp_col: str = "timestamp",
    interval: str = "15min",
    agg_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate mobility pings into GEOID x time buckets.

    Parameters
    ----------
    interval:
        Any pandas frequency string, e.g. ``"15min"`` or ``"1H"``.
    """
    if geoid_col not in pings.columns:
        raise ValueError(f"Missing required column: {geoid_col}")
    if timestamp_col not in pings.columns:
        raise ValueError(f"Missing required column: {timestamp_col}")

    df = pings.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df["time_bin"] = df[timestamp_col].dt.floor(interval)

    if agg_col is None:
        out = (
            df.groupby([geoid_col, "time_bin"]).size().rename("ping_count").reset_index()
        )
    else:
        if agg_col not in df.columns:
            raise ValueError(f"Missing aggregation column: {agg_col}")
        out = (
            df.groupby([geoid_col, "time_bin"])[agg_col]
            .sum(min_count=1)
            .rename(f"sum_{agg_col}")
            .reset_index()
        )
    return out.sort_values([geoid_col, "time_bin"]).reset_index(drop=True)
