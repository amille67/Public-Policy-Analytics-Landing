"""KPI card rendering helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_kpi_cards(df: pd.DataFrame) -> None:
    """Render summary metrics for current filtered view."""
    n_rows = len(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    metric_col = next((c for c in numeric_cols if "score" in c or "rate" in c), None)
    mean_value = float(df[metric_col].mean()) if metric_col else 0.0
    max_value = float(df[metric_col].max()) if metric_col else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{n_rows:,}")
    c2.metric("Primary metric (mean)", f"{mean_value:,.2f}")
    c3.metric("Primary metric (max)", f"{max_value:,.2f}")
