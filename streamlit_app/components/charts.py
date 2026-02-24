"""Altair chart helpers for trend/distribution diagnostics."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def render_numeric_distribution(df: pd.DataFrame) -> None:
    """Render a compact bar chart for the first numeric column."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("No numeric columns available for charting.")
        return

    col = numeric_cols[0]
    sample = df[[col]].dropna().copy()
    if sample.empty:
        st.info("No non-null values available for charting.")
        return

    sample["bin"] = pd.cut(sample[col], bins=10).astype(str)
    counts = sample.groupby("bin", as_index=False).size().rename(columns={"size": "count"})

    chart = (
        alt.Chart(counts)
        .mark_bar(color="#1e40af")
        .encode(x=alt.X("bin:N", sort=None, title=col), y=alt.Y("count:Q", title="Count"))
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
