"""TN Public Policy Analytics app shell and primary explorer."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.cards import render_kpi_cards
from streamlit_app.components.charts import render_numeric_distribution
from streamlit_app.components.maps import render_view_map
from streamlit_app.core.state import init_session_state
from streamlit_app.core.theme import inject_premium_theme
from streamlit_app.data.loaders import get_view_metadata, load_view

st.set_page_config(page_title="TN Public Policy Analytics", page_icon="🏛️", layout="wide")

init_session_state()
inject_premium_theme()
state = st.session_state.app_state

with st.sidebar:
    st.markdown("# TN Policy Analytics")
    metadata = get_view_metadata()
    view_options = {v["name"]: k for k, v in metadata.items()}

    state.selected_view = st.selectbox("Select High-Value View", list(view_options.keys()))
    view_id = view_options[state.selected_view]

    levels = ["lowest", "county"] if metadata[view_id]["default_level"] == "lowest" else ["county"]
    state.selected_geo_level = st.radio("Granularity", levels, horizontal=True)

    st.caption(f"ETL Vintage: 2024 | Chapter {metadata[view_id]['chapter']}")

st.title(state.selected_view)

with st.spinner("Loading geospatial layer..."):
    gdf = load_view(view_id, state.selected_geo_level)

if gdf.empty:
    st.stop()

render_kpi_cards(gdf)

left, right = st.columns([7, 3])
with left:
    render_view_map(gdf, state.selected_view)
with right:
    st.subheader("Key Insights")
    st.markdown(
        """
- Pre-computed from official ETL pipelines.
- Click map features for details.
- Data served at **lowest** or **county** level.
        """
    )
    geojson_bytes = gdf.to_json().encode("utf-8")
    st.download_button(
        "Export Current View (GeoJSON)",
        data=geojson_bytes,
        file_name=f"view_{view_id}_{state.selected_geo_level}.geojson",
        mime="application/geo+json",
    )

with st.expander("Methodology & Data Sources"):
    st.markdown(
        "Built on the 7 highest-value views from Ken Steif's Public Policy Analytics plus Nashville open data."
    )

st.subheader("Distribution")
render_numeric_distribution(gdf)
