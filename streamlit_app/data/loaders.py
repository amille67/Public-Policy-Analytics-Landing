"""Load pre-computed GeoParquet views for the TN dashboard."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

VIEW_DIR = Path("data/views")


@st.cache_data(ttl=3600, show_spinner="Loading view...")
def load_view(view_id: str, geo_level: str = "lowest") -> gpd.GeoDataFrame:
    file = VIEW_DIR / f"view_{view_id}_{geo_level}.geoparquet"
    if not file.exists():
        st.error(f"View file missing: {file}")
        return gpd.GeoDataFrame()

    df = pd.read_parquet(file)
    if df.empty:
        return gpd.GeoDataFrame()

    if "geometry" in df.columns:
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
    else:
        gdf = gpd.GeoDataFrame(df)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.to_crs("EPSG:4326")


@st.cache_data
def get_view_metadata() -> dict[str, dict[str, str | int]]:
    """Metadata map for the seven high-value views."""
    return {
        "01": {
            "name": "Nashville 311 Distress Risk",
            "chapter": 5,
            "default_level": "lowest",
        },
        "02": {
            "name": "Nashville Permits Transition",
            "chapter": 2,
            "default_level": "lowest",
        },
        "03": {
            "name": "Nashville Property Value Equity",
            "chapter": 3,
            "default_level": "lowest",
        },
        "04": {
            "name": "Nashville Urban Services District",
            "chapter": 2,
            "default_level": "lowest",
        },
        "05": {
            "name": "Nashville Industrial Footprints",
            "chapter": 5,
            "default_level": "lowest",
        },
        "06": {
            "name": "Nashville Transit Access Equity",
            "chapter": 1,
            "default_level": "lowest",
        },
        "07": {
            "name": "National Housing Context (TN Highlight)",
            "chapter": "National",
            "default_level": "county",
        },
    }
