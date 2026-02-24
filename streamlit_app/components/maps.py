"""Pydeck rendering for mixed polygon/point geospatial layers."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st


def _metric_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = ["risk_score", "proximity_score", "equity_gap", "hpi_yoy"]
    for col in candidates:
        if col in gdf.columns:
            return col
    score_like = [c for c in gdf.columns if "score" in c or "rate" in c or "count" in c]
    return score_like[0] if score_like else None


def _normalize_colors(gdf: gpd.GeoDataFrame, metric_col: str) -> gpd.GeoDataFrame:
    series = gdf[metric_col].astype(float)
    min_v = float(series.min(skipna=True) or 0)
    max_v = float(series.max(skipna=True) or 1)
    denom = (max_v - min_v) if max_v != min_v else 1.0

    def _get_color(x):
        if pd.isna(x):
            return [150, 150, 150, 100]  # neutral gray for points
        val = float(x)
        return [
            int(255 * ((val - min_v) / denom)),
            60,
            int(255 - 255 * ((val - min_v) / denom)),
            220,
        ]

    gdf = gdf.copy()
    gdf["__color"] = series.apply(_get_color)
    return gdf


def render_view_map(gdf: gpd.GeoDataFrame | pd.DataFrame, view_name: str) -> None:
    """Render mixed geospatial view with polygons and optional points."""
    if gdf.empty:
        st.warning("No records to display.")
        return

    if not isinstance(gdf, gpd.GeoDataFrame):
        st.info("🌍 This view contains non-spatial tabular data. Map rendering is disabled.")
        return

    map_gdf = gdf.to_crs("EPSG:4326").copy()
    has_points = map_gdf.geometry.geom_type.astype(str).str.contains("Point").any()

    layers: list[pdk.Layer] = []

    metric_col = _metric_column(map_gdf)
    if metric_col is not None:
        map_gdf = _normalize_colors(map_gdf, metric_col)

    # Polygons
    poly = map_gdf[
        map_gdf.geometry.geom_type.astype(str).isin(["Polygon", "MultiPolygon"])
    ].copy()
    if not poly.empty:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=poly.__geo_interface__,
                get_fill_color=(
                    "properties.__color"
                    if "__color" in poly.columns
                    else [66, 135, 245, 120]
                ),
                get_line_color=[30, 58, 138, 200],
                line_width_min_pixels=1,
                pickable=True,
                opacity=0.75,
            )
        )

    # Points (View 1 311 complaints, etc.)
    if has_points:
        pts = map_gdf[map_gdf.geometry.geom_type.astype(str) == "Point"].copy()
        if not pts.empty:
            pts["lon"] = pts.geometry.x
            pts["lat"] = pts.geometry.y
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pts,
                    get_position="[lon, lat]",
                    get_radius=120,
                    radius_min_pixels=2,
                    get_fill_color=[255, 100, 0, 180],
                    pickable=True,
                )
            )

    view_state = pdk.ViewState(latitude=35.86, longitude=-86.35, zoom=6.8, pitch=0)

    tooltip = {
        "html": (
            f"""
            <b>GEOID:</b> {{GEOID}}<br>
            <b>Value:</b> {{{metric_col}}}
            """
            if metric_col
            else "<b>Point record</b>"
        )
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v11",
            tooltip=tooltip,
        ),
        use_container_width=True,
    )
