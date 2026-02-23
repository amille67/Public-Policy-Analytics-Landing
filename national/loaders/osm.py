"""OpenStreetMap risk-proxy loader for tract-level aggregation."""

from __future__ import annotations

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd


def _chunk_polygons(tracts_wgs84: gpd.GeoDataFrame) -> list:
    """Create bounded polygon chunks for Overpass queries."""
    if "COUNTYFP" in tracts_wgs84.columns:
        dissolved = tracts_wgs84[["COUNTYFP", "geometry"]].dissolve(by="COUNTYFP")
        return dissolved.geometry.tolist()

    # fallback: fixed-size chunks by tract order
    chunk_size = 250
    geoms = tracts_wgs84.geometry.tolist()
    return [gpd.GeoSeries(geoms[i : i + chunk_size]).union_all() for i in range(0, len(geoms), chunk_size)]


def fetch_osm_risk_proxies(
    tracts_gdf: gpd.GeoDataFrame,
    *,
    tags: dict[str, str] | None = None,
) -> gpd.GeoDataFrame:
    """Aggregate OSM proxy features (e.g. alcohol shops, ruins) by tract GEOID.

    Queries are spatially chunked to avoid Overpass memory/time limits.
    """
    try:
        import osmnx as ox  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - runtime env specific
        raise ImportError("osmnx is required for OSM risk-proxy loading") from exc

    if "GEOID" not in tracts_gdf.columns:
        raise ValueError("tracts_gdf must include GEOID column")

    requested_tags = tags or {"shop": "alcohol", "building": "ruins"}
    tracts_wgs84 = tracts_gdf[["GEOID", "geometry", *(["COUNTYFP"] if "COUNTYFP" in tracts_gdf.columns else [])]].to_crs(epsg=4326)

    feature_frames: list[gpd.GeoDataFrame] = []
    for chunk_polygon in _chunk_polygons(tracts_wgs84):
        if chunk_polygon is None or chunk_polygon.is_empty:
            continue
        features = ox.features_from_polygon(chunk_polygon, tags=requested_tags)
        if isinstance(features, pd.DataFrame) and not features.empty:
            feature_frames.append(gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326"))

    if not feature_frames:
        return tracts_gdf.assign(osm_risk_count=0)

    feats = pd.concat(feature_frames, ignore_index=True)
    feats = gpd.GeoDataFrame(feats, geometry="geometry", crs="EPSG:4326")
    joined = gpd.sjoin(
        feats[["geometry"]],
        tracts_wgs84[["GEOID", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    counts = joined.groupby("GEOID").size().rename("osm_risk_count").reset_index()
    out = tracts_gdf.merge(counts, on="GEOID", how="left")
    out["osm_risk_count"] = out["osm_risk_count"].fillna(0).astype(int)
    return out
