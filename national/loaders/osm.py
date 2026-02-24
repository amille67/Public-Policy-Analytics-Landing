"""OpenStreetMap loaders for Nashville transit and industrial views."""

from __future__ import annotations

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd

from national.loaders.tiger import tiger_tracts


def _tract_join(points_or_polys: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(points_or_polys, tracts[["GEOID", "geometry"]], how="left", predicate="intersects")
    joined["tract_geoid"] = joined["GEOID"].astype(str)
    return joined.drop(columns=["index_right"], errors="ignore")


def osm_transit_stops(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Return bus stop points with tract_geoid for the given county FIPS codes."""
    if fips_list is None:
        fips_list = ["47037"]
    import osmnx as ox  # type: ignore[import-untyped]

    tracts = tiger_tracts(fips_list)
    search_area = tracts.to_crs("EPSG:4326").union_all()
    stops = ox.features_from_polygon(search_area, tags={"highway": "bus_stop"})
    stops = gpd.GeoDataFrame(stops, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)
    if stops.empty:
        return gpd.GeoDataFrame({"tract_geoid": []}, geometry=[], crs="EPSG:4326")
    return _tract_join(stops[["geometry"]], tracts.to_crs("EPSG:4326"))


def osm_industrial_footprints(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Return industrial building footprints >5000 m² with tract_geoid."""
    if fips_list is None:
        fips_list = ["47037"]
    import osmnx as ox  # type: ignore[import-untyped]

    tracts = tiger_tracts(fips_list)
    search_area = tracts.to_crs("EPSG:4326").union_all()
    buildings = ox.features_from_polygon(search_area, tags={"building": ["industrial", "warehouse"]})
    gdf = gpd.GeoDataFrame(buildings, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)
    if gdf.empty:
        return gpd.GeoDataFrame({"tract_geoid": []}, geometry=[], crs="EPSG:4326")

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf["footprint_area_m2"] = gdf.to_crs("EPSG:5070").geometry.area
    gdf = gdf[gdf["footprint_area_m2"] > 5000].copy()
    return _tract_join(gdf[["geometry", "footprint_area_m2"]], tracts.to_crs("EPSG:4326"))


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
    except ImportError as exc:  # pragma: no cover
        raise ImportError("osmnx is required for OSM risk-proxy loading") from exc

    if "GEOID" not in tracts_gdf.columns:
        raise ValueError("tracts_gdf must include GEOID column")

    requested_tags = tags or {"shop": "alcohol", "building": "ruins"}

    # Chunk by county to stay within Overpass API memory/time limits.
    tracts_wgs84 = tracts_gdf[
        ["GEOID", "geometry", *(["COUNTYFP"] if "COUNTYFP" in tracts_gdf.columns else [])]
    ].to_crs(epsg=4326)

    if "COUNTYFP" in tracts_wgs84.columns:
        chunks = tracts_wgs84[["COUNTYFP", "geometry"]].dissolve(by="COUNTYFP").geometry.tolist()
    else:
        chunk_size = 250
        geoms = tracts_wgs84.geometry.tolist()
        chunks = [
            gpd.GeoSeries(geoms[i : i + chunk_size]).union_all()
            for i in range(0, len(geoms), chunk_size)
        ]

    feature_frames: list[gpd.GeoDataFrame] = []
    for chunk_polygon in chunks:
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
