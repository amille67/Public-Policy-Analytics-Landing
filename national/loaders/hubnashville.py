"""hubNashville/Socrata loaders for Nashville MVP views with spatial tract joins."""

from __future__ import annotations

import os
from typing import Any

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd

from national.loaders.tiger import tiger_tracts

DATASET_IDS = {
    "requests_311": "7qhx-rexi",
    "permits": "3h5w-q8b7",
    "parcels": "p5mt-5q2i",
    "assessor": "wvtk-8qvc",
    "usd": "2k3w-hksv",
}


def _client() -> Any:
    from sodapy import Socrata  # type: ignore[import-untyped]

    return Socrata("data.nashville.gov", os.environ.get("SOCRATA_APP_TOKEN"), timeout=90)


def _records(dataset_id: str, *, limit: int = 50000) -> list[dict[str, Any]]:
    try:
        return list(_client().get(dataset_id, limit=limit))
    except Exception:
        return []


def _to_gdf(records: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    if df.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # 1. Socrata GeoJSON column (the_geom, mapped_location, geometry)
    geom_col = next((c for c in ["the_geom", "mapped_location", "geometry"] if c in df.columns), None)
    if geom_col is not None:
        from shapely.geometry import shape
        def parse_geom(x):
            try:
                return shape(x) if isinstance(x, dict) else None
            except Exception:
                return None
        df["geometry"] = df[geom_col].apply(parse_geom)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        return gdf[gdf.geometry.notna()].copy()

    # 2. Fallback: point columns (existing logic)
    lon_candidates = ["longitude", "lng", "lon", "x"]
    lat_candidates = ["latitude", "lat", "y"]
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    if lon_col and lat_col:
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        valid = lon.notna() & lat.notna()
        if valid.any():
            return gpd.GeoDataFrame(
                df.loc[valid], 
                geometry=gpd.points_from_xy(lon[valid], lat[valid]), 
                crs="EPSG:4326"
            )
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    df = df.loc[valid].copy()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon[valid], lat[valid]), crs="EPSG:4326")


def _attach_tract_geoid(gdf: gpd.GeoDataFrame, fips_list: list[str]) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["tract_geoid"] = pd.Series(dtype=str)
        return gdf
    tracts = tiger_tracts(fips_list)
    joined = gpd.sjoin(gdf, tracts[["GEOID", "geometry"]], how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined["tract_geoid"] = joined["GEOID"].astype(str)
    return joined


def hubnashville_311(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load 311 requests and assign risk category + tract geoid."""
    if fips_list is None:
        fips_list = ["47037"]
    gdf = _to_gdf(_records(DATASET_IDS["requests_311"]))
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["tract_geoid", "risk_category"], geometry=[], crs="EPSG:4326")

    req_type = gdf.get("request_type", pd.Series("", index=gdf.index)).astype(str).str.lower()
    gdf["risk_category"] = req_type.map(
        lambda x: "high" if any(k in x for k in ("junk", "debris", "abandoned")) else "medium"
    )
    out = _attach_tract_geoid(gdf, fips_list)
    return out[["tract_geoid", "risk_category", "geometry"]]


def hubnashville_permits(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load permit records filtered to demolition/rehab and assign tract geoid."""
    if fips_list is None:
        fips_list = ["47037"]
    gdf = _to_gdf(_records(DATASET_IDS["permits"]))
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["tract_geoid", "permit_type"], geometry=[], crs="EPSG:4326")

    permit_col = gdf.get("permit_type", pd.Series("", index=gdf.index)).astype(str)
    # Case-insensitive filter — Nashville API returns mixed-case values.
    gdf = gdf[permit_col.str.lower().isin(["demolition", "rehab"])].copy()
    out = _attach_tract_geoid(gdf, fips_list)
    out["permit_type"] = out.get("permit_type", pd.Series("", index=out.index)).astype(str)
    return out[["tract_geoid", "permit_type", "geometry"]]


def hubnashville_parcels(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load parcel points and assign tract geoid + standardized parcel area."""
    if fips_list is None:
        fips_list = ["47037"]
    gdf = _to_gdf(_records(DATASET_IDS["parcels"]))
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["parcel_id", "tract_geoid", "parcel_area_sqft"], geometry=[], crs="EPSG:4326"
        )

    out = _attach_tract_geoid(gdf, fips_list)
    out["parcel_id"] = out.get("parcel_id", pd.RangeIndex(len(out))).astype(str)
    out["parcel_area_sqft"] = pd.to_numeric(
        out.get("parcel_area_sqft", pd.Series(0, index=out.index)), errors="coerce"
    ).fillna(0)
    return out[["parcel_id", "tract_geoid", "parcel_area_sqft", "geometry"]]


def hubnashville_assessor(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load assessor records (single-family, nonzero valuation) with tract geoid."""
    if fips_list is None:
        fips_list = ["47037"]
    gdf = _to_gdf(_records(DATASET_IDS["assessor"]))
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["tract_geoid", "appraised_value"], geometry=[], crs="EPSG:4326")

    prop_type = gdf.get("property_type", pd.Series("", index=gdf.index)).astype(str)
    gdf = gdf[prop_type.str.contains("single", case=False, na=False)].copy()
    gdf["appraised_value"] = pd.to_numeric(
        gdf.get("appraised_value", pd.Series(0, index=gdf.index)), errors="coerce"
    ).fillna(0)
    gdf = gdf[gdf["appraised_value"] > 0].copy()
    out = _attach_tract_geoid(gdf, fips_list)
    return out[["tract_geoid", "appraised_value", "geometry"]]


def hubnashville_usd(fips_list: list[str] | None = None) -> gpd.GeoDataFrame:
    """Load USD features and assign intersecting tract geoid."""
    if fips_list is None:
        fips_list = ["47037"]
    gdf = _to_gdf(_records(DATASET_IDS["usd"]))
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["tract_geoid"], geometry=[], crs="EPSG:4326")

    out = _attach_tract_geoid(gdf, fips_list)
    return out[["tract_geoid", "geometry"]].drop_duplicates()
