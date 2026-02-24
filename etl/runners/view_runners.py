"""Thin runners for the seven Nashville MVP views with tract-level joins."""

from __future__ import annotations

from typing import Any

import pandas as pd

from national.loaders import housing, hubnashville, osm, tiger

DAVIDSON_FIPS = ["47037"]


def _counts_by_tract(df: pd.DataFrame, key: str = "tract_geoid") -> pd.Series:
    if df.empty or key not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby(key).size()


def _as_wgs84(df: Any) -> Any:
    return df.to_crs("EPSG:4326") if hasattr(df, "to_crs") else df


def run_view_01_nashville_311_risk() -> Any:
    """View 1: 311 risk (tract count + demographics)."""
    complaints = hubnashville.hubnashville_311(DAVIDSON_FIPS)
    tracts = tiger.tiger_tracts(DAVIDSON_FIPS)
    demos = tiger.tiger_demographics(DAVIDSON_FIPS)
    out = tracts.merge(demos, on="GEOID", how="left")
    out["risk_score"] = out["GEOID"].map(_counts_by_tract(complaints, key="tract_geoid")).fillna(0)
    return _as_wgs84(out)


def run_view_02_nashville_permits_transition() -> Any:
    permits = hubnashville.hubnashville_permits(DAVIDSON_FIPS)
    tracts = tiger.tiger_tracts(DAVIDSON_FIPS)
    out = tracts.copy()
    out["permit_transition_count"] = out["GEOID"].map(_counts_by_tract(permits)).fillna(0)
    return _as_wgs84(out)


def run_view_03_nashville_property_value_equity() -> Any:
    assessor = hubnashville.hubnashville_assessor(DAVIDSON_FIPS)
    tracts = tiger.tiger_tracts(DAVIDSON_FIPS)
    tract_stats = assessor.groupby("tract_geoid", as_index=False)["appraised_value"].median()
    tract_stats = tract_stats.rename(
        columns={"tract_geoid": "GEOID", "appraised_value": "median_assessed_value"}
    )
    out = tracts.merge(tract_stats, on="GEOID", how="left")
    out["median_assessed_value"] = out["median_assessed_value"].fillna(0)
    return _as_wgs84(out)


def run_view_04_nashville_urban_services_district() -> Any:
    tracts = tiger.tiger_tracts(DAVIDSON_FIPS)
    usd = hubnashville.hubnashville_usd(DAVIDSON_FIPS)
    parcels = hubnashville.hubnashville_parcels(DAVIDSON_FIPS)
    out = tracts.copy()
    out["parcel_count"] = out["GEOID"].map(_counts_by_tract(parcels)).fillna(0)
    usd_flag = set(usd.get("tract_geoid", pd.Series(dtype=str)).astype(str).tolist())
    out["is_usd"] = out["GEOID"].astype(str).isin(usd_flag)
    return _as_wgs84(out)


def run_view_05_nashville_industrial_footprints() -> Any:
    tracts = tiger.tiger_tracts(DAVIDSON_FIPS)
    industrial = osm.osm_industrial_footprints(DAVIDSON_FIPS)
    out = tracts.copy()
    out["industrial_site_count"] = out["GEOID"].map(_counts_by_tract(industrial)).fillna(0)
    return _as_wgs84(out)


def run_view_06_nashville_transit_access_equity() -> Any:
    tracts = tiger.tiger_tracts(DAVIDSON_FIPS)
    stops = osm.osm_transit_stops(DAVIDSON_FIPS)
    demos = tiger.tiger_demographics(DAVIDSON_FIPS)
    out = tracts.merge(demos, on="GEOID", how="left")
    out["transit_stop_count"] = out["GEOID"].map(_counts_by_tract(stops)).fillna(0)
    return _as_wgs84(out)


def run_view_07_national_housing_context() -> Any:
    hpi = housing.fhfa_hpi(["47037"])
    hud = housing.hud_county(["47037"])
    out = hpi.merge(hud, on="county_fips", how="left")
    out["is_nashville"] = out["county_fips"].eq("47037")
    return out
