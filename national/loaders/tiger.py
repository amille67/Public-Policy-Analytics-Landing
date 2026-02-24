"""TIGER/Line tract boundary loader for spatial joins with ACS attribute data.

Downloads tract-level polygon geometries directly from the Census TIGER REST
API and caches them on disk.  Combine with ``ppa.data.fetch_acs_tracts`` to
build fully spatial GeoDataFrames.
"""

from __future__ import annotations

import logging
import os
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import joblib
import pandas as pd
import requests

from ppa.data import fetch_acs_tracts

logger = logging.getLogger(__name__)

_CACHE_DIR: str = os.environ.get("PPA_CACHE_DIR", str(Path.home() / ".cache" / "ppa"))
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

_TIGER_TRACT_URL: str = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
    "cb_{year}_{state_fips}_tract_500k.zip"
)
_TIGER_BG_URL: str = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
    "cb_{year}_{state_fips}_bg_500k.zip"
)


@_memory.cache  # type: ignore[misc]
def _fetch_tiger_bytes(year: int, state_fips: str, geography: str) -> bytes:
    template = _TIGER_TRACT_URL if geography == "tract" else _TIGER_BG_URL
    url = template.format(year=year, state_fips=state_fips)
    logger.info("Downloading TIGER boundaries: %s", url)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    return resp.content


def _read_tiger_zip(raw_zip: bytes, cache_key: str) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(BytesIO(raw_zip)) as zf:
        shp_names = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_names:
            raise ValueError("No .shp file found in TIGER ZIP archive")

        tmp_dir = Path(_CACHE_DIR) / cache_key
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(str(tmp_dir))
        shp_path = tmp_dir / shp_names[0]
    return gpd.read_file(str(shp_path))


def fetch_tiger_tracts(
    year: int, state_fips: str, *, target_epsg: int = 4326
) -> gpd.GeoDataFrame:
    """Fetch Census TIGER tract polygon geometries for a state.

    Downloads the Census Cartographic Boundary 1:500k shapefile for census
    tracts, caches the raw ZIP, and returns a GeoDataFrame with a ``GEOID``
    column ready to join with ACS attribute tables.
    """
    raw_zip = _fetch_tiger_bytes(year, state_fips, "tract")
    gdf = _read_tiger_zip(raw_zip, f"tiger_tract_{year}_{state_fips}").to_crs(
        epsg=target_epsg
    )
    if "GEOID" not in gdf.columns:
        gdf["GEOID"] = gdf["STATEFP"] + gdf["COUNTYFP"] + gdf["TRACTCE"]
    keep = ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "NAME", "geometry"]
    return gdf[[c for c in keep if c in gdf.columns]]




def _group_fips_by_state(fips_list: list[str]) -> dict[str, set[str]]:
    state_to_counties: dict[str, set[str]] = defaultdict(set)
    for fip in fips_list:
        if len(fip) != 5 or not fip.isdigit():
            raise ValueError(f"Invalid 5-digit FIPS: {fip}")
        state_to_counties[fip[:2]].add(fip[2:])
    return state_to_counties


def tiger_tracts(
    fips_list: list[str] | None = None, *, year: int = 2022
) -> gpd.GeoDataFrame:
    """Return tract polygons for the given county FIPS codes."""
    if fips_list is None:
        fips_list = ["47037"]
    state_to_counties = _group_fips_by_state(fips_list)

    gdfs: list[gpd.GeoDataFrame] = []
    for state_fips, county_fips in state_to_counties.items():
        gdf = fetch_tiger_tracts(year, state_fips)
        gdfs.append(gdf[gdf["COUNTYFP"].isin(county_fips)].copy())

    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "NAME", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    return pd.concat(gdfs, ignore_index=True)


def tiger_block_groups(
    fips_list: list[str] | None = None, *, year: int = 2022
) -> gpd.GeoDataFrame:
    """Return block-group polygons for the given county FIPS codes."""
    if fips_list is None:
        fips_list = ["47037"]
    state_to_counties = _group_fips_by_state(fips_list)

    gdfs: list[gpd.GeoDataFrame] = []
    for state_fips, county_fips in state_to_counties.items():
        raw_zip = _fetch_tiger_bytes(year, state_fips, "bg")
        gdf = _read_tiger_zip(raw_zip, f"tiger_bg_{year}_{state_fips}").to_crs(epsg=4326)
        if "GEOID" not in gdf.columns:
            gdf["GEOID"] = gdf["STATEFP"] + gdf["COUNTYFP"] + gdf["BLKGRPCE"]
        keep = ["GEOID", "STATEFP", "COUNTYFP", "BLKGRPCE", "NAME", "geometry"]
        gdf = gdf[[c for c in keep if c in gdf.columns]]
        gdfs.append(gdf[gdf["COUNTYFP"].isin(county_fips)].copy())

    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["GEOID", "STATEFP", "COUNTYFP", "BLKGRPCE", "NAME", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    return pd.concat(gdfs, ignore_index=True)


def tiger_demographics(
    fips_list: list[str] | None = None, *, year: int = 2022
) -> pd.DataFrame:
    """Return tract-level demographic rates for the given county FIPS codes.

    Fetches four ACS 5-Year variables explicitly so column names are
    predictable regardless of which preset ``fetch_acs_tracts`` uses by default.
    """
    if fips_list is None:
        fips_list = ["47037"]
    state_to_counties = _group_fips_by_state(fips_list)

    dfs: list[pd.DataFrame] = []
    for state_fips, county_fips in state_to_counties.items():
        # Request raw Census codes mapped to stable friendly names.
        acs = fetch_acs_tracts(
            year,
            state_fips,
            variables={
                "B01003_001E": "pop_total",
                "B03002_003E": "pop_non_hisp_white",
                "B17001_002E": "pop_poverty",
                "B08201_002E": "pop_transit_commute",
            },
        )
        acs = acs[acs["COUNTYFP"].isin(county_fips)].copy()

        total = pd.to_numeric(acs["pop_total"], errors="coerce").fillna(0)
        non_hisp_white = pd.to_numeric(acs["pop_non_hisp_white"], errors="coerce").fillna(
            0
        )
        poverty = pd.to_numeric(acs["pop_poverty"], errors="coerce").fillna(0)
        commute_transit = pd.to_numeric(
            acs["pop_transit_commute"], errors="coerce"
        ).fillna(0)

        out = pd.DataFrame({"GEOID": acs["GEOID"].astype(str)})
        out["population"] = total
        out["share_non_hisp_white"] = (non_hisp_white / total.replace(0, pd.NA)).fillna(0)
        out["poverty_rate"] = (poverty / total.replace(0, pd.NA)).fillna(0)
        out["transit_commute_rate"] = (commute_transit / total.replace(0, pd.NA)).fillna(
            0
        )
        dfs.append(out)

    if not dfs:
        return pd.DataFrame(
            columns=[
                "GEOID",
                "population",
                "share_non_hisp_white",
                "poverty_rate",
                "transit_commute_rate",
            ]
        )

    return pd.concat(dfs, ignore_index=True)
