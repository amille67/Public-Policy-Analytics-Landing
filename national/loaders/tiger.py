"""TIGER/Line tract boundary loader for spatial joins with ACS attribute data.

Downloads tract-level polygon geometries directly from the Census TIGER REST
API and caches them on disk.  Combine with ``ppa.data.fetch_acs_tracts`` to
build fully spatial GeoDataFrames.

References
----------
- TIGER/Line Shapefiles: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- Census Cartographic Boundary Files: https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
"""

from __future__ import annotations

import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import joblib
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR: str = os.environ.get("PPA_CACHE_DIR", str(Path.home() / ".cache" / "ppa"))
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

# Census Cartographic Boundary File URL (5m generalised — good for viz)
_TIGER_URL: str = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
    "cb_{year}_{state_fips}_tract_500k.zip"
)


@_memory.cache  # type: ignore[misc]
def _fetch_tiger_bytes(year: int, state_fips: str) -> bytes:
    """Download TIGER tract shapefile ZIP as bytes.

    Parameters
    ----------
    year : int
        Vintage year (e.g. 2022).
    state_fips : str
        Two-digit state FIPS code.

    Returns
    -------
    bytes
        Raw ZIP archive content.

    Raises
    ------
    requests.HTTPError
        On non-2xx HTTP status.
    """
    url = _TIGER_URL.format(year=year, state_fips=state_fips)
    logger.info("Downloading TIGER tract boundaries: %s", url)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    return resp.content


def fetch_tiger_tracts(
    year: int,
    state_fips: str,
    *,
    target_epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """Fetch Census TIGER tract polygon geometries for a state.

    Downloads the Census Cartographic Boundary 1:500k shapefile for census
    tracts, caches the raw ZIP, and returns a GeoDataFrame with a ``GEOID``
    column ready to join with ACS attribute tables.

    Parameters
    ----------
    year : int
        Vintage year matching the ACS data (e.g. 2022).
    state_fips : str
        Two-digit state FIPS code (e.g. ``"42"`` for Pennsylvania).
    target_epsg : int
        Output CRS.  Defaults to 4326 for storage.  Pass a projected EPSG
        (e.g. 26918) for distance/area calculations.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``GEOID``, ``STATEFP``, ``COUNTYFP``, ``TRACTCE``,
        ``NAME``, ``geometry`` (polygons in ``target_epsg``).

    Raises
    ------
    requests.HTTPError
        If the TIGER download fails.

    Examples
    --------
    >>> tiger = fetch_tiger_tracts(2022, "42")        # doctest: +SKIP
    >>> acs   = fetch_acs_tracts(2022, "42")           # doctest: +SKIP
    >>> gdf   = tiger.merge(acs.drop(columns="geometry"), on="GEOID")  # doctest: +SKIP
    """
    raw_zip = _fetch_tiger_bytes(year, state_fips)

    with zipfile.ZipFile(BytesIO(raw_zip)) as zf:
        # Find the .shp entry
        shp_names = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_names:
            raise ValueError("No .shp file found in TIGER ZIP archive")

        # Extract all members to a temp dir in the cache
        tmp_dir = Path(_CACHE_DIR) / f"tiger_{year}_{state_fips}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(str(tmp_dir))
        shp_path = tmp_dir / shp_names[0]

    gdf: gpd.GeoDataFrame = gpd.read_file(str(shp_path))

    # Normalise GEOID: state(2) + county(3) + tract(6)
    if "GEOID" not in gdf.columns:
        gdf["GEOID"] = gdf["STATEFP"] + gdf["COUNTYFP"] + gdf["TRACTCE"]

    gdf = gdf.to_crs(epsg=target_epsg)

    keep = ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "NAME", "geometry"]
    existing = [c for c in keep if c in gdf.columns]
    gdf = gdf[existing]

    logger.info(
        "Loaded %d tract polygons for state=%s year=%d (EPSG:%d)",
        len(gdf),
        state_fips,
        year,
        target_epsg,
    )
    return gdf
