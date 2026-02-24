"""Shared data loaders for the PPA package.

Provides cached ACS tract fetching via the Census Bureau API and CRS
standardization utilities for GeoDataFrames / GeoJSON files.

Caching
-------
API responses are cached on disk via ``joblib.Memory`` to avoid redundant
network requests.  The default cache directory is ``~/.cache/ppa``, which
can be overridden by setting the ``PPA_CACHE_DIR`` environment variable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import joblib
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache setup
# ---------------------------------------------------------------------------

_CACHE_DIR: str = os.environ.get("PPA_CACHE_DIR", str(Path.home() / ".cache" / "ppa"))
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

# Default storage CRS — WGS 84 geographic
_DEFAULT_STORAGE_EPSG: int = 4326

# Census Bureau ACS 5-Year API base URL
_ACS5_BASE_URL: str = "https://api.census.gov/data/{year}/acs/acs5"


# ---------------------------------------------------------------------------
# ACS variable presets
# ---------------------------------------------------------------------------

#: Common variable bundles keyed by short name.  Each maps a Census variable
#: code to a human-readable column name.
ACS_VARIABLE_PRESETS: dict[str, dict[str, str]] = {
    "demographics": {
        "B01003_001E": "total_pop",
        "B02001_002E": "pop_white",
        "B03001_003E": "pop_hispanic",
        "B01002_001E": "median_age",
    },
    "housing": {
        "B25077_001E": "median_home_value",
        "B25064_001E": "median_rent",
        "B25003_001E": "total_tenure",
        "B25003_002E": "tenure_owner",
        "B25003_003E": "tenure_renter",
    },
    "income": {
        "B19013_001E": "median_hh_income",
        "B17001_002E": "pop_below_poverty",
    },
    "education": {
        "B15003_022E": "pop_bachelors",
        "B15003_023E": "pop_masters",
        "B15003_025E": "pop_doctorate",
    },
}


# ---------------------------------------------------------------------------
# Internal: cached raw API fetch
# ---------------------------------------------------------------------------


@_memory.cache  # type: ignore[misc]
def _fetch_acs_raw(
    year: int,
    variables: tuple[str, ...],
    state_fips: str,
    county_fips: str | None,
    api_key: str | None,
) -> list[list[str]]:
    """Fetch raw ACS 5-Year data from the Census Bureau API.

    Parameters
    ----------
    year : int
        ACS vintage year (e.g. 2022).
    variables : tuple[str, ...]
        Census variable codes to request.  Passed as a tuple for hashability.
    state_fips : str
        Two-digit state FIPS code (e.g. ``"42"`` for Pennsylvania).
    county_fips : str or None
        Three-digit county FIPS code, or ``None`` for all counties in the state.
    api_key : str or None
        Census API key.  Optional but recommended.

    Returns
    -------
    list[list[str]]
        Raw JSON rows including the header row.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status.
    ValueError
        If the response is empty or not valid JSON.
    """
    url = _ACS5_BASE_URL.format(year=year)

    get_vars = ",".join(["NAME", *variables])

    if county_fips is not None:
        for_clause = f"tract:*&in=state:{state_fips}&in=county:{county_fips}"
    else:
        for_clause = f"tract:*&in=state:{state_fips}"

    params: dict[str, str] = {
        "get": get_vars,
        "for": for_clause,
    }
    if api_key:
        params["key"] = api_key

    logger.info(
        "Fetching ACS %d data: state=%s county=%s (%d variables)",
        year,
        state_fips,
        county_fips or "*",
        len(variables),
    )

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    data: list[list[str]] = resp.json()
    if not data or len(data) < 2:
        raise ValueError(
            f"ACS API returned empty or header-only response for "
            f"state={state_fips}, county={county_fips}, year={year}"
        )

    logger.info("Received %d tract rows from Census API", len(data) - 1)
    return data


@retry(
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_acs_raw_with_retry(
    year: int,
    variables: tuple[str, ...],
    state_fips: str,
    county_fips: str | None,
    api_key: str | None,
) -> list[list[str]]:
    """Fetch raw ACS data with exponential-backoff retries."""
    return _fetch_acs_raw(
        year=year,
        variables=variables,
        state_fips=state_fips,
        county_fips=county_fips,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Public: fetch_acs_tracts
# ---------------------------------------------------------------------------


def fetch_acs_tracts(
    year: int,
    state_fips: str,
    *,
    county_fips: str | None = None,
    variables: dict[str, str] | None = None,
    preset: str | None = "demographics",
    api_key: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch ACS 5-Year tract-level data and return as a GeoDataFrame.

    Data is cached on disk via ``joblib`` so repeated calls with the same
    parameters hit the local cache instead of the network.

    Parameters
    ----------
    year : int
        ACS vintage year (e.g. 2022).  Must be >= 2009.
    state_fips : str
        Two-digit state FIPS code (e.g. ``"42"`` for PA).
    county_fips : str or None
        Three-digit county FIPS, or ``None`` for all counties in the state.
    variables : dict[str, str] or None
        Mapping of Census variable codes to desired column names.  If
        ``None``, falls back to the ``preset`` bundle.
    preset : str or None
        Name of a built-in variable bundle from :data:`ACS_VARIABLE_PRESETS`.
        Ignored when ``variables`` is provided explicitly.  Defaults to
        ``"demographics"``.
    api_key : str or None
        Census API key.  Falls back to the ``CENSUS_API_KEY`` env var.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per tract with a ``GEOID`` column, the requested ACS
        variables (renamed), and tract polygon geometry (EPSG:4326).

    Raises
    ------
    ValueError
        If both ``variables`` and ``preset`` are ``None``, or the preset
        name is unknown.

    Examples
    --------
    >>> gdf = fetch_acs_tracts(2022, "42", county_fips="101")
    >>> gdf.columns.tolist()  # doctest: +SKIP
    ['GEOID', 'NAME', 'total_pop', 'pop_white', ..., 'geometry']
    """
    # Resolve API key
    resolved_key = api_key or os.environ.get("CENSUS_API_KEY")

    # Resolve variables
    var_map: dict[str, str]
    if variables is not None:
        var_map = variables
    elif preset is not None:
        if preset not in ACS_VARIABLE_PRESETS:
            raise ValueError(
                f"Unknown ACS preset {preset!r}. "
                f"Available: {list(ACS_VARIABLE_PRESETS)}"
            )
        var_map = ACS_VARIABLE_PRESETS[preset]
    else:
        raise ValueError("Either 'variables' or 'preset' must be provided")

    census_codes = tuple(var_map.keys())

    # Fetch (cached)
    raw_rows = _fetch_acs_raw_with_retry(
        year=year,
        variables=census_codes,
        state_fips=state_fips,
        county_fips=county_fips,
        api_key=resolved_key,
    )

    # Parse into DataFrame
    header = raw_rows[0]
    df = pd.DataFrame(raw_rows[1:], columns=header)

    # Build GEOID: state + county + tract
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # Rename Census codes → friendly names
    df = df.rename(columns=var_map)

    # Convert numeric columns
    friendly_names = list(var_map.values())
    for col in friendly_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only useful columns
    keep_cols = ["GEOID", "NAME", *friendly_names]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    from national.loaders.tiger import fetch_tiger_tracts  # lazy to avoid circular import

    tiger_gdf = fetch_tiger_tracts(year=year, state_fips=state_fips, target_epsg=_DEFAULT_STORAGE_EPSG)
    tiger_gdf["GEOID"] = tiger_gdf["GEOID"].astype(str)
    df["GEOID"] = df["GEOID"].astype(str)

    gdf = tiger_gdf.merge(df, on="GEOID", how="inner", suffixes=("", "_acs"))

    if not gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("ACS tract join produced non-polygon geometries; TIGER merge failed")

    gdf = standardize_crs(gdf, target_epsg=_DEFAULT_STORAGE_EPSG)

    logger.info(
        "Loaded %d ACS tracts for state=%s county=%s year=%d",
        len(gdf),
        state_fips,
        county_fips or "*",
        year,
    )
    return gdf


# ---------------------------------------------------------------------------
# Public: standardize_crs
# ---------------------------------------------------------------------------


def standardize_crs(
    gdf: gpd.GeoDataFrame,
    *,
    target_epsg: int = _DEFAULT_STORAGE_EPSG,
    source_epsg: int | None = None,
) -> gpd.GeoDataFrame:
    """Standardize the CRS of a GeoDataFrame.

    By default, converts to EPSG:4326 (WGS 84) for storage and
    interoperability.  Pass a projected EPSG (e.g. a state-plane code)
    for distance/area calculations.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with a geometry column.
    target_epsg : int
        Target EPSG code.  Defaults to 4326.
    source_epsg : int or None
        If the GeoDataFrame has no CRS set, assume this EPSG before
        reprojecting.  Defaults to ``None``, which will raise if CRS
        is missing.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame reprojected to ``target_epsg``.

    Raises
    ------
    ValueError
        If ``gdf`` has no CRS and ``source_epsg`` is not provided.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs="EPSG:26918")
    >>> result = standardize_crs(gdf, target_epsg=4326)
    >>> result.crs.to_epsg()
    4326
    """
    if gdf.crs is None:
        if source_epsg is not None:
            logger.warning(
                "GeoDataFrame has no CRS; assuming EPSG:%d before reprojection",
                source_epsg,
            )
            gdf = gdf.set_crs(epsg=source_epsg)
        else:
            raise ValueError(
                "GeoDataFrame has no CRS. Set 'source_epsg' to specify the "
                "assumed coordinate reference system."
            )

    current_epsg: int | None = gdf.crs.to_epsg()

    if current_epsg == target_epsg:
        logger.debug("GeoDataFrame already in EPSG:%d — no reprojection needed", target_epsg)
        return gdf

    logger.info("Reprojecting from EPSG:%s to EPSG:%d", current_epsg, target_epsg)
    result: gpd.GeoDataFrame = gdf.to_crs(epsg=target_epsg)
    return result


# ---------------------------------------------------------------------------
# Public: load_geojson
# ---------------------------------------------------------------------------


def load_geojson(
    path: str | Path,
    *,
    target_epsg: int = _DEFAULT_STORAGE_EPSG,
) -> gpd.GeoDataFrame:
    """Load a GeoJSON file and standardize its CRS.

    Parameters
    ----------
    path : str or Path
        Path to the GeoJSON file.
    target_epsg : int
        Target EPSG code for the output.  Defaults to 4326.

    Returns
    -------
    geopandas.GeoDataFrame
        Loaded and reprojected GeoDataFrame.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {p}")

    gdf: gpd.GeoDataFrame = gpd.read_file(str(p))
    logger.info("Loaded %d features from %s (CRS: %s)", len(gdf), p.name, gdf.crs)

    return standardize_crs(gdf, target_epsg=target_epsg, source_epsg=4326)
