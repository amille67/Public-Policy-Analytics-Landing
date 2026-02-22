"""BLS QCEW loader for national employment and wage benchmarks.

Caches responses on disk with ``joblib.Memory`` so repeated calls are instant.

References
----------
- BLS QCEW API: https://www.bls.gov/cew/about_data/downloadable_file_layouts/quarterly/naics-based-quarterly-layout.htm
- JOLTS API: https://www.bls.gov/developers/api_python.htm
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR: str = os.environ.get("PPA_CACHE_DIR", str(Path.home() / ".cache" / "ppa"))
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

# BLS QCEW CSV download URL template
_QCEW_URL: str = (
    "https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area_fips}.csv"
)

# BLS Public Data API v2 (for JOLTS series)
_BLS_API_URL: str = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


@_memory.cache  # type: ignore[misc]
def _fetch_qcew_raw(year: int, quarter: str, area_fips: str) -> str:
    """Download raw QCEW CSV from BLS.

    Parameters
    ----------
    year : int
        Reference year (e.g. 2022).
    quarter : str
        Quarter: ``"1"``, ``"2"``, ``"3"``, ``"4"``, or ``"a"`` (annual).
    area_fips : str
        5-character area FIPS code (e.g. ``"42101"`` for Philadelphia County).

    Returns
    -------
    str
        Raw CSV text.

    Raises
    ------
    requests.HTTPError
        On non-2xx HTTP status.
    """
    url = _QCEW_URL.format(year=year, qtr=quarter, area_fips=area_fips)
    logger.info("Fetching BLS QCEW: %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.text


def fetch_qcew(
    year: int,
    area_fips: str,
    *,
    quarter: str = "a",
    naics_filter: str | None = None,
    ownership: str = "5",
) -> pd.DataFrame:
    """Fetch BLS QCEW employment and wage data for a county or MSA.

    Parameters
    ----------
    year : int
        Reference year (e.g. 2022).
    area_fips : str
        5-character area FIPS code.  Use ``"US000"`` for national totals,
        ``"42101"`` for Philadelphia County PA, etc.
    quarter : str
        ``"1"``-``"4"`` for quarterly, or ``"a"`` for annual (default).
    naics_filter : str or None
        Keep only rows whose ``industry_code`` starts with this prefix
        (e.g. ``"44"`` for retail trade).  ``None`` returns all industries.
    ownership : str
        ``"5"`` = private (default), ``"0"`` = all, ``"1"`` = federal govt.

    Returns
    -------
    pandas.DataFrame
        Columns include ``area_fips``, ``industry_code``, ``own_code``,
        ``annual_avg_estabs``, ``annual_avg_emplvl``, ``total_annual_wages``,
        ``avg_annual_pay``.

    Raises
    ------
    requests.HTTPError
        If the BLS endpoint returns a non-2xx status.

    Examples
    --------
    >>> df = fetch_qcew(2022, "42101")  # doctest: +SKIP
    >>> df[["industry_code", "annual_avg_emplvl"]].head()
    """
    raw_csv = _fetch_qcew_raw(year, quarter, area_fips)
    from io import StringIO

    df = pd.read_csv(StringIO(raw_csv), dtype=str)

    # Standardise column names (BLS uses both space- and underscore-separated
    # names depending on vintage)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Filter ownership
    if "own_code" in df.columns:
        df = df[df["own_code"] == ownership]

    # Filter NAICS
    if naics_filter is not None and "industry_code" in df.columns:
        df = df[df["industry_code"].str.startswith(naics_filter)]

    # Coerce numeric columns
    numeric_cols = [
        "annual_avg_estabs",
        "annual_avg_emplvl",
        "total_annual_wages",
        "avg_annual_pay",
        "annual_contributions",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")

    logger.info(
        "Loaded %d QCEW rows for area=%s year=%d q=%s",
        len(df),
        area_fips,
        year,
        quarter,
    )
    return df.reset_index(drop=True)
