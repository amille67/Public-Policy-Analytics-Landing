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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_CACHE_DIR: str = os.environ.get("PPA_CACHE_DIR", str(Path.home() / ".cache" / "ppa"))
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

_QCEW_URL: str = "https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area_fips}.csv"
_BLS_API_URL: str = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


@_memory.cache  # type: ignore[misc]
def _fetch_qcew_raw(
    year: int,
    quarter: str,
    area_fips: str,
    registration_key: str | None,
) -> str:
    """Download raw QCEW CSV from BLS."""
    url = _QCEW_URL.format(year=year, qtr=quarter, area_fips=area_fips)
    params: dict[str, str] = {}
    if registration_key:
        params["registrationkey"] = registration_key

    logger.info("Fetching BLS QCEW: %s", url)
    resp = requests.get(url, params=params or None, timeout=120)
    resp.raise_for_status()
    return resp.text


@retry(
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_qcew_raw_with_retry(
    year: int,
    quarter: str,
    area_fips: str,
    registration_key: str | None,
) -> str:
    """Download QCEW CSV with exponential-backoff retries."""
    return _fetch_qcew_raw(
        year=year,
        quarter=quarter,
        area_fips=area_fips,
        registration_key=registration_key,
    )


def fetch_qcew(
    year: int,
    area_fips: str,
    *,
    quarter: str = "a",
    naics_filter: str | None = None,
    ownership: str = "5",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch BLS QCEW employment and wage data for a county or MSA."""
    resolved_key = api_key or os.environ.get("BLS_API_KEY")
    raw_csv = _fetch_qcew_raw_with_retry(year, quarter, area_fips, resolved_key)
    from io import StringIO

    df = pd.read_csv(StringIO(raw_csv), dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "own_code" in df.columns:
        df = df[df["own_code"] == ownership]

    if naics_filter is not None and "industry_code" in df.columns:
        df = df[df["industry_code"].str.startswith(naics_filter)]

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
