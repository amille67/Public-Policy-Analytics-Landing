"""CDC WONDER data loader for mortality and health disparity analysis.

CDC WONDER requires agreement to data-use terms per session.  This module
targets the **public-use compressed mortality** endpoint, which supports
programmatic access without a browser session for most demographic breakdowns.

For Chapter 7 (algorithmic fairness), the key use case is pulling age-adjusted
mortality rates by county and race/ethnicity to benchmark COMPAS findings
against structural health disparities.

References
----------
- CDC WONDER: https://wonder.cdc.gov/
- Compressed Mortality File (1999-2020): https://wonder.cdc.gov/cmf-icd10.html
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR: str = os.environ.get("PPA_CACHE_DIR", str(Path.home() / ".cache" / "ppa"))
_memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

# CDC WONDER JSON API endpoint (requires Accept: application/json header)
_WONDER_BASE_URL: str = "https://wonder.cdc.gov/controller/datarequest/{dataset_id}"


@_memory.cache  # type: ignore[misc]
def _fetch_wonder_raw(
    dataset_id: str,
    request_body: str,
) -> dict[str, Any]:
    """POST a CDC WONDER data request and return raw JSON.

    Parameters
    ----------
    dataset_id : str
        CDC WONDER dataset identifier (e.g. ``"D76"`` for Compressed Mortality).
    request_body : str
        URL-encoded parameter string as accepted by WONDER's XML API.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    requests.HTTPError
        On non-2xx status.
    ValueError
        If the response is not valid JSON.
    """
    url = _WONDER_BASE_URL.format(dataset_id=dataset_id)
    logger.info("Fetching CDC WONDER dataset=%s", dataset_id)
    resp = requests.post(
        url,
        data=request_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=180,
    )
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def fetch_wonder(
    dataset_id: str,
    params: dict[str, str],
    *,
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch tabular data from CDC WONDER.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (e.g. ``"D76"`` = Compressed Mortality 1999-2020).
    params : dict[str, str]
        Parameter key/value pairs for the WONDER request.  See the WONDER
        documentation for each dataset's accepted parameters.
    rename : dict[str, str] or None
        Optional column rename mapping applied to the resulting DataFrame.

    Returns
    -------
    pandas.DataFrame
        Tabulated data with one row per group defined by ``params``.

    Raises
    ------
    requests.HTTPError
        On API error.
    ValueError
        If the response contains no data rows.

    Examples
    --------
    >>> # Mortality by county, race/ethnicity for 2018-2020
    >>> df = fetch_wonder(  # doctest: +SKIP
    ...     "D76",
    ...     {"B_1": "D76.V9", "B_2": "D76.V6", "M_1": "D76.M1"},
    ... )
    """
    import urllib.parse

    body = urllib.parse.urlencode(params)
    raw = _fetch_wonder_raw(dataset_id, body)

    # WONDER returns data under "data" key with "headers" and "rows"
    if "data" not in raw:
        raise ValueError(
            f"CDC WONDER response for dataset={dataset_id} contains no 'data' key. "
            f"Keys present: {list(raw.keys())}"
        )

    rows: list[list[str]] = raw["data"]
    headers: list[str] = raw.get("headers", [f"col_{i}" for i in range(len(rows[0]))])

    df = pd.DataFrame(rows, columns=headers)

    # Coerce obvious numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    if rename:
        df = df.rename(columns=rename)

    logger.info("Loaded %d CDC WONDER rows for dataset=%s", len(df), dataset_id)
    return df
