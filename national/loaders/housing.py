"""Housing-market and subsidy loaders for national ETL and Nashville context view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FHFA_HPI_URL = "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv"


def load_fhfa_hpi(path: str | Path) -> pd.DataFrame:
    """Load FHFA HPI master CSV from local path and standardize FIPS columns."""
    df = pd.read_csv(path, dtype=str)
    if "fips" in df.columns:
        df["fips"] = df["fips"].str.zfill(5)
    if "state_fips" in df.columns and "county_fips" in df.columns:
        df["state_fips"] = df["state_fips"].str.zfill(2)
        df["county_fips"] = df["county_fips"].str.zfill(3)
        df["fips"] = df["state_fips"] + df["county_fips"]
    return df


def load_hud_subsidized_households(path: str | Path) -> pd.DataFrame:
    """Load HUD data from local path and standardize FIPS keys."""
    df = pd.read_csv(path, dtype=str)
    for col in ("state_fips", "county_fips", "tractce"):
        if col in df.columns:
            width = {"state_fips": 2, "county_fips": 3, "tractce": 6}[col]
            df[col] = df[col].str.zfill(width)

    if {"state_fips", "county_fips", "tractce"}.issubset(df.columns):
        df["GEOID"] = df["state_fips"] + df["county_fips"] + df["tractce"]
    elif {"state_fips", "county_fips"}.issubset(df.columns):
        df["fips"] = df["state_fips"] + df["county_fips"]
    return df


def fhfa_hpi(fips_list: list[str] | None = None) -> pd.DataFrame:
    """Download FHFA HPI master file and return latest YoY by county FIPS."""
    if fips_list is None:
        fips_list = ["47037"]

    df = pd.read_csv(FHFA_HPI_URL, low_memory=False)

    # Detect column names robustly; raise clearly if the schema has changed.
    county_col = next((c for c in ("fips", "FIPS") if c in df.columns), None)
    date_col = next((c for c in ("yr", "year") if c in df.columns), None)
    hpi_col = next((c for c in ("index_nsa", "index_sa") if c in df.columns), None)
    if county_col is None or date_col is None or hpi_col is None:
        raise ValueError(
            f"FHFA HPI CSV schema unexpected. Columns found: {list(df.columns[:10])}"
        )

    slim = df[[county_col, date_col, hpi_col]].copy()
    slim.columns = ["county_fips", "year", "hpi"]
    slim["county_fips"] = slim["county_fips"].astype(str).str.zfill(5)
    slim["hpi"] = pd.to_numeric(slim["hpi"], errors="coerce")
    slim = slim.dropna(subset=["hpi"]).sort_values(["county_fips", "year"])
    slim["hpi_yoy"] = slim.groupby("county_fips")["hpi"].pct_change(periods=1).fillna(0)

    latest = slim.groupby("county_fips", as_index=False).tail(1)
    # Return only the requested FIPS codes.
    return latest[latest["county_fips"].isin(set(fips_list))].reset_index(drop=True)


def hud_county(fips_list: list[str] | None = None) -> pd.DataFrame:
    """Return lightweight HUD-style county totals placeholder for MVP contexts."""
    if fips_list is None:
        fips_list = ["47037"]
    return pd.DataFrame(
        {
            "county_fips": fips_list,
            "hud_total_assisted_households": [0] * len(fips_list),
        }
    )
