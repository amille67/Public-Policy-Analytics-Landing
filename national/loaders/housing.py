"""Housing-market and subsidy loaders for national ETL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fhfa_hpi(path: str | Path) -> pd.DataFrame:
    """Load FHFA HPI master CSV and standardize FIPS columns."""
    df = pd.read_csv(path, dtype=str)
    if "fips" in df.columns:
        df["fips"] = df["fips"].str.zfill(5)
    if "state_fips" in df.columns and "county_fips" in df.columns:
        df["state_fips"] = df["state_fips"].str.zfill(2)
        df["county_fips"] = df["county_fips"].str.zfill(3)
        df["fips"] = df["state_fips"] + df["county_fips"]
    return df


def load_hud_subsidized_households(path: str | Path) -> pd.DataFrame:
    """Load HUD Picture of Subsidized Households and standardize FIPS keys."""
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
