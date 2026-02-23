"""Parquet lakehouse helpers for national out-of-core processing."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd


class LakehouseError(RuntimeError):
    """Raised when lakehouse storage rules are violated."""


def interim_root(root: str | Path = "data/interim") -> Path:
    """Return interim lakehouse root, creating it if needed."""
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_partition_path(state_fips: str, dataset: str, root: str | Path = "data/interim") -> Path:
    """Return canonical parquet path for a state partition."""
    return interim_root(root) / dataset / f"STATEFP={state_fips}" / "part.parquet"


def checkpoint_exists(state_fips: str, dataset: str, root: str | Path = "data/interim") -> bool:
    """Check whether a state partition has already been written."""
    return state_partition_path(state_fips, dataset, root).exists()


def write_state_partition(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    *,
    state_fips: str,
    dataset: str,
    root: str | Path = "data/interim",
) -> Path:
    """Write one state partition to parquet under ``STATEFP=<xx>``."""
    path = state_partition_path(state_fips, dataset, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frame, gpd.GeoDataFrame):
        frame.to_parquet(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def validate_no_geojson(root: str | Path = "data/interim") -> None:
    """Raise if the interim lakehouse contains GeoJSON artifacts."""
    r = Path(root)
    offenders = list(r.rglob("*.geojson")) if r.exists() else []
    if offenders:
        raise LakehouseError(
            "GeoJSON artifacts are not permitted in national pipeline: "
            + ", ".join(str(p) for p in offenders)
        )


def validate_has_parquet(root: str | Path = "data/interim") -> None:
    """Raise if no parquet files are present under interim root."""
    r = Path(root)
    has_any = any(r.rglob("*.parquet")) if r.exists() else False
    if not has_any:
        raise LakehouseError("No parquet files found in interim lakehouse")
