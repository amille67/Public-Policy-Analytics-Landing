"""Artifact writers: geoparquet, parquet, csv, json, png."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_geoparquet(gdf: Any, path: Path) -> None:
    """Write a GeoDataFrame to geoparquet.

    Args:
        gdf: GeoDataFrame to write.
        path: Output file path (will create parent dirs).
    """
    _ensure_parent(path)
    gdf.to_parquet(path, index=False)
    logger.info("Wrote geoparquet: %s (%d rows)", path, len(gdf))


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet.

    Args:
        df: DataFrame to write.
        path: Output file path (will create parent dirs).
    """
    _ensure_parent(path)
    df.to_parquet(path, index=False)
    logger.info("Wrote parquet: %s (%d rows)", path, len(df))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV.

    Args:
        df: DataFrame to write.
        path: Output file path (will create parent dirs).
    """
    _ensure_parent(path)
    df.to_csv(path, index=False)
    logger.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(obj: Any, path: Path) -> None:
    """Write a JSON-serializable object to a file.

    Args:
        obj: JSON-serializable dict/list.
        path: Output file path (will create parent dirs).
    """
    _ensure_parent(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    logger.info("Wrote JSON: %s", path)


def write_figure(fig: Any, path: Path, dpi: int = 150) -> None:
    """Write a matplotlib figure to a PNG file.

    Args:
        fig: Matplotlib Figure object.
        path: Output file path (will create parent dirs).
        dpi: Image resolution in dots per inch.
    """
    import matplotlib.pyplot as plt

    _ensure_parent(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure: %s", path)


def _json_default(obj: Any) -> Any:
    """JSON encoder for non-serializable types."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
