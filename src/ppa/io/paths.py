"""Canonical path resolution for data and output artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def get_data_root() -> Path:
    """Return the data root path, respecting PPA_DATA_ROOT env var."""
    return Path(os.environ.get("PPA_DATA_ROOT", "data/raw/DATA"))


def get_output_root() -> Path:
    """Return the output root path, respecting PPA_OUTPUT_ROOT env var."""
    return Path(os.environ.get("PPA_OUTPUT_ROOT", "outputs"))


def chapter_output_dir(chapter_id: str, output_root: Path | None = None) -> Path:
    """Return the output directory for a chapter.

    Args:
        chapter_id: Chapter identifier (e.g., "ch01").
        output_root: Override for output root. Uses env var or default if None.

    Returns:
        Path to the chapter output directory.
    """
    root = output_root or get_output_root()
    return root / chapter_id


def chapter_figures_dir(chapter_id: str, output_root: Path | None = None) -> Path:
    """Return the figures directory for a chapter."""
    return chapter_output_dir(chapter_id, output_root) / "figures"


def raw_data_path(relative: str, data_root: Path | None = None) -> Path:
    """Resolve a raw data path relative to the data root.

    Args:
        relative: Relative path (e.g., "Chapter1/SEPTA_Broad.geojson").
        data_root: Override for data root.

    Returns:
        Absolute Path to the raw data file.
    """
    root = data_root or get_data_root()
    return root / relative
