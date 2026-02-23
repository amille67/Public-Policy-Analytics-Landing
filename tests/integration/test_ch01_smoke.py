"""Integration smoke test for Ch01 transit indicators pipeline.

Runs Ch01 against the actual data (or small sample) and verifies that:
- Required output files are created
- Output schema matches specification
- CRS is correct
- Numeric constraints are satisfied
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_ROOT = Path("data/raw/DATA")
CH01_DATA = DATA_ROOT / "Chapter1"
REQUIRED_DATA = [
    CH01_DATA / "SEPTA_Broad.geojson",
    CH01_DATA / "SEPTA_El.geojson",
    CH01_DATA / "PHL_CT00.geojson",
]


def data_available() -> bool:
    return all(p.exists() for p in REQUIRED_DATA)


@pytest.mark.skipif(
    not data_available(), reason="Ch01 data not available in data/raw/DATA"
)
def test_ch01_smoke(tmp_path: Path) -> None:
    """Full smoke run: outputs exist and pass schema checks."""
    import geopandas as gpd

    from chapters.ch01_transit_indicators import build_pipeline
    from ppa.util.config import ChapterConfig, PPASettings, load_chapter_config

    settings = PPASettings(
        data_root=DATA_ROOT,
        output_root=tmp_path,
        seed=42,
        log_level="WARNING",
    )

    config_path = Path("config/chapters/ch01.yaml")
    if config_path.exists():
        cfg = load_chapter_config(config_path, settings)
    else:
        cfg = ChapterConfig(
            chapter_id="ch01",
            crs_epsg=26918,
            sample=50,
            inputs={
                "broad_stations": "Chapter1/SEPTA_Broad.geojson",
                "el_stations": "Chapter1/SEPTA_El.geojson",
                "tracts": "Chapter1/PHL_CT00.geojson",
            },
        )
    cfg.sample = 50  # Force small sample for CI

    build_pipeline(cfg, settings, output_root=tmp_path)

    # ── Artifact existence checks ─────────────────────────────────────────────
    out_dir = tmp_path / "ch01"
    assert (out_dir / "features.geoparquet").exists(), "features.geoparquet not found"
    assert (out_dir / "model_metrics.json").exists(), "model_metrics.json not found"

    figures = list((out_dir / "figures").glob("*.png"))
    assert len(figures) >= 1, f"Expected at least 1 figure PNG, found {len(figures)}"

    # ── Schema checks ─────────────────────────────────────────────────────────
    gdf = gpd.read_parquet(out_dir / "features.geoparquet")
    required_cols = {
        "tract_id",
        "median_rent",
        "dist_to_transit_m",
        "rent_q5",
        "geometry",
    }
    missing = required_cols - set(gdf.columns)
    assert not missing, f"Missing columns in features.geoparquet: {missing}"

    # ── CRS check ────────────────────────────────────────────────────────────
    assert gdf.crs is not None, "Output GeoDataFrame has no CRS"
    assert gdf.crs.to_epsg() == 26918, f"Expected EPSG:26918, got {gdf.crs.to_epsg()}"

    # ── Numeric constraint checks ─────────────────────────────────────────────
    assert (
        gdf["dist_to_transit_m"].dropna() >= 0
    ).all(), "dist_to_transit_m has negative values"
    valid_q5 = gdf["rent_q5"].dropna()
    if len(valid_q5) > 0:
        assert (
            valid_q5.astype(float).between(1, 5).all()
        ), "rent_q5 values outside [1,5]"

    # ── Metrics JSON check ────────────────────────────────────────────────────
    with open(out_dir / "model_metrics.json") as f:
        metrics = json.load(f)

    assert "model" in metrics, "metrics JSON missing 'model' key"
    assert "n" in metrics, "metrics JSON missing 'n' key"
    assert "r2" in metrics, "metrics JSON missing 'r2' key"


@pytest.mark.skipif(not data_available(), reason="Ch01 data not available")
def test_ch01_deterministic(tmp_path: Path) -> None:
    """Run twice with same seed; metrics should be identical."""
    import json

    from chapters.ch01_transit_indicators import build_pipeline
    from ppa.util.config import ChapterConfig, PPASettings

    def run_pipeline(out: Path) -> dict:
        settings = PPASettings(data_root=DATA_ROOT, output_root=out, seed=42)
        cfg = ChapterConfig(
            chapter_id="ch01",
            crs_epsg=26918,
            sample=30,
            inputs={
                "broad_stations": "Chapter1/SEPTA_Broad.geojson",
                "el_stations": "Chapter1/SEPTA_El.geojson",
                "tracts": "Chapter1/PHL_CT00.geojson",
            },
        )
        build_pipeline(cfg, settings, output_root=out)
        with open(out / "ch01" / "model_metrics.json") as f:
            return json.load(f)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    m1 = run_pipeline(out1)
    m2 = run_pipeline(out2)

    assert m1["n"] == m2["n"], "Row count differs across runs"

    # R² within tolerance
    if m1.get("r2") is not None and m2.get("r2") is not None:
        assert abs(m1["r2"] - m2["r2"]) < 1e-6, "R² differs across runs"
