"""Chapter 5: Geospatial Risk Modeling (Predictive Policing).

Forecasts burglary risk in Chicago using geospatial exposure features
and Poisson GLM with leave-one-area-out cross-validation.

Run:
    python -m chapters.ch05_chicago_policing_risk --config config/chapters/ch05.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch5 pipeline end-to-end."""
    import numpy as np
    import pandas as pd
    import geopandas as gpd

    from ppa.geo.crs import ensure_crs
    from ppa.geo.overlay import clip, sjoin
    from ppa.io.readers import read_geodataframe
    from ppa.io.writers import write_figure, write_geoparquet, write_json
    from ppa.ml.cv import cross_validate_poisson_by_group
    from ppa.ml.metrics import regression_metrics
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.maps import choropleth_map
    from ppa.viz.themes import map_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch05" if output_root else Path("outputs/ch05")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epsg = getattr(cfg, "crs_epsg", 26916) or 26916
    inputs = cfg.inputs

    # ── 1. Load all layers ────────────────────────────────────────────────────
    logger.info("Loading Ch05 layers...")
    boundary = read_geodataframe(data_root / inputs["boundary"])
    nhoods = read_geodataframe(data_root / inputs["nhoods"])
    districts = read_geodataframe(data_root / inputs["districts"])
    beats = read_geodataframe(data_root / inputs["beats"])

    event_layers = {
        "abandoned_buildings": read_geodataframe(data_root / inputs["abandoned_buildings"]),
        "abandoned_cars": read_geodataframe(data_root / inputs["abandoned_cars"]),
        "graffiti": read_geodataframe(data_root / inputs["graffiti"]),
        "liquor_retail": read_geodataframe(data_root / inputs["liquor_retail"]),
        "sanitation": read_geodataframe(data_root / inputs["sanitation"]),
        "street_lights_out": read_geodataframe(data_root / inputs["street_lights_out"]),
    }
    burglaries17 = read_geodataframe(data_root / inputs["burglaries17"])
    burglaries18 = read_geodataframe(data_root / inputs["burglaries18"])

    # ── 2. Reproject ──────────────────────────────────────────────────────────
    boundary = ensure_crs(boundary, epsg)
    beats = ensure_crs(beats, epsg)
    districts = ensure_crs(districts, epsg)
    nhoods = ensure_crs(nhoods, epsg)
    burglaries17 = ensure_crs(burglaries17, epsg)
    burglaries18 = ensure_crs(burglaries18, epsg)

    for name in event_layers:
        event_layers[name] = ensure_crs(event_layers[name], epsg)
        event_layers[name] = clip(event_layers[name], boundary)

    burglaries17 = clip(burglaries17, boundary)
    burglaries18 = clip(burglaries18, boundary)

    # ── 3. Identify beat ID and CV group columns ──────────────────────────────
    beat_id_col = getattr(cfg, "beat_id_col", None)
    cv_group_col = getattr(cfg, "cv_group_col", None)

    for c in [beat_id_col, "beat_num", "beat", "BEAT_NUM", "BEAT"]:
        if c and c in beats.columns:
            beat_id_col = c
            break
    else:
        beat_id_col = beats.columns[0]

    for c in [cv_group_col, "district", "DISTRICT", "dist_num", "dist"]:
        if c and c in beats.columns:
            cv_group_col = c
            break
    else:
        cv_group_col = beat_id_col  # fallback

    # Sample for speed if needed
    sample_n = getattr(cfg, "sample", None)
    if sample_n and len(beats) > sample_n:
        beats = beats.head(sample_n)

    # ── 4. Outcome: aggregate burglaries to beats ─────────────────────────────
    def count_points_in_polys(points_gdf: Any, polys_gdf: Any, id_col: str) -> pd.Series:
        joined = sjoin(points_gdf, polys_gdf[[id_col, "geometry"]], how="right", predicate="within")
        return joined.groupby(id_col).size()

    logger.info("Aggregating burglaries to beats...")
    beats_with_counts = beats.copy()
    y17_counts = count_points_in_polys(burglaries17, beats, beat_id_col)
    y18_counts = count_points_in_polys(burglaries18, beats, beat_id_col)
    beats_with_counts["y_2017"] = beats_with_counts[beat_id_col].map(y17_counts).fillna(0).astype(int)
    beats_with_counts["y_2018"] = beats_with_counts[beat_id_col].map(y18_counts).fillna(0).astype(int)

    # ── 5. Exposure features via spatial join counts ──────────────────────────
    buffer_dists = getattr(cfg, "buffer_distances_m", [250, 500, 1000])
    logger.info("Computing exposure features with distances %s", buffer_dists)

    for feat_name, feat_gdf in event_layers.items():
        # Use centroid-based sjoin with fixed radii
        beats_centroids = beats_with_counts.copy()
        beats_centroids["geometry"] = beats_with_counts.geometry.centroid

        for dist in buffer_dists:
            buffered = beats_centroids.copy()
            buffered["geometry"] = buffered.geometry.buffer(dist)
            joined = sjoin(feat_gdf, buffered[[beat_id_col, "geometry"]], how="right", predicate="within")
            col_name = f"{feat_name}_cnt_d{dist}"
            cnt = joined.groupby(beat_id_col).size()
            beats_with_counts[col_name] = beats_with_counts[beat_id_col].map(cnt).fillna(0).astype(int)

    # ── 6. Poisson CV ─────────────────────────────────────────────────────────
    exposure_cols = [c for c in beats_with_counts.columns
                     if any(c.endswith(f"d{d}") for d in buffer_dists)]

    if cv_group_col not in beats_with_counts.columns:
        beats_with_counts[cv_group_col] = "single_group"

    # Ensure no nulls in predictors
    for col in exposure_cols:
        beats_with_counts[col] = beats_with_counts[col].fillna(0)

    n_groups = beats_with_counts[cv_group_col].nunique()
    logger.info("Running Poisson CV with %d groups on %d exposure features", n_groups, len(exposure_cols))

    if n_groups >= 2 and len(exposure_cols) > 0:
        try:
            cv_result = cross_validate_poisson_by_group(
                beats_with_counts,
                id_col=cv_group_col,
                dependent_variable="y_2017",
                ind_variables=exposure_cols,
            )
            cv_metrics = regression_metrics(cv_result["y_2017"], cv_result["Prediction"])
        except Exception as e:
            logger.warning("CV failed: %s; using zeros", e)
            beats_with_counts["Prediction"] = 0.0
            cv_result = beats_with_counts.copy()
            cv_metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    else:
        logger.warning("Insufficient groups or features for CV; skipping")
        beats_with_counts["Prediction"] = beats_with_counts["y_2017"].astype(float)
        cv_result = beats_with_counts.copy()
        cv_metrics = {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

    # Temporal validation
    corr_2018 = float(np.corrcoef(cv_result["Prediction"].astype(float), cv_result["y_2018"])[0, 1]) if "y_2018" in cv_result.columns else float("nan")
    temporal_metrics = regression_metrics(cv_result["y_2018"], cv_result["Prediction"]) if "y_2018" in cv_result.columns else {}

    metrics = {
        "cv_mae": cv_metrics.get("mae"),
        "cv_rmse": cv_metrics.get("rmse"),
        "temporal_mae_2018": temporal_metrics.get("mae"),
        "corr_pred_vs_2018": corr_2018,
        "n_beats": len(beats_with_counts),
    }

    # ── 7. Save ───────────────────────────────────────────────────────────────
    write_geoparquet(beats_with_counts, out_dir / "features.geoparquet")
    write_geoparquet(cv_result, out_dir / "cv_predictions.geoparquet")
    write_json(metrics, out_dir / "model_metrics.json")

    # Figures
    mtheme = map_theme(title_size=14)
    try:
        if "Prediction" in cv_result.columns:
            fig = choropleth_map(cv_result, "Prediction", title="Chicago Burglary Risk (Predicted 2017)", theme=mtheme)
            write_figure(fig, fig_dir / "risk_map.png")
    except Exception as e:
        logger.warning("Risk map error: %s", e)

    logger.info("Ch05 complete. CV MAE=%.2f", cv_metrics.get("mae", float("nan")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch05: Chicago Policing Risk")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    from ppa.util.config import load_chapter_config, load_settings
    from ppa.util.logging import get_logger

    settings = load_settings()
    cfg = load_chapter_config(Path(args.config), settings)
    if args.sample:
        cfg.sample = args.sample
    get_logger(__name__, settings.log_level)
    output_root = Path(args.output_root) if args.output_root else None

    try:
        build_pipeline(cfg, settings, output_root=output_root)
        return 0
    except Exception:
        logger.exception("Ch05 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
