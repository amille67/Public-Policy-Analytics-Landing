"""Chapter 1: Indicators for Transit Oriented Development.

Builds tract-level transit proximity indicators for Philadelphia and
analyzes whether renters pay a premium for transit access.

Run:
    python -m chapters.ch01_transit_indicators --config config/chapters/ch01.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch1 pipeline end-to-end.

    Args:
        cfg: ChapterConfig with Ch1-specific fields.
        settings: PPASettings with global settings.
        output_root: Override output directory root.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    from ppa.geo.crs import ensure_crs
    from ppa.geo.nearest import mean_knn_distance
    from ppa.io.paths import chapter_output_dir
    from ppa.io.readers import read_geodataframe
    from ppa.io.writers import write_figure, write_geoparquet, write_json
    from ppa.stats.quantiles import q5, qbr
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.maps import choropleth_map, scatter_plot
    from ppa.viz.themes import map_theme, plot_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch01" if output_root else chapter_output_dir("ch01")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epsg = getattr(cfg, "crs_epsg", 26918) or 26918

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("Loading input layers...")
    inputs = cfg.inputs
    broad = read_geodataframe(data_root / inputs["broad_stations"])
    el = read_geodataframe(data_root / inputs["el_stations"])
    tracts = read_geodataframe(data_root / inputs["tracts"])

    sample_n = getattr(cfg, "sample", None)
    if sample_n:
        tracts = tracts.head(sample_n)
        logger.info("Sample mode: using %d tracts", len(tracts))

    # ── 1b. Handle long-format tract data ────────────────────────────────────
    # If tracts data is in long format (variable/value columns), pivot to wide
    long_format = getattr(cfg, "long_format", False)
    if long_format or ("variable" in tracts.columns and "value" in tracts.columns):
        logger.info("Pivoting long-format tract data to wide format")
        long_var_col = getattr(cfg, "long_variable_col", "variable")
        long_val_col = getattr(cfg, "long_value_col", "value")
        # Identify ID column before pivot
        id_col_candidate = getattr(cfg, "tract_id_col", "GEOID")
        if id_col_candidate not in tracts.columns:
            for c in ["GEOID", "GEOID10", "NAME"]:
                if c in tracts.columns:
                    id_col_candidate = c
                    break
        # Pivot and re-join geometry
        geom_col = tracts.geometry.name
        tracts_pivot = (
            tracts[[id_col_candidate, long_var_col, long_val_col]]
            .pivot_table(
                index=id_col_candidate,
                columns=long_var_col,
                values=long_val_col,
                aggfunc="first",
            )
            .reset_index()
        )
        # Get unique geometry per tract
        geom_df = tracts[[id_col_candidate, geom_col]].drop_duplicates(
            subset=id_col_candidate
        )
        tracts = gpd.GeoDataFrame(
            tracts_pivot.merge(geom_df, on=id_col_candidate, how="left"),
            geometry=geom_col,
            crs=tracts.crs,
        )
        logger.info(
            "Pivoted tracts: %d rows, %d columns", len(tracts), len(tracts.columns)
        )

    # ── 2. Reproject ─────────────────────────────────────────────────────────
    broad = ensure_crs(broad, epsg)
    el = ensure_crs(el, epsg)
    tracts = ensure_crs(tracts, epsg)

    # ── 3. Combine station layers ─────────────────────────────────────────────
    stations = pd.concat([broad, el], ignore_index=True)
    stations_gdf = gpd.GeoDataFrame(stations, crs=f"EPSG:{epsg}")

    # ── 4. Feature engineering ────────────────────────────────────────────────
    tract_id_col = getattr(cfg, "tract_id_col", None)
    # Try to find a tract ID column
    if not (tract_id_col and tract_id_col in tracts.columns):
        # Auto-detect
        for candidate in ["GEOID10", "GEOID", "tractid", "tract_id", "TRACTCE"]:
            if candidate in tracts.columns:
                tract_id_col = candidate
                break
        else:
            tract_id_col = "tract_id_auto"

    # Compute tract centroids
    tracts_proj = tracts.copy()
    tracts_proj["centroid"] = tracts_proj.geometry.centroid
    centroids_xy = np.column_stack(
        [tracts_proj["centroid"].x, tracts_proj["centroid"].y]
    )
    stations_xy = np.column_stack([stations_gdf.geometry.x, stations_gdf.geometry.y])

    # Nearest station distance
    k = getattr(cfg, "knn_k", 1)
    dist_to_transit = mean_knn_distance(centroids_xy, stations_xy, k=k)
    tracts_proj["dist_to_transit_m"] = dist_to_transit

    # Find rent column
    rent_col = getattr(cfg, "median_rent_col", None)
    if rent_col and rent_col in tracts_proj.columns:
        tracts_proj["median_rent"] = pd.to_numeric(
            tracts_proj[rent_col], errors="coerce"
        )
    else:
        # Try auto-detect including Census variable codes
        for candidate in [
            "medRent",
            "median_rent",
            "MedRent",
            "med_rent",
            "B25058e1",
            "H056001",
            "B25058_001E",
            "median_gross_rent",
        ]:
            if candidate in tracts_proj.columns:
                tracts_proj["median_rent"] = pd.to_numeric(
                    tracts_proj[candidate], errors="coerce"
                )
                rent_col = candidate
                break
        else:
            logger.warning("Could not find rent column; using placeholder zeros")
            tracts_proj["median_rent"] = 0.0

    # Quintile bins
    tracts_proj["rent_q5"] = q5(tracts_proj["median_rent"]).astype("Int64")
    rent_breaks = qbr(tracts_proj, "median_rent", rnd=None)
    logger.info("Rent quantile breaks: %s", rent_breaks)

    # ── 5. Modeling ───────────────────────────────────────────────────────────
    model_df = tracts_proj[["median_rent", "dist_to_transit_m"]].dropna()
    metrics: dict[str, Any] = {"model": "ols", "n": len(model_df)}

    if len(model_df) >= 5:
        X = sm.add_constant(model_df[["dist_to_transit_m"]].values)
        y = model_df["median_rent"].values
        ols = sm.OLS(y, X).fit(
            cov_type=(
                getattr(cfg, "model", {}).get("robust_se", "HC1")
                if isinstance(getattr(cfg, "model", None), dict)
                else "HC1"
            )
        )
        y_pred = ols.predict(X)
        residuals = y - y_pred
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))

        metrics.update(
            {
                "r2": round(r2, 6),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "coef": {
                    k: float(v)
                    for k, v in zip(["const", "dist_to_transit_m"], ols.params.tolist())
                },
                "pvalues": {
                    k: float(v)
                    for k, v in zip(
                        ["const", "dist_to_transit_m"], ols.pvalues.tolist()
                    )
                },
            }
        )
        logger.info("OLS R2=%.4f MAE=%.2f RMSE=%.2f", r2, mae, rmse)
    else:
        logger.warning("Not enough data for modeling (n=%d)", len(model_df))
        metrics.update(
            {
                "r2": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "coef": {},
                "pvalues": {},
            }
        )

    # ── 6. Build output GeoDataFrame ──────────────────────────────────────────
    out_gdf = tracts_proj[
        [tract_id_col, "median_rent", "dist_to_transit_m", "rent_q5", "geometry"]
    ].copy()
    out_gdf = out_gdf.rename(columns={tract_id_col: "tract_id"})

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    write_geoparquet(out_gdf, out_dir / "features.geoparquet")
    write_json(metrics, out_dir / "model_metrics.json")

    # ── 8. Figures ────────────────────────────────────────────────────────────
    mtheme = map_theme(title_size=16)
    ptheme = plot_theme(title_size=16)

    try:
        fig_map = choropleth_map(
            out_gdf,
            "rent_q5",
            title="Median Rent Quintiles",
            cmap="YlOrRd",
            overlay_gdfs=[stations_gdf],
            overlay_colors=["blue"],
            theme=mtheme,
        )
        write_figure(fig_map, fig_dir / "rent_quintiles.png")
    except Exception as e:
        logger.warning("Could not create rent_quintiles map: %s", e)

    try:
        valid = tracts_proj.dropna(subset=["dist_to_transit_m", "median_rent"])
        fig_scatter = scatter_plot(
            valid["dist_to_transit_m"],
            valid["median_rent"],
            xlabel="Distance to Transit (m)",
            ylabel="Median Rent ($)",
            title="Median Rent vs Distance to Transit",
            theme=ptheme,
        )
        write_figure(fig_scatter, fig_dir / "rent_vs_dist.png")
    except Exception as e:
        logger.warning("Could not create rent_vs_dist scatter: %s", e)

    logger.info("Ch01 pipeline complete. Outputs in %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Ch01 pipeline."""
    parser = argparse.ArgumentParser(
        description="Ch01: Transit Indicators for Philadelphia"
    )
    parser.add_argument("--config", required=True, help="Path to ch01.yaml config")
    parser.add_argument("--sample", type=int, default=None, help="Sample N tracts")
    parser.add_argument("--output-root", default=None, help="Override output root path")
    args = parser.parse_args(argv)

    from ppa.util.config import load_chapter_config, load_settings
    from ppa.util.logging import get_logger

    settings = load_settings()
    cfg = load_chapter_config(Path(args.config), settings)

    if args.sample:
        cfg.sample = args.sample
    if args.output_root:
        import os

        os.environ["PPA_OUTPUT_ROOT"] = args.output_root

    log = get_logger(__name__, settings.log_level)
    log.info("Starting Ch01 pipeline")

    output_root = Path(args.output_root) if args.output_root else None

    try:
        build_pipeline(cfg, settings, output_root=output_root)
        return 0
    except Exception:
        log.exception("Ch01 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
