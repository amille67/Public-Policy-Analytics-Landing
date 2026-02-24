"""Chapter 3: Intro to Geospatial ML — Part 1 (Boston Home Prices Baseline).

Predicts home prices in Boston using nearest-neighbor crime features and
evaluates accuracy and generalizability.

Run:
    python -m chapters.ch03_boston_prices_baseline --config config/chapters/ch03.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch3 pipeline end-to-end."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from ppa.geo.crs import ensure_crs
    from ppa.geo.nearest import mean_knn_distance
    from ppa.geo.overlay import sjoin
    from ppa.io.paths import chapter_output_dir
    from ppa.io.readers import read_csv, read_geodataframe
    from ppa.io.writers import write_figure, write_json, write_parquet
    from ppa.ml.metrics import metrics_by_group, regression_metrics
    from ppa.ml.models import fit_random_forest, save_model
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.plots import pred_vs_actual
    from ppa.viz.themes import plot_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch03" if output_root else chapter_output_dir("ch03")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epsg = getattr(cfg, "crs_epsg", 26919) or 26919
    inputs = cfg.inputs

    # ── 1. Load ───────────────────────────────────────────────────────────────
    houses_df = read_csv(data_root / inputs["houses"], encoding="latin-1")
    crimes_df = read_csv(data_root / inputs["crimes"], encoding="latin-1")

    sample_n = getattr(cfg, "sample", None)
    if sample_n:
        houses_df = houses_df.head(sample_n)

    # ── 2. Build GeoDataFrames ────────────────────────────────────────────────
    lon_col = getattr(cfg, "house_lon_col", "LON")
    lat_col = getattr(cfg, "house_lat_col", "LAT")
    target_col = getattr(cfg, "target_price_col", "SalePrice")

    # Find coordinate columns
    for lc in [lon_col, "LON", "lon", "longitude", "Longitude"]:
        if lc in houses_df.columns:
            lon_col = lc
            break
    for lac in [lat_col, "LAT", "lat", "latitude", "Latitude"]:
        if lac in houses_df.columns:
            lat_col = lac
            break
    for tc in [target_col, "SalePrice", "sale_price", "price", "Price"]:
        if tc in houses_df.columns:
            target_col = tc
            break

    # Drop rows missing coordinates or target
    houses_df = houses_df.dropna(subset=[lon_col, lat_col, target_col])
    houses_df[target_col] = pd.to_numeric(houses_df[target_col], errors="coerce")
    houses_df = houses_df[houses_df[target_col] > 0]

    houses_gdf = gpd.GeoDataFrame(
        houses_df,
        geometry=gpd.points_from_xy(houses_df[lon_col], houses_df[lat_col]),
        crs="EPSG:4326",
    )
    houses_gdf = ensure_crs(houses_gdf, epsg)

    # Crime features
    crime_lon = getattr(cfg, "crime_lon_col", "Long")
    crime_lat = getattr(cfg, "crime_lat_col", "Lat")
    for lc in [crime_lon, "Long", "lon", "longitude", "X", "x"]:
        if lc in crimes_df.columns:
            crime_lon = lc
            break
    for lac in [crime_lat, "Lat", "lat", "latitude", "Y", "y"]:
        if lac in crimes_df.columns:
            crime_lat = lac
            break

    crimes_df = crimes_df.dropna(subset=[crime_lon, crime_lat])
    crimes_gdf = gpd.GeoDataFrame(
        crimes_df,
        geometry=gpd.points_from_xy(crimes_df[crime_lon], crimes_df[crime_lat]),
        crs="EPSG:4326",
    )
    crimes_gdf = ensure_crs(crimes_gdf, epsg)

    # Spatial join to neighborhoods (optional — fall back to spatial grid)
    nhoods_path = data_root / inputs.get("nhoods", "")
    if nhoods_path.exists():
        nhoods_gdf = read_geodataframe(nhoods_path)
        nhoods_gdf = ensure_crs(nhoods_gdf, epsg)
        nhood_id_col = None
        for c in ["Name", "NAME", "nhood_id", "Neighborhood", "neighborhood"]:
            if c in nhoods_gdf.columns:
                nhood_id_col = c
                break
        houses_with_nhood = sjoin(
            houses_gdf,
            nhoods_gdf[[nhood_id_col, "geometry"]] if nhood_id_col else nhoods_gdf,
            how="left",
            predicate="within",
        )
        if nhood_id_col and nhood_id_col + "_right" in houses_with_nhood.columns:
            houses_with_nhood["nhood_id"] = houses_with_nhood[nhood_id_col + "_right"]
        elif nhood_id_col and nhood_id_col in houses_with_nhood.columns:
            houses_with_nhood["nhood_id"] = houses_with_nhood[nhood_id_col]
        else:
            houses_with_nhood["nhood_id"] = "unknown"
    else:
        logger.warning(
            "Nhoods file not found at %s; using spatial grid cells", nhoods_path
        )
        houses_with_nhood = houses_gdf.copy()
        # 5x5 grid of cells as surrogate neighborhoods
        x_bins = np.searchsorted(
            np.linspace(houses_gdf.geometry.x.min(), houses_gdf.geometry.x.max(), 6),
            np.asarray(houses_gdf.geometry.x),
        ).clip(0, 4)
        y_bins = np.searchsorted(
            np.linspace(houses_gdf.geometry.y.min(), houses_gdf.geometry.y.max(), 6),
            np.asarray(houses_gdf.geometry.y),
        ).clip(0, 4)
        houses_with_nhood["nhood_id"] = (x_bins * 5 + y_bins).astype(str)

    # kNN crime distance
    k = getattr(cfg, "knn_k", 5)
    houses_xy = np.column_stack(
        [houses_with_nhood.geometry.x, houses_with_nhood.geometry.y]
    )
    crimes_xy = np.column_stack([crimes_gdf.geometry.x, crimes_gdf.geometry.y])

    if len(crimes_xy) >= k:
        crime_dist = mean_knn_distance(houses_xy, crimes_xy, k=min(k, len(crimes_xy)))
    else:
        crime_dist = np.zeros(len(houses_xy))

    houses_with_nhood = houses_with_nhood.copy()
    houses_with_nhood["crime_knn_mean_dist_m"] = crime_dist

    # ── 3. Model ──────────────────────────────────────────────────────────────
    # Feature columns: numeric columns minus target and geo columns
    exclude = {
        target_col,
        lon_col,
        lat_col,
        "geometry",
        "nhood_id",
        crime_lon,
        crime_lat,
    }
    numeric_cols = [
        c
        for c in houses_with_nhood.select_dtypes(include="number").columns
        if c not in exclude and "Unnamed" not in c
    ]
    feature_cols = list(set([*numeric_cols, "crime_knn_mean_dist_m"]))

    feat_df = houses_with_nhood[[*feature_cols, target_col, "nhood_id"]].dropna()

    if len(feat_df) < 10:
        logger.warning("Too few rows after dropna (%d); skipping model", len(feat_df))
        feat_df_save = houses_with_nhood[
            [target_col, "crime_knn_mean_dist_m", "nhood_id"]
        ].copy()
        write_parquet(feat_df_save, out_dir / "features.parquet")
        write_json(
            {"rmse": None, "mae": None, "r2": None, "by_neighborhood": {}},
            out_dir / "model_metrics.json",
        )
        return

    X = np.asarray(feat_df[feature_cols], dtype=float)
    y = np.asarray(feat_df[target_col], dtype=float)
    groups = np.asarray(feat_df["nhood_id"].astype(str))

    train_frac = getattr(cfg, "train_frac", 0.8)
    X_train, X_test, y_train, y_test, _, g_test = train_test_split(
        X, y, groups, test_size=1 - train_frac, random_state=settings.seed
    )

    model_cfg = getattr(cfg, "model", {}) or {}
    n_est = model_cfg.get("n_estimators", 100) if isinstance(model_cfg, dict) else 100

    model = fit_random_forest(X_train, y_train, n_estimators=n_est, seed=settings.seed)
    y_pred = model.predict(X_test)

    global_metrics = regression_metrics(y_test, y_pred)
    test_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "nhood_id": g_test})
    by_nhood = metrics_by_group(test_df, "y_true", "y_pred", "nhood_id")

    metrics = {**global_metrics, "by_neighborhood": by_nhood}

    # ── 4. Save ───────────────────────────────────────────────────────────────
    save_model(model, out_dir / "model.pkl")
    feat_out = houses_with_nhood[
        [target_col, "crime_knn_mean_dist_m", "nhood_id"]
    ].copy()
    write_parquet(feat_out, out_dir / "features.parquet")
    write_json(metrics, out_dir / "model_metrics.json")

    # Figure
    try:
        ptheme = plot_theme(title_size=14)
        fig = pred_vs_actual(
            y_test,
            y_pred,
            title="Boston Home Prices: Predicted vs Actual",
            theme=ptheme,
        )
        write_figure(fig, fig_dir / "pred_vs_actual.png")
    except Exception as e:
        logger.warning("Figure error: %s", e)

    logger.info(
        "Ch03 complete. RMSE=%.2f R2=%.4f", global_metrics["rmse"], global_metrics["r2"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch03: Boston Prices Baseline")
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
        logger.exception("Ch03 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
