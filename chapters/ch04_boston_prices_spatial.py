"""Chapter 4: Intro to Geospatial ML — Part 2 (Spatial CV).

Extends Ch3 to incorporate spatial cross-validation and spatially-informed
features for Boston home price prediction.

Run:
    python -m chapters.ch04_boston_prices_spatial --config config/chapters/ch04.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch4 pipeline end-to-end."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from ppa.geo.crs import ensure_crs
    from ppa.geo.nearest import mean_knn_distance
    from ppa.geo.overlay import sjoin
    from ppa.io.paths import chapter_output_dir
    from ppa.io.readers import read_csv, read_geodataframe
    from ppa.io.writers import write_figure, write_json, write_parquet
    from ppa.ml.metrics import regression_metrics
    from ppa.ml.models import fit_random_forest, save_model
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.themes import plot_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch04" if output_root else chapter_output_dir("ch04")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epsg = getattr(cfg, "crs_epsg", 26919) or 26919
    inputs = cfg.inputs

    # ── 1. Load data (reuse Ch3 logic) ────────────────────────────────────────
    houses_df = read_csv(data_root / inputs["houses"], encoding="latin-1")
    crimes_df = read_csv(data_root / inputs["crimes"], encoding="latin-1")

    sample_n = getattr(cfg, "sample", None)
    if sample_n:
        houses_df = houses_df.head(sample_n)

    lon_col = getattr(cfg, "house_lon_col", "LON")
    lat_col = getattr(cfg, "house_lat_col", "LAT")
    target_col = getattr(cfg, "target_price_col", "SalePrice")

    for lc in [lon_col, "LON", "lon", "longitude"]:
        if lc in houses_df.columns:
            lon_col = lc
            break
    for lac in [lat_col, "LAT", "lat", "latitude"]:
        if lac in houses_df.columns:
            lat_col = lac
            break
    for tc in [target_col, "SalePrice", "sale_price"]:
        if tc in houses_df.columns:
            target_col = tc
            break

    houses_df = houses_df.dropna(subset=[lon_col, lat_col, target_col])
    houses_df[target_col] = pd.to_numeric(houses_df[target_col], errors="coerce")
    houses_df = houses_df[houses_df[target_col] > 0]

    houses_gdf = gpd.GeoDataFrame(
        houses_df,
        geometry=gpd.points_from_xy(houses_df[lon_col], houses_df[lat_col]),
        crs="EPSG:4326",
    )
    houses_gdf = ensure_crs(houses_gdf, epsg)

    # Spatial join to neighborhoods (optional — fall back to spatial grid)
    nhoods_path = data_root / inputs.get("nhoods", "")
    if nhoods_path.exists():
        nhoods_gdf = read_geodataframe(nhoods_path)
        nhoods_gdf = ensure_crs(nhoods_gdf, epsg)
        nhood_id_col = None
        for c in ["Name", "NAME", "Neighborhood", "neighborhood"]:
            if c in nhoods_gdf.columns:
                nhood_id_col = c
                break
        houses_j = sjoin(houses_gdf, nhoods_gdf, how="left", predicate="within")
        if nhood_id_col and nhood_id_col + "_right" in houses_j.columns:
            houses_j["nhood_id"] = houses_j[nhood_id_col + "_right"]
        elif nhood_id_col and nhood_id_col in houses_j.columns:
            houses_j["nhood_id"] = houses_j[nhood_id_col]
        else:
            houses_j["nhood_id"] = "unknown"
    else:
        logger.warning(
            "Nhoods file not found at %s; using spatial grid cells", nhoods_path
        )
        houses_j = houses_gdf.copy()
        x_bins = np.searchsorted(
            np.linspace(houses_gdf.geometry.x.min(), houses_gdf.geometry.x.max(), 6),
            np.asarray(houses_gdf.geometry.x),
        ).clip(0, 4)
        y_bins = np.searchsorted(
            np.linspace(houses_gdf.geometry.y.min(), houses_gdf.geometry.y.max(), 6),
            np.asarray(houses_gdf.geometry.y),
        ).clip(0, 4)
        houses_j["nhood_id"] = (x_bins * 5 + y_bins).astype(str)

    crime_lon = getattr(cfg, "crime_lon_col", "Long")
    crime_lat = getattr(cfg, "crime_lat_col", "Lat")
    for lc in [crime_lon, "Long", "lon", "longitude", "X"]:
        if lc in crimes_df.columns:
            crime_lon = lc
            break
    for lac in [crime_lat, "Lat", "lat", "latitude", "Y"]:
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

    k = getattr(cfg, "knn_k", 5)
    houses_xy = np.column_stack([houses_j.geometry.x, houses_j.geometry.y])
    crimes_xy = np.column_stack([crimes_gdf.geometry.x, crimes_gdf.geometry.y])

    if len(crimes_xy) >= k:
        crime_dist = mean_knn_distance(houses_xy, crimes_xy, k=min(k, len(crimes_xy)))
    else:
        crime_dist = np.zeros(len(houses_xy))

    houses_j["crime_knn_mean_dist_m"] = crime_dist

    # ── 2. Spatial CV: leave-one-neighborhood-out ─────────────────────────────
    exclude = {target_col, lon_col, lat_col, "geometry", "nhood_id"}
    numeric_cols = [
        c
        for c in houses_j.select_dtypes(include="number").columns
        if c not in exclude and "Unnamed" not in c
    ]
    feature_cols = list(set([*numeric_cols, "crime_knn_mean_dist_m"]))

    feat_df = houses_j[[*feature_cols, target_col, "nhood_id"]].dropna()

    if len(feat_df) < 10:
        logger.warning("Too few rows after dropna; skipping CV")
        write_json(
            {"cv_rmse": None, "cv_mae": None, "cv_r2": None},
            out_dir / "model_metrics.json",
        )
        write_parquet(feat_df, out_dir / "cv_predictions.parquet")
        return

    cv_preds = []
    neighborhoods = feat_df["nhood_id"].unique()

    for nhood in neighborhoods:
        train_mask = feat_df["nhood_id"] != nhood
        test_mask = feat_df["nhood_id"] == nhood

        train = feat_df[train_mask]
        test = feat_df[test_mask]

        if len(train) < 5 or len(test) == 0:
            continue

        # Spatial feature: neighborhood mean price from training fold only
        nhood_means = train.groupby("nhood_id")[target_col].mean()
        train = train.copy()
        test = test.copy()
        train["nhood_mean_price"] = train["nhood_id"].map(nhood_means)
        test["nhood_mean_price"] = (
            test["nhood_id"].map(nhood_means).fillna(train[target_col].mean())
        )

        fcols_spatial = [*feature_cols, "nhood_mean_price"]

        model = fit_random_forest(
            train[fcols_spatial].values,
            train[target_col].values,
            n_estimators=50,
            seed=settings.seed,
        )
        preds = model.predict(test[fcols_spatial].values)
        fold_df = pd.DataFrame(
            {
                "y_true": test[target_col].values,
                "y_pred": preds,
                "nhood_id": nhood,
                "fold_id": nhood,
            }
        )
        cv_preds.append(fold_df)

    if not cv_preds:
        logger.warning("No CV folds completed")
        write_json({}, out_dir / "model_metrics.json")
        return

    cv_df = pd.concat(cv_preds, ignore_index=True)
    cv_df["residual"] = cv_df["y_true"] - cv_df["y_pred"]

    global_metrics = regression_metrics(cv_df["y_true"], cv_df["y_pred"])
    metrics = {
        **global_metrics,
        "n_folds": len(neighborhoods),
    }

    # Save
    write_parquet(cv_df, out_dir / "cv_predictions.parquet")
    write_json(metrics, out_dir / "model_metrics.json")

    # Optional: save final model on all data
    final_model = fit_random_forest(
        feat_df[feature_cols].values,
        feat_df[target_col].values,
        n_estimators=100,
        seed=settings.seed,
    )
    save_model(final_model, out_dir / "model.pkl")

    # Figure
    try:
        import matplotlib.pyplot as plt

        ptheme = plot_theme(title_size=14)
        with plt.rc_context(ptheme):
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped = cv_df.groupby("nhood_id")["residual"].agg(["mean", "std"])
            grouped.plot(kind="bar", y="mean", ax=ax, legend=False)
            ax.set_xlabel("Neighborhood")
            ax.set_ylabel("Mean Residual")
            ax.set_title("Mean Prediction Error by Neighborhood")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
        write_figure(fig, fig_dir / "residuals_by_neighborhood.png")
    except Exception as e:
        logger.warning("Figure error: %s", e)

    logger.info(
        "Ch04 complete. CV RMSE=%.2f R2=%.4f",
        global_metrics["rmse"],
        global_metrics["r2"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch04: Boston Prices Spatial CV")
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
        logger.exception("Ch04 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
