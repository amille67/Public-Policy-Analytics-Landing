"""Chapter 7: People-Based ML — Algorithmic Fairness.

Evaluates disparate impact using COMPAS recidivism data and group-specific
threshold optimization via iterateFairness.

Run:
    python -m chapters.ch07_compas_fairness --config config/chapters/ch07.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch7 pipeline end-to-end."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from ppa.io.paths import chapter_output_dir
    from ppa.io.readers import read_csv
    from ppa.io.writers import write_csv, write_figure, write_json
    from ppa.ml.fairness import iterate_fairness
    from ppa.ml.metrics import classification_metrics
    from ppa.ml.models import fit_logistic_regression, save_model
    from ppa.ml.thresholds import iterate_thresholds
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.plots import fpr_fnr_tradeoff
    from ppa.viz.themes import plot_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch07" if output_root else chapter_output_dir("ch07")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs = cfg.inputs

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = read_csv(data_root / inputs["compas"])

    group_col = getattr(cfg, "group_col", "race")
    group_a = getattr(cfg, "group_a", "African-American")
    group_b = getattr(cfg, "group_b", "Caucasian")
    observed_col_raw = getattr(cfg, "observed_col", "two_year_recid")
    feature_cols_cfg = getattr(cfg, "feature_cols", None) or [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
    ]
    threshold_by = float(getattr(cfg, "threshold_by", 0.1))
    min_group_n = int(getattr(cfg, "min_group_n", 10))

    # Normalize labels
    if observed_col_raw in df.columns:
        df["Recidivated"] = np.where(
            df[observed_col_raw].astype(str) == "1", "Recidivate", "notRecidivate"
        )
    else:
        raise ValueError(f"Observed column '{observed_col_raw}' not found in dataset")

    # Filter to known groups
    df = df[df[group_col].isin([group_a, group_b])].copy()

    for g in [group_a, group_b]:
        cnt = (df[group_col] == g).sum()
        if cnt < min_group_n:
            raise ValueError(
                f"Group '{g}' has only {cnt} rows (min required: {min_group_n})"
            )

    # Features
    feature_cols = [c for c in feature_cols_cfg if c in df.columns]
    if not feature_cols:
        feature_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in {observed_col_raw, group_col}
        ][:5]

    df = df.dropna(subset=[*feature_cols, group_col, "Recidivated"])

    sample_n = getattr(cfg, "sample", None)
    if sample_n:
        df = df.head(sample_n)

    X = df[feature_cols].values
    y_binary = (df["Recidivated"] == "Recidivate").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use index-based split so group labels can be aligned to the same test rows
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=settings.seed, stratify=y_binary
    )
    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_train = y_binary[train_idx]
    y_test = y_binary[test_idx]
    groups_test = df.iloc[test_idx][group_col].to_numpy()

    model = fit_logistic_regression(X_train, y_train, seed=settings.seed)
    save_model(model, out_dir / "model.pkl")

    y_proba = model.predict_proba(X_test)[:, 1]
    clf_metrics = classification_metrics(y_test, y_proba)

    # ── 2. Threshold sweep by group ───────────────────────────────────────────
    eval_df = pd.DataFrame(
        {
            "target": y_test,
            "p_recid": y_proba,
            group_col: groups_test,
        }
    )

    thresholds_by_group = iterate_thresholds(
        eval_df, "target", "p_recid", group=group_col, step=0.01
    )

    # ── 3. Fairness grid ──────────────────────────────────────────────────────
    df_test_for_fairness = pd.DataFrame(
        {
            group_col: groups_test,
            "Recidivated": np.where(y_test == 1, "Recidivate", "notRecidivate"),
        }
    )

    # Wrap model to return probabilities matching the test set
    class FixedProbModel:
        def __init__(self, probs: np.ndarray) -> None:
            self._probs = probs

        def predict_proba(self, X: Any) -> np.ndarray:
            n = len(X) if hasattr(X, "__len__") else len(self._probs)
            return np.column_stack([1 - self._probs[:n], self._probs[:n]])

    wrapped_model = FixedProbModel(y_proba)

    try:
        fairness_grid = iterate_fairness(
            df_test_for_fairness,
            wrapped_model,
            threshold_by=threshold_by,
            observed_col="Recidivated",
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            feature_cols=None,  # using wrapped model with pre-computed probs
        )
    except Exception as e:
        logger.warning("Fairness grid failed: %s; creating empty grid", e)
        fairness_grid = pd.DataFrame()

    # ── 4. Select optimal threshold ───────────────────────────────────────────
    selected_thresholds: dict[str, Any] = {"threshold_a": 0.5, "threshold_b": 0.5}
    if not fairness_grid.empty:
        grid_wide = fairness_grid.pivot_table(
            index="threshold",
            columns=group_col,
            values=["False_Positive_Rate", "False_Negative_Rate", "Accuracy"],
        )
        min_acc = float(getattr(cfg, "min_accuracy", 0.55))
        if (
            group_a in grid_wide["Accuracy"].columns
            and group_b in grid_wide["Accuracy"].columns
        ):
            min_acc_mask = (grid_wide["Accuracy"][group_a].fillna(0) >= min_acc) & (
                grid_wide["Accuracy"][group_b].fillna(0) >= min_acc
            )
            if min_acc_mask.any():
                filtered = grid_wide[min_acc_mask]
                disparity = (
                    (
                        filtered["False_Positive_Rate"][group_a]
                        - filtered["False_Positive_Rate"][group_b]
                    ).abs()
                    + (
                        filtered["False_Negative_Rate"][group_a]
                        - filtered["False_Negative_Rate"][group_b]
                    ).abs()
                ).fillna(float("inf"))
                best_thresh_str = disparity.idxmin()
                selected_thresholds["optimal_threshold_pair"] = str(best_thresh_str)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    write_csv(thresholds_by_group, out_dir / "thresholds_by_group.csv")
    if not fairness_grid.empty:
        write_csv(fairness_grid, out_dir / "fairness_grid.csv")
    write_json(
        {
            "roc_auc": clf_metrics["roc_auc"],
            "selected_thresholds": selected_thresholds,
            "n_test": len(y_test),
        },
        out_dir / "model_metrics.json",
    )

    # Figure
    try:
        if not fairness_grid.empty:
            ptheme = plot_theme(title_size=16)
            fig = fpr_fnr_tradeoff(fairness_grid, group_col=group_col, theme=ptheme)
            write_figure(fig, fig_dir / "fairness_tradeoff.png")
    except Exception as e:
        logger.warning("Figure error: %s", e)

    logger.info("Ch07 complete. ROC-AUC=%.4f", clf_metrics["roc_auc"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch07: Algorithmic Fairness")
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
        logger.exception("Ch07 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
