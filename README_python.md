# Public Policy Analytics — Python Rebuild

Python reimplementation of the [Public Policy Analytics](https://github.com/urbanSpatial/Public-Policy-Analytics-Landing) R case studies (Chapters 1–8).

## Setup

```bash
# Install with all extras (recommended)
pip install -e ".[dev,geo,raster]"

# Core + geospatial only (no raster support)
pip install -e ".[dev,geo]"
```

## Data

Copy the upstream repository's `DATA/` folder into `data/raw/DATA/`:

```
data/raw/DATA/
  Chapter1/   SEPTA_Broad.geojson, SEPTA_El.geojson, PHL_CT00.geojson
  Chapter2/   studyAreaTowns.geojson, Urban_Growth_Boundary.geojson, ...
  Chapter3_4/ bostonHousePriceData_clean.csv, bostonCrimes.csv, Boston_Nhoods/...
  Chapter5/   chicagoBoundary.geojson, policeBeats.geojson, burglaries17.geojson, ...
  Chapter6/   churnBounce.csv, housingSubsidy.csv
  Chapter7/   compas-scores-two-years.csv
  Chapter8/   chicago_rideshare_trips_nov_dec_18_clean_sample.csv
```

## Running Chapters

```bash
# Chapter 1 — Transit Indicators (Philadelphia)
python -m chapters.ch01_transit_indicators --config config/chapters/ch01.yaml

# With sample mode (for testing):
python -m chapters.ch01_transit_indicators --config config/chapters/ch01.yaml --sample 200

# Override output directory:
python -m chapters.ch01_transit_indicators --config config/chapters/ch01.yaml --output-root my_outputs

# All chapters follow the same pattern:
python -m chapters.ch02_ugb_sprawl             --config config/chapters/ch02.yaml
python -m chapters.ch03_boston_prices_baseline --config config/chapters/ch03.yaml
python -m chapters.ch04_boston_prices_spatial  --config config/chapters/ch04.yaml
python -m chapters.ch05_chicago_policing_risk  --config config/chapters/ch05.yaml
python -m chapters.ch06_churn_bounce           --config config/chapters/ch06.yaml
python -m chapters.ch07_compas_fairness        --config config/chapters/ch07.yaml
python -m chapters.ch08_rideshare_demand       --config config/chapters/ch08.yaml
```

## Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=ppa --cov-report=term-missing tests/unit/

# Run integration tests (requires data)
pytest tests/integration/
```

## Lint and Type Check

```bash
ruff check .
black --check .
mypy src/ppa
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PPA_DATA_ROOT` | Root of the raw data directory | `data/raw/DATA` |
| `PPA_OUTPUT_ROOT` | Root of the output directory | `outputs` |
| `PPA_SEED` | Global random seed | `42` |
| `PPA_LOG_LEVEL` | Logging level | `INFO` |

## Architecture

```
src/ppa/
  io/           readers, writers, paths
  util/         config, logging, reproducibility, errors
  geo/          CRS enforcement, kNN distance, ring buffers, overlays
  raster/       raster→DataFrame conversion
  stats/        quantile binning (q5, qbr)
  ml/           Poisson CV, threshold sweep, fairness grid, models, metrics
  viz/          matplotlib themes (plot_theme, map_theme), maps, plots

chapters/       Ch01–Ch08 pipeline scripts
config/         YAML configs (default.yaml + chapters/chXX.yaml)
tests/unit/     Unit tests for all src/ppa helpers
tests/integration/ Ch01 smoke test
```

## Chapter Outputs

Each chapter writes artifacts to `outputs/chXX/`:

| Chapter | Key Outputs |
|---------|-------------|
| Ch01 | `features.geoparquet`, `model_metrics.json`, `figures/rent_quintiles.png` |
| Ch02 | `rings.geoparquet`, `ring_metrics.parquet`, `town_metrics.csv`, figures |
| Ch03 | `features.parquet`, `model.pkl`, `model_metrics.json`, figures |
| Ch04 | `cv_predictions.parquet`, `model.pkl`, `model_metrics.json`, figures |
| Ch05 | `features.geoparquet`, `cv_predictions.geoparquet`, `model_metrics.json`, figures |
| Ch06 | `thresholds.csv`, `model.pkl`, `model_metrics.json`, figures |
| Ch07 | `fairness_grid.csv`, `thresholds_by_group.csv`, `model_metrics.json`, figures |
| Ch08 | `time_series.parquet`, `predictions.parquet`, `model.pkl`, `model_metrics.json`, figures |

## R → Python Helper Mapping

| R (`functions.r`) | Python (`src/ppa`) |
|---|---|
| `plotTheme` | `ppa.viz.themes.plot_theme` |
| `mapTheme` | `ppa.viz.themes.map_theme` |
| `q5` | `ppa.stats.quantiles.q5` |
| `qBr` | `ppa.stats.quantiles.qbr` |
| `rast` | `ppa.raster.convert.rast_to_df` |
| `nn_function` | `ppa.geo.nearest.mean_knn_distance` |
| `multipleRingBuffer` | `ppa.geo.buffers.multiple_ring_buffer` |
| `crossValidate` | `ppa.ml.cv.cross_validate_poisson_by_group` |
| `iterateThresholds` | `ppa.ml.thresholds.iterate_thresholds` |
| `iterateFairness` | `ppa.ml.fairness.iterate_fairness` |
