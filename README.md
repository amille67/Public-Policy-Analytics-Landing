# Public Policy Analytics — Python Reimplementation

> **This is the Python reimplementation and national-scale extension of Ken Steif's
> [Public Policy Analytics](https://urbanspatial.github.io/PublicPolicyAnalytics/)
> case studies.** The original book teaches R-based data science at the intersection
> of geospatial analysis and public policy. This repository ports every chapter to
> idiomatic Python (`geopandas`, `scikit-learn`, `statsmodels`) and extends each
> case study with national-scale comparisons using open federal data (Census/ACS,
> BLS, CDC WONDER).

Original book: <https://urbanspatial.github.io/PublicPolicyAnalytics/>
Original repo: <https://github.com/urbanSpatial/Public-Policy-Analytics-Landing>

---

## Installation

### Prerequisites

- Python 3.11+
- `pip` or [`uv`](https://github.com/astral-sh/uv) for dependency management

### Quick Start

```bash
# Clone the repository
git clone https://github.com/amille67/Public-Policy-Analytics-Landing.git
cd Public-Policy-Analytics-Landing

# Install in editable mode with all extras (recommended)
pip install -e ".[dev,raster]"

# Or with uv (faster)
uv pip install -e ".[dev,raster]"
```

### Data Setup

The repository ships with a `DATA/` directory containing the original book
datasets. Run the consolidation script to move data into the standard layout:

```bash
bash cleanup.sh
```

This copies `DATA/` contents into `data/raw/DATA/`, which is the location
expected by the chapter pipelines (and is git-ignored for large files).

---

## Running Chapters

Each chapter is a self-contained pipeline script driven by a YAML config:

```bash
# Chapter 1 — Transit Indicators (Philadelphia)
python -m chapters.ch01_transit_indicators --config config/chapters/ch01.yaml

# Sample mode for quick testing
python -m chapters.ch01_transit_indicators --config config/chapters/ch01.yaml --sample 200

# All chapters follow the same pattern:
python -m chapters.ch02_ugb_sprawl             --config config/chapters/ch02.yaml
python -m chapters.ch03_boston_prices_baseline   --config config/chapters/ch03.yaml
python -m chapters.ch04_boston_prices_spatial    --config config/chapters/ch04.yaml
python -m chapters.ch05_chicago_policing_risk   --config config/chapters/ch05.yaml
python -m chapters.ch06_churn_bounce            --config config/chapters/ch06.yaml
python -m chapters.ch07_compas_fairness         --config config/chapters/ch07.yaml
python -m chapters.ch08_rideshare_demand        --config config/chapters/ch08.yaml
```

### National Extensions (Coming Soon)

```bash
# National extension notebooks live under national/
jupyter lab national/ch01_national_transit.ipynb
```

---

## Testing & Quality

```bash
# Unit tests
pytest tests/unit/

# Unit tests with coverage
pytest --cov=ppa --cov-report=term-missing tests/unit/

# Integration tests (requires data in data/raw/DATA/)
pytest tests/integration/

# Lint and type check
ruff check .
black --check .
mypy src/ppa
```

---

## Progress Table

| Ch | Title | Original R | Python Port | National Extension | Status |
|----|-------|:----------:|:-----------:|:------------------:|--------|
| 1 | Transit-Oriented Development Indicators | `tidycensus` + `sf` | `geopandas` + Census API | Multi-MSA transit premium (ACS 5-yr) | Port complete; national planned |
| 2 | Urban Growth Boundary & Sprawl | `sf` overlays | `geopandas` ring buffers | BLS QCEW land-use comparison | Port complete; national planned |
| 3 | Boston Home Prices — Baseline | `lm()` + `caret` | `scikit-learn` linear regression | Zillow ZHVI national trends | Port complete; national planned |
| 4 | Boston Home Prices — Spatial | `spdep` lag features | `libpysal` / manual lag | National spatial autocorrelation | Port complete; national planned |
| 5 | Predictive Policing (Chicago) | `spatstat` risk kernel | Poisson + KDE grid | FBI UCR / NIBRS comparison | Port complete; national planned |
| 6 | People-Based ML — Churn/Bounce | `caret` classifiers | `scikit-learn` classifiers | BLS JOLTS churn benchmarks | Port complete; national planned |
| 7 | Algorithmic Fairness (COMPAS) | Threshold + fairness grid | `scikit-learn` + fairness grid | CDC WONDER disparity analysis | Port complete; national planned |
| 8 | Rideshare Demand Forecasting | Space/time panel | Space/time panel (`statsmodels`) | National TNC trip comparison | Port complete; national planned |

### Legend

- **Port complete** — Python pipeline produces equivalent outputs to the R original.
- **National planned** — National-scale extension designed but not yet implemented.
- **In progress** — Actively being developed.

---

## Architecture

```
src/ppa/                   Installable Python package
  __init__.py              Version + public API re-exports
  data.py                  Shared data loaders (ACS fetcher, CRS standardizer)
  io/                      Dataset readers, writers, path resolution
  util/                    Config, logging, reproducibility, custom errors
  geo/                     CRS enforcement, kNN distance, ring buffers, overlays
  raster/                  Raster-to-DataFrame conversion
  stats/                   Quantile binning (q5, qbr)
  ml/                      Poisson CV, threshold sweep, fairness grid, models, metrics
  viz/                     Matplotlib themes (plot_theme, map_theme), maps, plots

chapters/                  Ch01–Ch08 pipeline scripts (one per chapter)
national/                  National-scale extension notebooks (planned)
config/                    YAML configs: default.yaml + chapters/chXX.yaml
tests/unit/                Unit tests for src/ppa helpers
tests/integration/         Integration / smoke tests
data/raw/DATA/             Raw datasets (git-ignored)
outputs/                   Chapter artifacts (git-ignored)
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PPA_DATA_ROOT` | Root of the raw data directory | `data/raw/DATA` |
| `PPA_OUTPUT_ROOT` | Root of the output directory | `outputs` |
| `PPA_SEED` | Global random seed | `42` |
| `PPA_LOG_LEVEL` | Logging level | `INFO` |
| `CENSUS_API_KEY` | Census Bureau API key (optional, increases rate limits) | — |

---

## R-to-Python Helper Mapping

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

---

## Datasets

All of the book's data is free and open source. The `DATA/` directory contains
datasets organized by chapter. See the
[original repository](https://github.com/urbanSpatial/Public-Policy-Analytics-Landing)
for full dataset provenance and API links.

| Chapter | Key Datasets | Format |
|---------|-------------|--------|
| 1 | SEPTA stations, Philadelphia census tracts | GeoJSON |
| 2 | Lancaster County buildings, UGB, green space | GeoJSON |
| 3–4 | Boston house prices, crimes, neighborhoods | CSV, Shapefile, GeoJSON |
| 5 | Chicago 311 calls, burglaries, police boundaries | GeoJSON |
| 6 | Churn/bounce, housing subsidy | CSV |
| 7 | COMPAS recidivism scores | CSV |
| 8 | Chicago rideshare trips (Nov–Dec 2018) | CSV |

---

## License

MIT — see [LICENSE](LICENSE) for details.
