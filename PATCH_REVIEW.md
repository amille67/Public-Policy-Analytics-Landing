# Patch Review: Nashville MVP — 7 Pre-Computed High-Value Views

Reviewer: Claude Code
Branch: `claude/review-patch-integration-mUG5u`
Date: 2026-02-23

---

## Overview

This patch adds a Nashville MVP ETL pipeline, a Streamlit dashboard, and seven
pre-computed geospatial views materialised from open-data sources (hubNashville/
Socrata, TIGER/ACS, FHFA, OSM).  The changes span five layers:

1. `etl/` — orchestration entry-point and runner dispatch
2. `national/loaders/` — new loaders for Nashville, updated housing/OSM/TIGER
3. `src/ppa/geo/overlay.py` — new `apportion_by_area` utility
4. `streamlit_app/` — full dashboard scaffold
5. `tests/unit/` — three new test modules

The architecture is sound and the module boundaries are clean.  There are,
however, several bugs ranging from critical (tests that will not pass) to
medium (silent data errors) that must be addressed before merging.

---

## Integration Plan

### Prerequisites (before any code runs)

```
pip install sodapy pydeck altair streamlit
```

`sodapy`, `pydeck`, `altair`, and `streamlit` are used by the patch but are
**not listed in `pyproject.toml`** (`osmnx` is already present).  They must be
added to `[project.dependencies]` or a new `[project.optional-dependencies]`
group (e.g. `nashville`).

### Recommended integration sequence

1. Add missing deps to `pyproject.toml`.
2. Fix the critical and high-priority bugs listed below.
3. Add `streamlit_app/__init__.py` (empty file).
4. Materialise views: `python -m etl.create_high_value_views --views all`
5. Run unit tests: `pytest tests/unit/ -v`
6. Launch dashboard: `streamlit run streamlit_app/main.py`

### Data flow

```
hubNashville Socrata API ──────────────────────────────────┐
TIGER/Census API ──────────────────────────────────────────┤
FHFA HPI CSV (live download) ──────────────────────────────┤
OSM / osmnx ───────────────────────────────────────────────┤
                                                           ▼
                        etl/create_high_value_views.py
                        (calls 7 runners in etl/runners/view_runners.py)
                                                           │
                        data/views/view_XX_{lowest,county}.geoparquet
                                                           │
                        streamlit_app/main.py  ◄───────────┘
```

---

## Bug Report

### Critical

---

#### BUG-01 — Tests fail: monkeypatching puts callables into `RUNNER_MAP` but `_resolve_runner` expects strings

**Files:** `tests/unit/test_high_value_views_orchestrator.py` (lines 18–19),
`tests/unit/test_views_match_expected_counts.py` (lines 13–14),
`etl/create_high_value_views.py:_resolve_runner`

Both test files patch `RUNNER_MAP` by replacing its string values with the
`_fake_runner` callable:

```python
monkeypatch.setattr(mod, "RUNNER_MAP", {k: _fake_runner for k in mod.RUNNER_MAP})
```

`_resolve_runner` then does:

```python
return getattr(view_runners, RUNNER_MAP[runner_id])
```

`RUNNER_MAP[runner_id]` is now the callable `_fake_runner`, and
`getattr(module, <callable>)` raises `TypeError: attribute name must be string,
not 'function'`.  Both tests will error, not pass.

**Fix:** Monkeypatch `_resolve_runner` itself, not the map:

```python
monkeypatch.setattr(mod, "_resolve_runner", lambda runner_id: _fake_runner)
```

---

#### BUG-02 — `fhfa_hpi()` returns the entire national DataFrame when called with the default Nashville FIPS

**File:** `national/loaders/housing.py:fhfa_hpi` (line ~52)

```python
return latest[latest["county_fips"].isin(set(fips_list))
              | (len(fips_list) == 1 and fips_list[0] == "47037")]
```

When `fips_list == ["47037"]` (the default), the right-hand side of `|`
evaluates to `True` (a Python scalar), so pandas broadcasts it as all-True
and the entire `latest` DataFrame is returned — potentially thousands of
county rows from a national file instead of one row for Davidson County.

**Fix:** Remove the special-case clause entirely:

```python
return latest[latest["county_fips"].isin(set(fips_list))]
```

---

#### BUG-03 — `_save_outputs` routes non-geo View 7 through `write_geoparquet`, breaking the Streamlit reader

**File:** `etl/create_high_value_views.py:_save_outputs` (lines ~79–89)

The guard `if hasattr(lowest, "to_parquet")` is always True for both
`pd.DataFrame` and `gpd.GeoDataFrame`, so even the non-spatial View 7 output
(a plain DataFrame from `run_view_07_national_housing_context`) is written via
`write_geoparquet`.  The file will lack geo-metadata, and
`gpd.read_parquet(file)` in `streamlit_app/data/loaders.py` will raise.

**Fix:** Check for an actual geometry column:

```python
if hasattr(lowest, "geometry") and "geometry" in lowest.columns:
    write_geoparquet(lowest, low_path)
else:
    write_parquet(pd.DataFrame(lowest), low_path)
```

Apply the same fix for the county output in the same function.

---

### High Priority

---

#### BUG-04 — Missing `streamlit_app/__init__.py`

**File:** `streamlit_app/main.py` (and all page files)

`main.py` uses package-style imports:

```python
from streamlit_app.components.cards import render_kpi_cards
```

Without an `__init__.py` in `streamlit_app/`, Python does not treat it as a
package and these imports fail with `ModuleNotFoundError`.

**Fix:** Add an empty (or one-line docstring) `streamlit_app/__init__.py`.

---

#### BUG-05 — `tiger_demographics` raises `AttributeError` when an ACS column is absent

**File:** `national/loaders/tiger.py:tiger_demographics` (lines ~96–103)

```python
total = pd.to_numeric(acs.get("B01003_001E"), errors="coerce").fillna(0)
```

`acs.get("B01003_001E")` returns `None` when the column is missing (e.g.
when `fetch_acs_tracts` is called with a non-demographics preset).
`pd.to_numeric(None, errors="coerce")` returns the scalar `float('nan')`, and
`float('nan').fillna(0)` raises `AttributeError: 'float' object has no
attribute 'fillna'`.

`fetch_acs_tracts` is called with `preset="demographics"` by default, so the
four ACS variables should normally be present.  But the function does not
assert or document this dependency, and any caller that previously called
`fetch_acs_tracts` with a different preset (or no preset) will silently cache
results missing the required columns.

**Fix:** Call `fetch_acs_tracts` with the explicit variables needed:

```python
acs = fetch_acs_tracts(
    year, state_fips,
    variables={
        "B01003_001E": "B01003_001E",
        "B03002_003E": "B03002_003E",
        "B17001_002E": "B17001_002E",
        "B08201_002E": "B08201_002E",
    },
)
```

Or guard against missing columns by wrapping in `pd.Series` with the correct
length before calling `fillna`.

---

#### BUG-06 — `test_config_must_define_exactly_seven_views` passes for the wrong reason

**File:** `tests/unit/test_high_value_views_orchestrator.py` (lines 41–48)

The test writes a YAML file and expects `run_selected` to raise `ValueError`:

```python
bad.write_text("views: [{view_id: 1, runner: run_view_01_nashville_311_risk}]")
```

`_load_config` calls `json.loads`, which raises `json.JSONDecodeError` on
YAML content (YAML keys are unquoted).  `json.JSONDecodeError` is a subclass
of `ValueError`, so the test's bare `except ValueError` catches it and
incorrectly reports success — the "exactly 7 views" guard has never been
exercised.

**Fix:** Write valid JSON in the test:

```python
bad.write_text('{"views": [{"view_id": 1, "runner_id": "01", "name": "x", "chapter": 1}]}')
```

---

#### BUG-07 — `fetch_osm_risk_proxies` regression: chunking removed for large metro areas

**File:** `national/loaders/osm.py:fetch_osm_risk_proxies`

The original implementation chunked queries by county polygon to stay within
Overpass API memory/timeout limits.  The patch replaces this with a single
`union_all()` query over the entire area.  For Davidson County — a large
metro — Overpass can time out or return incomplete results.

The `_chunk_polygons` helper has been deleted entirely.  Any existing callers
that relied on chunking (e.g. `test_national.py`) may now silently fail or
hang.

**Fix:** Either restore the chunking logic or document that the simplified
version is intentional and bounded to small query areas.

---

### Medium Priority

---

#### BUG-08 — Mutable default arguments across multiple loaders

**Files:** `national/loaders/housing.py`, `national/loaders/hubnashville.py`,
`national/loaders/tiger.py`, `national/loaders/osm.py`

All new public functions use `fips_list: list[str] = ["47037"]` as a default.
Python evaluates default values once at definition time; mutations to this
list in any call would affect all subsequent calls that use the default.

**Fix:** Use `None` as the sentinel:

```python
def tiger_tracts(fips_list: list[str] | None = None, *, year: int = 2022) -> ...:
    if fips_list is None:
        fips_list = ["47037"]
```

---

#### BUG-09 — `tiger_tracts` / `tiger_demographics` assume all FIPS codes are in the same state

**File:** `national/loaders/tiger.py` (lines ~63, ~82)

```python
state_fips = fips_list[0][:2]
county_fips = {f[2:] for f in fips_list}
```

If `fips_list` contains FIPS codes from more than one state, only the state
of the first entry is downloaded but all counties are used for filtering.
Tracts from the second state are silently dropped.

**Fix:** Group FIPS codes by state before downloading, then concatenate
results, or document the single-state constraint and raise if violated.

---

#### BUG-10 — Hardcoded "47037" in `_rollup_county` fallback path

**File:** `etl/create_high_value_views.py:_rollup_county` (line ~52)

```python
county_polys = tiger_tracts(["47037"])[["GEOID", "geometry"]].copy()
```

This branch (reached when `lowest` has no GEOID column) always downloads
Davidson County regardless of the actual data being processed.  For any
non-Nashville view this silently produces wrong county geometries.

**Fix:** Thread the `fips_list` through from the runner or pass it as a
parameter to `_rollup_county`.

---

#### BUG-11 — Hardcoded output path `Path("data/views")` in `run_selected`

**File:** `etl/create_high_value_views.py:run_selected` (line ~84)

`_save_outputs` accepts an `out_dir: Path` parameter, but the call site
always passes `Path("data/views")` — a relative path that resolves relative
to CWD at runtime.  Running the script from any directory other than the
project root will silently write files to the wrong location.

**Fix:** Resolve against the project root explicitly, or make `out_dir` a
CLI argument.

---

#### BUG-12 — `fhfa_hpi` column detection falls back to potentially wrong column names

**File:** `national/loaders/housing.py:fhfa_hpi` (lines ~37–41)

```python
county_col = "fips" if "fips" in df.columns else "FIPS"
date_col   = "yr"   if "yr"   in df.columns else "year"
hpi_col    = "index_nsa" if "index_nsa" in df.columns else df.columns[-1]
```

If neither "fips" nor "FIPS" is present, the code proceeds with
`county_col = "FIPS"` and immediately raises `KeyError` during the
`slim = df[[county_col, ...]]` selection.  The FHFA master CSV column layout
should be validated with an explicit check and a clear error message.

---

#### BUG-13 — `hubnashville_permits` hardcoded type filter may always be empty

**File:** `national/loaders/hubnashville.py:hubnashville_permits` (line ~87)

```python
gdf = gdf[permit_col.isin(["Demolition", "Rehab"])].copy()
```

The actual hubNashville permits dataset uses values such as `"DEMOLITION"`,
`"New Construction"`, or `"Rehab/Renovation"`.  The exact-string filter will
return zero rows for common casing variants, and the downstream runner will
produce a zero-count `permit_transition_count` for every tract with no error.

**Fix:** Use case-insensitive matching:

```python
gdf = gdf[permit_col.str.lower().isin(["demolition", "rehab"])].copy()
```

---

### Low Priority / Cosmetic

---

#### BUG-14 — `_fetch_tiger_bytes` cache key change invalidates existing disk cache

**File:** `national/loaders/tiger.py:_fetch_tiger_bytes`

The function gains a new `geography: str` parameter.  Joblib caches by
argument signature, so existing cached entries keyed on `(year, state_fips)`
will be ignored and re-downloaded.  This is not a correctness bug, but users
with warm caches will see unexpected network traffic after upgrading.

**Recommendation:** Document in the commit message.  No code change required.

---

#### BUG-15 — `selected_year` in `AppState` is defined but never used

**File:** `streamlit_app/core/state.py`, `streamlit_app/main.py`

`AppState.selected_year` is initialised to `2024` but no year-filter control
or filter logic references it.  The sidebar caption hard-codes `"ETL Vintage:
2024"` instead of reading from state.

**Recommendation:** Either wire up a year selector or remove the field.

---

#### BUG-16 — `views_config.json` chapter numbers are inconsistent with book chapters

Chapter assignments:

| View | Assigned chapter | Expected (book) |
|------|-----------------|-----------------|
| 5 — industrial footprints | 8 | 2 or 5 (land use / 311) |
| 7 — national housing     | 0 | N/A (national) |

Chapter `0` is non-standard.  Recommend using `"national"` or `null`.

---

## Summary Table

| ID | Severity | File | Description |
|----|----------|------|-------------|
| BUG-01 | **Critical** | `tests/unit/test_high_value_views_orchestrator.py`, `test_views_match_expected_counts.py` | Tests fail: callables injected into string-keyed map |
| BUG-02 | **Critical** | `national/loaders/housing.py` | `fhfa_hpi` returns full national dataset for default FIPS |
| BUG-03 | **Critical** | `etl/create_high_value_views.py` | View 7 written via `write_geoparquet` despite having no geometry |
| BUG-04 | **High** | `streamlit_app/` | Missing `__init__.py` — package imports fail |
| BUG-05 | **High** | `national/loaders/tiger.py` | `tiger_demographics` raises `AttributeError` on missing ACS columns |
| BUG-06 | **High** | `tests/unit/test_high_value_views_orchestrator.py` | Config test passes for wrong reason (YAML → JSON parse error) |
| BUG-07 | **High** | `national/loaders/osm.py` | `fetch_osm_risk_proxies` regression: chunking removed |
| BUG-08 | Medium | multiple loaders | Mutable default argument `fips_list = ["47037"]` |
| BUG-09 | Medium | `national/loaders/tiger.py` | Single-state assumption with no guard |
| BUG-10 | Medium | `etl/create_high_value_views.py` | Hardcoded "47037" in county rollup fallback |
| BUG-11 | Medium | `etl/create_high_value_views.py` | Hardcoded relative output path `data/views` |
| BUG-12 | Medium | `national/loaders/housing.py` | `fhfa_hpi` column detection raises `KeyError` silently |
| BUG-13 | Medium | `national/loaders/hubnashville.py` | Permit type filter case-sensitive; likely always empty |
| BUG-14 | Low | `national/loaders/tiger.py` | Cache invalidation from signature change |
| BUG-15 | Low | `streamlit_app/core/state.py` | `selected_year` unused |
| BUG-16 | Low | `etl/views_config.json` | Chapter numbers inconsistent |

**Missing from `pyproject.toml`:** `sodapy`, `streamlit`, `pydeck`, `altair`

---

## What Is Working Well

- The `apportion_by_area` implementation in `ppa/geo/overlay.py` is clean,
  well-documented, and its tests are sound (aside from the EPSG:5070 /
  equatorial-coordinates edge-case in the test fixture).
- The Socrata loader pattern in `hubnashville.py` — lazy `sodapy` import
  inside `_client()`, silent empty GeoDataFrame on failure, automatic spatial
  join to TIGER tracts — is defensive and easy to extend.
- The Streamlit component split (`cards.py`, `charts.py`, `maps.py`,
  `core/state.py`) is a solid foundation for the dashboard.
- `tiger_tracts` / `tiger_block_groups` wrappers over the existing
  `fetch_tiger_tracts` are clean and additive with no breaking changes to
  existing callers.
