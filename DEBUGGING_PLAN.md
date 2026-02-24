# Deep Dive Complete — Exhaustive Debugging & Completion Plan for codex/review-codebase-for-remaining-problems
I fully explored the repo on this exact branch (file tree, raw source of every key file, PATCH_REVIEW.md, pyproject.toml, CI workflows, chapter pipelines, new Nashville MVP ETL, streamlit_app, tests, etc.).

## Quick Diagnosis (Why You're Stuck)

- Core = Solid Python port of Ken Steif’s Public Policy Analytics (8 chapters fully ported to ppa package + geopandas/rasterio/scikit-learn).
- New addition on this branch = Nashville MVP (7 pre-computed high-value geospatial views via Socrata/TIGER/FHFA/OSM → GeoParquet + Streamlit dashboard).
- Remaining blockers (from PATCH_REVIEW.md + code inspection):
  - Critical test mocking issue around RUNNER_MAP / _resolve_runner (tests expect callables; orchestrator now supports both but some tests may still fail).
  - streamlit_app/__init__.py missing (package requirement).
  - Views not yet materialized (data/views/ empty → dashboard breaks).
  - Possible follow-on issues: mypy on new apportion_by_area, rasterio/GDAL env quirks, path resolution in _project_root(), county roll-up geometry handling, CI smoke tests.

- CI is excellent (lint + mypy + unit matrix + ch01 smoke) but will fail until the above are fixed.
- No other TODOs/FIXMEs; everything else is clean.

Goal: Green CI, materialized views, working dashboard, shippable v0.2.0 in <4 hours.

## PHASE 0 – Reset & Reproduce (10 min)
- [x] Attempt fetch/checkout/pull
- [x] Start DEBUGGING_LOG.md

## PHASE 1 – Environment & Data (20 min)
- [ ] Create env (conda/mamba or uv)
- [ ] Install deps
- [ ] copy .env
- [ ] run cleanup.sh

## PHASE 2 – Fix PATCH_REVIEW.md Blockers (30–45 min)
- [ ] Verify mocking tests
- [ ] Ensure streamlit_app/__init__.py exists
- [ ] Materialize 7 views
- [ ] Dashboard smoke

## PHASE 3 – Full Quality Gates (45 min)
- [ ] Config validation
- [ ] Lint/format
- [ ] mypy
- [ ] unit tests
- [ ] ch01 smoke

## PHASE 4 – Chapter Regression (optional)
- [ ] Sample runs ch01..ch08

## PHASE 5 – Nashville Dashboard Polish & Final Verification
- [ ] Launch Streamlit and verify all 7 views

## PHASE 6 – CI Simulation & Release
- [ ] simulate CI
- [ ] commit/push/tag

## Advanced Debugging Tips
- GDAL/rasterio issues: install via conda-forge or pip --no-build-isolation.
- Import errors: sanity import run_selected.
- Geometry errors in roll-up: enforce EPSG:4326 before _rollup_county.
