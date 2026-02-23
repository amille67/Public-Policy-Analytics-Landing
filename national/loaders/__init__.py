"""Cached loaders for federal open-data APIs used in national extensions."""

from national.loaders.bls import fetch_qcew
from national.loaders.cdc_wonder import fetch_wonder
from national.loaders.housing import load_fhfa_hpi, load_hud_subsidized_households
from national.loaders.landuse import (
    compute_nlcd_impervious_by_tract,
    load_building_footprints,
    load_padus,
)
from national.loaders.mobility import build_spacetime_panel
from national.loaders.osm import fetch_osm_risk_proxies
from national.loaders.tiger import fetch_tiger_tracts

__all__: list[str] = [
    "build_spacetime_panel",
    "compute_nlcd_impervious_by_tract",
    "fetch_osm_risk_proxies",
    "fetch_qcew",
    "fetch_tiger_tracts",
    "fetch_wonder",
    "load_building_footprints",
    "load_fhfa_hpi",
    "load_hud_subsidized_households",
    "load_padus",
]
