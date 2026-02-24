"""Cached loaders for federal/local open-data APIs used in PPA extensions."""

from national.loaders.bls import fetch_qcew
from national.loaders.cdc_wonder import fetch_wonder
from national.loaders.housing import (
    fhfa_hpi,
    hud_county,
    load_fhfa_hpi,
    load_hud_subsidized_households,
)
from national.loaders.hubnashville import (
    hubnashville_311,
    hubnashville_assessor,
    hubnashville_parcels,
    hubnashville_permits,
    hubnashville_usd,
)
from national.loaders.landuse import (
    compute_nlcd_impervious_by_tract,
    load_building_footprints,
    load_padus,
)
from national.loaders.mobility import build_spacetime_panel
from national.loaders.osm import (
    fetch_osm_risk_proxies,
    osm_industrial_footprints,
    osm_transit_stops,
)
from national.loaders.tiger import (
    fetch_tiger_tracts,
    tiger_block_groups,
    tiger_demographics,
    tiger_tracts,
)

__all__: list[str] = [
    "build_spacetime_panel",
    "compute_nlcd_impervious_by_tract",
    "fetch_osm_risk_proxies",
    "fetch_qcew",
    "fetch_tiger_tracts",
    "fetch_wonder",
    "fhfa_hpi",
    "hubnashville_311",
    "hubnashville_assessor",
    "hubnashville_parcels",
    "hubnashville_permits",
    "hubnashville_usd",
    "hud_county",
    "load_building_footprints",
    "load_fhfa_hpi",
    "load_hud_subsidized_households",
    "load_padus",
    "osm_industrial_footprints",
    "osm_transit_stops",
    "tiger_block_groups",
    "tiger_demographics",
    "tiger_tracts",
]
