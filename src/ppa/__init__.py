"""Public Policy Analytics — Python package.

This is the Python reimplementation and national-scale extension of Ken Steif's
*Public Policy Analytics* case studies.  The ``ppa`` package provides shared
utilities for geospatial data engineering, machine learning, visualization, and
open-data access (Census/ACS, BLS, CDC WONDER).
"""

from __future__ import annotations

__version__ = "0.2.0"

# Re-export core data-engineering functions so users can do:
#   from ppa import fetch_acs_tracts, standardize_crs
from ppa.data import fetch_acs_tracts, standardize_crs

__all__: list[str] = [
    "__version__",
    "fetch_acs_tracts",
    "standardize_crs",
]
