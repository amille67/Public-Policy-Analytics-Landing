"""Public Policy Analytics — Python package.

This is the Python reimplementation and national-scale extension of Ken Steif's
*Public Policy Analytics* case studies.  The ``ppa`` package provides shared
utilities for geospatial data engineering, machine learning, visualization, and
open-data access (Census/ACS, BLS, CDC WONDER).

Environment variables (``PPA_DATA_ROOT``, ``CENSUS_API_KEY``, etc.) are loaded
automatically from a ``.env`` file at the repo root via ``python-dotenv`` when
the package is first imported.
"""

from __future__ import annotations

# Load .env at import time so all submodules see the env vars.
# find_dotenv() walks up from cwd, so this works whether the user runs from
# the repo root, a chapter dir, or a Jupyter notebook inside national/.
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

__version__ = "0.2.0"

# Re-export core data-engineering functions so users can do:
#   from ppa import fetch_acs_tracts, standardize_crs
from ppa.data import fetch_acs_tracts, standardize_crs  # noqa: E402

__all__: list[str] = [
    "__version__",
    "fetch_acs_tracts",
    "standardize_crs",
]
