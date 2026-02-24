"""Typed Streamlit session state for TN Public Policy Analytics."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class AppState:
    selected_view: str = "Nashville 311 Distress Risk"
    selected_geo_level: str = "lowest"


def init_session_state() -> None:
    """Initialize typed state once per session."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
