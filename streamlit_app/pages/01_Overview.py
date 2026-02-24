"""Overview page with all seven view summaries."""

from __future__ import annotations

import streamlit as st

from streamlit_app.data.loaders import get_view_metadata

st.title("TN Public Policy Analytics — Overview")
st.caption("Seven high-value, precomputed geospatial policy views.")

meta = get_view_metadata()
for view_id, info in meta.items():
    with st.container(border=True):
        st.markdown(f"### {view_id}. {info['name']}")
        st.write(f"Chapter: {info['chapter']} · Default level: {info['default_level']}")

st.info("Use View Explorer for map interactions and data export.")
