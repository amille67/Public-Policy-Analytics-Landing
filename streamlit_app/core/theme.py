"""Brand styling injection for Streamlit UI."""

from __future__ import annotations

import streamlit as st


def inject_premium_theme() -> None:
    """Inject custom CSS tokens and card styling."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem !important; max-width: 1480px !important;}
        div[data-testid="stMetric"] {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            transition: all 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        }
        .stSelectbox label, .stRadio label {
            font-weight: 600;
            color: #334155;
        }
        .tn-brand {
            font-weight: 700;
            color: #1e40af;
            letter-spacing: 0.2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
