"""
TrafficAI officer dashboard (Streamlit).

    streamlit run dashboard/app.py

Pages: Overview · Review queue · Cameras · Upload proof · Violations · Vehicles · Analytics · Models · Evaluation ·
Seatbelt Lab · Settings.
Every number on screen comes from the database or from files the system wrote; nothing is hard-coded.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import streamlit as st

from backend.database.database import SessionLocal, ensure_schema
from backend.db_models.models import Violation
from dashboard import ui
from backend.config.settings import settings
from dashboard.views import (analytics, cameras, citizen_upload, evaluation, models_page, overview,
                             review, settings_page, vehicles, violations, seatbelt_test)

st.set_page_config(page_title="TrafficAI · Officer console", page_icon="🚦", layout="wide",
                   initial_sidebar_state="collapsed")
@st.cache_resource(show_spinner=False)
def _schema_ready() -> bool:
    ensure_schema()  # creates/migrates tables; only needed once per server start, not on every click
    return True


_schema_ready()
ui.inject_css()

PAGES = {
    "Overview": overview.render,
    "Review queue": review.render,
    "Cameras": cameras.render,
    "Upload proof": citizen_upload.render,
    "Violations": violations.render,
    "Vehicles": vehicles.render,
    "Analytics": analytics.render,
    "Models": models_page.render,
    "Evaluation": evaluation.render,
    "Seatbelt Lab": seatbelt_test.render,
    "Settings": settings_page.render,
}
OFFICERS = ["Insp. R. Kumar", "SI P. Shetty", "Insp. S. Rao", "Demo officer"]


def pending_count() -> int:
    db = SessionLocal()
    try:
        return db.query(Violation).filter(Violation.status == "PENDING_REVIEW").count()
    finally:
        db.close()


def top_bar() -> str:
    cur_nav = st.session_state.get("nav")
    if cur_nav == "Live cameras":
        st.session_state.cam_tab = "Live detection"
        st.session_state.nav = "Cameras"
    elif cur_nav == "Camera setup":
        st.session_state.cam_tab = "Calibration"
        st.session_state.nav = "Cameras"
    elif cur_nav not in PAGES:
        st.session_state.nav = "Overview"
    n = pending_count()
    with st.container(key="topbar"):
        c1, c3 = st.columns([5, 1.4], vertical_alignment="center")
        with c1:
            badge = (f'<span style="margin-left:14px;background:#FFF4D6;color:#8A5D00;font-size:13px;font-weight:700;'
                     f'padding:3px 10px;border-radius:12px">{n} waiting for review</span>' if n else "")
            n_models = sum(1 for m in (settings.MODEL_TRAFFIC, settings.MODEL_CUSTOM, "models/triple_riding_best.pt",
                                       getattr(settings, "MODEL_SEATBELT", "models/seatbelt_best.pt")) if Path(m).exists())
            st.markdown('<div class="brand"><span class="lamp"><i style="background:#FF5A4F"></i>'
                        '<i style="background:#4A3B12"></i><i style="background:#1E4D2E"></i></span>TrafficAI'
                        '<span class="sub">Traffic violation monitoring · Bengaluru</span>'
                        f'{badge}<span class="sys"><i></i>{n_models} AI models ready · {ui.esc(settings.DEVICE.upper())}</span>'
                        '</div>', unsafe_allow_html=True)
        with c3:
            st.selectbox("On duty", OFFICERS, key="officer")
        # fixed labels + key: the selected tab never changes when the waiting count changes
        page = st.radio("Page", list(PAGES), key="nav", horizontal=True, label_visibility="collapsed")
    return page


def sidebar(page: str):
    custom = Path(settings.MODEL_CUSTOM)
    sb_path = Path(getattr(settings, "MODEL_SEATBELT", "models/seatbelt_best.pt"))
    with st.sidebar:
        st.markdown('<div class="brand" style="color:#18212B">TrafficAI</div>'
                    '<div class="small">B.Tech Design Project-I · Bengaluru HQ</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat" style="margin:12px 0"><span>Active view</span><b>{ui.esc(page)}</b></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<dl class="kv"><dt>On duty</dt><dd>{ui.esc(st.session_state.get("officer", "—"))}</dd>'
                    f'<dt>Hardware</dt><dd>{ui.esc(settings.DEVICE.upper())}</dd>'
                    f'<dt>Traffic model</dt><dd>{ui.esc(Path(settings.MODEL_TRAFFIC).name)}</dd>'
                    f'<dt>Helmet model</dt><dd>{ui.esc(custom.name if custom.exists() else "not trained yet")}</dd>'
                    f'<dt>Seatbelt model</dt><dd>{ui.esc(sb_path.name if sb_path.exists() else "standby")}</dd>'
                    f'<dt>Tracker</dt><dd>{ui.esc(settings.TRACKER_ALGO)}</dd>'
                    f'<dt>OCR</dt><dd>{ui.esc(settings.OCR_ENGINE)}</dd></dl>', unsafe_allow_html=True)
        st.caption("Detection settings can be changed on the Settings page.")


page = top_bar()
sidebar(page)
PAGES[page]()
