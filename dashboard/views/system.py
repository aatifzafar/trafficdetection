"""System: the administrator pages grouped under one tab - AI models, accuracy, seatbelt test and settings.

Officers work in the other tabs; these pages are for the team configuring and evaluating the system.
"""

import streamlit as st

from dashboard.views import evaluation, models_page, seatbelt_test, settings_page

MODELS, ACCURACY, SEATBELT, SETTINGS = "AI models", "Accuracy", "Seatbelt test (experimental)", "Settings"
TABS = [MODELS, ACCURACY, SEATBELT, SETTINGS]
RENDER = {MODELS: models_page.render, ACCURACY: evaluation.render,
          SEATBELT: seatbelt_test.render, SETTINGS: settings_page.render}


def render():
    current = st.session_state.get("sys_tab", TABS[0])
    if current not in TABS:
        current = TABS[0]
    # links on other pages set sys_tab; carry it into the radio before the radio is drawn
    if st.session_state.get("sys_tab_radio") != current:
        st.session_state["sys_tab_radio"] = current
    with st.container(key="sys_nav_chips"):
        selected = st.radio("System section", TABS, horizontal=True, label_visibility="collapsed",
                            key="sys_tab_radio",
                            on_change=lambda: st.session_state.update(sys_tab=st.session_state.sys_tab_radio))
    st.session_state.sys_tab = selected
    RENDER[selected]()
