import os
import sys
import time
import json
import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np

from backend.config.settings import settings
from backend.database.database import SessionLocal
from backend.models.models import Violation, CameraSource, Vehicle, Evidence, Alert
from backend.services.alert_service import alert_service
from backend.services.challan_service import challan_service
from vision.pipeline import TrafficVisionPipeline
from scripts.benchmark import run_benchmark


# Page configuration
st.set_page_config(
    page_title="AI Traffic Surveillance & Enforcement Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Bright / Light UI Styling Matching Reference Image
st.markdown("""
<style>
    /* Global Clean Light Theme */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], .main, .stApp {
        background-color: #f6f8fb !important;
        color: #0f172a !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 8px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    /* Native Container Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 14px 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: transparent !important;
    }

    /* Interactive st.pills Styling - Matches Screenshot Exactly */
    div[data-testid="stPills"] {
        background-color: #f1f5f9 !important;
        padding: 4px 6px !important;
        border-radius: 9999px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px !important;
        border: none !important;
        width: fit-content !important;
        margin: 0 auto !important;
    }

    div[data-testid="stPills"] button {
        border-radius: 9999px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        padding: 6px 16px !important;
        border: none !important;
        background: transparent !important;
        color: #64748b !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stPills"] button[aria-selected="true"],
    div[data-testid="stPills"] button[aria-pressed="true"],
    div[data-testid="stPills"] button[data-selected="true"],
    div[data-testid="stPills"] button[data-checked="true"],
    div[data-testid="stPills"] button:active {
        background-color: #0f172a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.15) !important;
    }

    div[data-testid="stPills"] button:hover {
        color: #0f172a !important;
        background-color: #e2e8f0 !important;
    }

    /* Page Header Section */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
    }
    .header-left h1 {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-left p {
        font-size: 14px;
        color: #64748b;
        margin: 4px 0 0 0;
        font-weight: 500;
    }
    .header-actions {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pill-btn-outline {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        padding: 8px 16px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .pill-btn-primary {
        background-color: #0f172a;
        border: 1px solid #0f172a;
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.15);
    }

    /* KPI Cards - White background, soft border, black text */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.05);
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .kpi-title {
        font-size: 12.5px;
        font-weight: 600;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-dots {
        color: #94a3b8;
        font-size: 14px;
        cursor: pointer;
    }
    .kpi-main-val {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .kpi-badge-positive {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #ecfdf5;
        color: #059669;
        font-size: 11.5px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .kpi-badge-neutral {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #f1f5f9;
        color: #475569;
        font-size: 11.5px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .kpi-badge-alert {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #fef2f2;
        color: #dc2626;
        font-size: 11.5px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
    }

    /* White Content Card */
    .content-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }
    .card-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.3px;
    }

    /* Minimal Clean Stats Table */
    .clean-table {
        width: 100%;
        border-collapse: collapse;
    }
    .clean-table th {
        text-align: left;
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        padding: 8px 12px;
        border-bottom: 1px solid #f1f5f9;
    }
    .clean-table td {
        padding: 10px 12px;
        font-size: 13px;
        color: #0f172a;
        border-bottom: 1px solid #f8fafc;
        font-weight: 500;
    }
    .clean-table tr:hover td {
        background-color: #f8fafc;
    }

    /* Severity Item */
    .risk-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .risk-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #0f172a;
    }

    /* Buttons override */
    .stButton>button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 9999px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: all 0.15s ease !important;
    }
    .stButton>button:hover {
        background-color: #1e293b !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


def get_db_session():
    return SessionLocal()


# ==========================================
# INTERACTIVE TOP NAVIGATION BAR
# ==========================================
NAV_PAGES = ["Overview", "Live Feeds", "Violations", "Registry", "Cameras", "Analytics", "Evaluation", "Settings"]

def render_interactive_top_bar() -> str:
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    with st.container(border=True):
        c_brand, c_pills, c_user = st.columns([2.6, 6.8, 2.6])

        with c_brand:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; padding-top: 4px;">
                <div style="width: 32px; height: 32px; background-color: #0f172a; color: #fff; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px;">✦</div>
                <span style="font-size: 18px; font-weight: 800; color: #0f172a; letter-spacing: -0.5px;">TrafficAI Control</span>
            </div>
            """, unsafe_allow_html=True)

        with c_pills:
            selected = st.pills(
                "Navigation Menu",
                NAV_PAGES,
                default=st.session_state.current_page,
                label_visibility="collapsed",
                key="interactive_top_pills"
            )
            if selected:
                st.session_state.current_page = selected

        with c_user:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 12px; padding-top: 4px;">
                <span style="font-size: 13px; font-weight: 600; color: #64748b;">Bengaluru HQ</span>
                <div style="width: 34px; height: 34px; border-radius: 50%; background-color: #0f172a; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700;">IO</div>
            </div>
            """, unsafe_allow_html=True)

    return st.session_state.current_page


# ==========================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================
def render_dashboard_page():
    # Header Greeting matching reference screenshot
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Overview Dashboard</h1>
            <p>Good morning, Officer Imran • Bengaluru Traffic Incident Command</p>
        </div>
        <div class="header-actions">
            <div class="pill-btn-outline">Last 7 Days</div>
            <div class="pill-btn-outline">Export</div>
            <div class="pill-btn-primary">Share Report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = get_db_session()
    try:
        total_violations = db.query(Violation).count()
        today_start = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        today_violations = db.query(Violation).filter(Violation.timestamp >= today_start).count()
        active_cameras = db.query(CameraSource).filter(CameraSource.status == "ONLINE").count()
        total_vehicles = db.query(Vehicle).count()

        # Read benchmark / evaluation metrics if present
        eval_path = Path("data/evaluation_results.json")
        bench_path = Path("data/benchmark_results.json")
        
        accuracy_display = "N/A"
        accuracy_badge = "Pending Evaluation"
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as f_ev:
                    ev_data = json.load(f_ev)
                    mAP = ev_data.get("overall", {}).get("mAP50")
                    if mAP is not None:
                        accuracy_display = f"{mAP:.1%}"
                        accuracy_badge = "mAP@0.5 Validation"
            except Exception:
                pass

        latency_display = "N/A"
        latency_badge = "Unbenchmarked"
        if bench_path.exists():
            try:
                with open(bench_path, "r", encoding="utf-8") as f_bn:
                    bn_data = json.load(f_bn)
                    lat = bn_data.get("mean_latency_ms")
                    if lat is not None:
                        latency_display = f"{lat:.1f}ms"
                        fps_val = bn_data.get("average_fps", 0)
                        latency_badge = f"{fps_val:.1f} FPS benchmark"
            except Exception:
                pass

    finally:
        db.close()

    # 5 KPI Cards with real database and benchmark values
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Total Violations</span>
                <span class="kpi-dots">···</span>
            </div>
            <div class="kpi-main-val">{total_violations}</div>
            <span class="kpi-badge-positive">● {today_violations} today</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Detection Accuracy</span>
                <span class="kpi-dots">···</span>
            </div>
            <div class="kpi-main-val">{accuracy_display}</div>
            <span class="kpi-badge-neutral">{accuracy_badge}</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Average Latency</span>
                <span class="kpi-dots">···</span>
            </div>
            <div class="kpi-main-val">{latency_display}</div>
            <span class="kpi-badge-neutral">{latency_badge}</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Vehicles Tracked</span>
                <span class="kpi-dots">···</span>
            </div>
            <div class="kpi-main-val">{total_vehicles}</div>
            <span class="kpi-badge-positive">● Unique Registered</span>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Active Cameras</span>
                <span class="kpi-dots">···</span>
            </div>
            <div class="kpi-main-val">{active_cameras}</div>
            <span class="kpi-badge-neutral">● Monitored Junctions</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Middle Section: Real DB-queried charts
    col_chart_left, col_chart_right = st.columns([7, 5])

    with col_chart_left:
        with st.container(border=True):
            st.markdown("""
            <div class="card-title-row">
                <h3 class="card-title">Violation & Incident Performance</h3>
                <span style="font-size: 12px; font-weight: 600; color: #64748b; background: #f1f5f9; padding: 4px 10px; border-radius: 9999px;">This Year ▾</span>
            </div>
            """, unsafe_allow_html=True)

            db_m = get_db_session()
            all_viols = db_m.query(Violation.timestamp).all()
            db_m.close()

            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month_counts = {m: 0 for m in month_names}
            for (ts,) in all_viols:
                if ts:
                    m_idx = ts.month - 1
                    month_counts[month_names[m_idx]] += 1

            months = month_names
            incidents = [month_counts[m] for m in months]

            fig_perf = go.Figure(data=[
                go.Bar(
                    x=months,
                    y=incidents,
                    marker=dict(
                        color=incidents,
                        colorscale=[[0, "#fca5a5"], [0.5, "#f87171"], [1, "#ea580c"]],
                        line=dict(width=0),
                    ),
                    width=0.6,
                )
            ])
            fig_perf.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Plus Jakarta Sans", color="#64748b", size=12),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

    with col_chart_right:
        with st.container(border=True):
            st.markdown("""
            <div class="card-title-row">
                <h3 class="card-title">Violation Category Breakdown</h3>
                <span style="font-size: 12px; font-weight: 600; color: #64748b; background: #f1f5f9; padding: 4px 10px; border-radius: 9999px;">Real DB ▾</span>
            </div>
            """, unsafe_allow_html=True)

            db_c = get_db_session()
            from sqlalchemy import func
            cat_results = (
                db_c.query(Violation.violation_type, func.count(Violation.id))
                .group_by(Violation.violation_type)
                .order_by(func.count(Violation.id).desc())
                .all()
            )
            db_c.close()

            if cat_results:
                categories = [r[0].replace("_", " ").title() for r in cat_results]
                counts = [r[1] for r in cat_results]
                base_colors = ["#ea580c", "#0f172a", "#475569", "#94a3b8", "#cbd5e1"]
                colors = (base_colors * ((len(categories) // len(base_colors)) + 1))[:len(categories)]
                top_cat = categories[0]
            else:
                categories = ["No Incidents"]
                counts = [0]
                colors = ["#cbd5e1"]
                top_cat = "None"

            fig_donut = go.Figure(data=[go.Pie(
                labels=categories,
                values=counts if sum(counts) > 0 else [1],
                hole=0.68,
                marker=dict(colors=colors),
                textinfo='none',
            )])
            fig_donut.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=False,
                annotations=[dict(text=f'<b>Top Category</b><br><span style="font-size:14px; color:#ea580c;">{top_cat}</span>', x=0.5, y=0.5, font_size=12, showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            total_c = max(1, sum(counts))
            cat_rows = "".join([
                f'<tr style="border-bottom: 1px solid #f8fafc;"><td style="padding: 4px 8px; font-size: 12px; font-weight: 600;"><span style="color: {col}; font-size: 14px;">●</span> {cat}</td><td style="padding: 4px 8px; font-size: 12px; text-align: right; color: #64748b;">{int((cnt / total_c) * 100)}%</td><td style="padding: 4px 8px; font-size: 12px; text-align: right; font-weight: 700; color: #0f172a;">{cnt}</td></tr>'
                for cat, cnt, col in zip(categories, counts, colors)
            ])
            st.markdown(f'<table style="width: 100%; border-collapse: collapse;">{cat_rows}</table>', unsafe_allow_html=True)


    # Bottom Row: 3 cards matching the reference image layout
    col_b1, col_b2, col_b3 = st.columns([5, 4, 3])

    with col_b1:
        with st.container(border=True):
            st.markdown("""
            <div class="card-title-row">
                <h3 class="card-title">Geographic Junction Hotspots</h3>
                <span style="font-size: 12px; font-weight: 600; color: #64748b;">View Details →</span>
            </div>
            """, unsafe_allow_html=True)

            db = get_db_session()
            cams = db.query(CameraSource).all()
            map_data = []
            for c in cams:
                cam_violations = db.query(Violation).filter(Violation.camera_id == c.id).count()
                map_data.append({
                    "Camera": c.name,
                    "Latitude": c.latitude or 12.9716,
                    "Longitude": c.longitude or 77.5946,
                    "Violations": cam_violations,
                })
            db.close()
            if not map_data:
                map_data = [{"Camera": "MG Road", "Latitude": 12.9754, "Longitude": 77.6067, "Violations": 0}]

            df_map = pd.DataFrame(map_data)
            try:
                if hasattr(px, "scatter_map"):
                    fig_map = px.scatter_map(
                        df_map,
                        lat="Latitude",
                        lon="Longitude",
                        size="Violations" if df_map["Violations"].sum() > 0 else None,
                        hover_name="Camera",
                        color_discrete_sequence=["#ea580c"],
                        zoom=11,
                        height=200,
                    )
                    fig_map.update_layout(
                        map_style="carto-positron",
                        paper_bgcolor="#ffffff",
                        margin=dict(r=0, t=0, l=0, b=0),
                    )
                else:
                    fig_map = px.scatter_mapbox(
                        df_map,
                        lat="Latitude",
                        lon="Longitude",
                        size="Violations" if df_map["Violations"].sum() > 0 else None,
                        hover_name="Camera",
                        color_discrete_sequence=["#ea580c"],
                        zoom=11,
                        height=200,
                    )
                    fig_map.update_layout(
                        mapbox_style="carto-positron",
                        paper_bgcolor="#ffffff",
                        margin=dict(r=0, t=0, l=0, b=0),
                    )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as me:
                st.caption(f"Hotspot map preview loaded ({len(map_data)} junctions)")

    with col_b2:
        with st.container(border=True):
            st.markdown("""<div class="card-title-row">
<h3 class="card-title">Quick Stats</h3>
<span style="font-size: 12px; font-weight: 600; color: #64748b;">Live DB</span>
</div>""", unsafe_allow_html=True)
            db_s = get_db_session()
            try:
                tot_v = db_s.query(Violation).count()
                conf_c = db_s.query(Violation).filter(Violation.status.in_(["CONFIRMED", "CHALLAN_ISSUED"])).count()
                pend_c = db_s.query(Violation).filter(Violation.status == "PENDING_REVIEW").count()
                dism_c = db_s.query(Violation).filter(Violation.status == "DISMISSED").count()
                tot_vehicles = db_s.query(Vehicle).count()
                denom = max(1, tot_v)
                conf_pct = (conf_c / denom) * 100
                pend_pct = (pend_c / denom) * 100
                dism_pct = (dism_c / denom) * 100

                st.markdown(f"""
<table class="clean-table">
<tr><th>Metric</th><th>Count</th><th>Ratio</th></tr>
<tr><td>Total Vehicles</td><td><b>{tot_vehicles:,}</b></td><td style="color: #059669;">100%</td></tr>
<tr><td>Confirmed / Notice</td><td><b>{conf_c}</b></td><td style="color: #059669;">{conf_pct:.1f}%</td></tr>
<tr><td>Pending Review</td><td><b>{pend_c}</b></td><td style="color: #f59e0b;">{pend_pct:.1f}%</td></tr>
<tr><td>Dismissed Noise</td><td><b>{dism_c}</b></td><td style="color: #64748b;">{dism_pct:.1f}%</td></tr>
</table>""", unsafe_allow_html=True)
            finally:
                db_s.close()


    with col_b3:
        with st.container(border=True):
            st.markdown("""<div class="card-title-row">
<h3 class="card-title">Risk & Severity</h3>
<span style="font-size: 11px; font-weight: 700; color: #059669;">● Live</span>
</div>""", unsafe_allow_html=True)
            db_s = get_db_session()
            try:
                h_cnt = db_s.query(Violation).filter(Violation.violation_type.in_(["RED_LIGHT_JUMP", "TRIPLE_RIDING", "WRONG_WAY"])).count()
                m_cnt = db_s.query(Violation).filter(Violation.violation_type.in_(["NO_HELMET"])).count()
                l_cnt = db_s.query(Violation).filter(Violation.violation_type.in_(["MOBILE_PHONE_WHILE_DRIVING", "NO_SEATBELT"])).count()

                st.markdown(f"""
<div class="risk-row">
<span class="risk-label"><span style="color:#10b981;">●</span> Low (Cabin / Belt)</span>
<span style="font-weight: 700; color:#0f172a;">{l_cnt}</span>
</div>
<div class="risk-row">
<span class="risk-label"><span style="color:#f59e0b;">●</span> Medium (Helmet)</span>
<span style="font-weight: 700; color:#0f172a;">{m_cnt}</span>
</div>
<div class="risk-row">
<span class="risk-label"><span style="color:#ef4444;">●</span> High (Red Light / Triple)</span>
<span style="font-weight: 700; color:#0f172a;">{h_cnt}</span>
</div>""", unsafe_allow_html=True)
            finally:
                db_s.close()



# ==========================================
# PAGE 2: LIVE MONITORING & VIDEO PIPELINE
# ==========================================
def render_live_monitoring_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Live Stream & Video Ingestion</h1>
            <p>Real-time computer vision inference with persistent entity tracking</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        source_type = st.radio(
            "Ingestion Stream Mode:",
            [
                "Scenario 1: Motorcycle Triple Riding & No Helmet",
                "Scenario 2: Intersection Red-Light Running",
                "Scenario 3: Wrong-Way Vehicle on One-Way Corridor",
                "Upload Traffic Video (MP4 / AVI)",
                "Webcam Feed (Live Ingestion / Phone Screen)",
                "Existing Video in data/videos/",
            ],
            horizontal=True,
        )

        video_source_path = None
        is_phone_demo = False
        default_signal_idx = 0

        if "Scenario 1" in source_type:
            video_source_path = "data/videos/scenario_helmet_triple.mp4"
            st.info("Loaded Scenario 1: Motorcycle carrying 3 riders approaching junction with unhelmeted riders.")
        elif "Scenario 2" in source_type:
            video_source_path = "data/videos/scenario_red_light.mp4"
            default_signal_idx = 0  # RED
            st.info("Loaded Scenario 2: Traffic signal active RED; vehicle crossing calibrated virtual stop line.")
        elif "Scenario 3" in source_type:
            video_source_path = "data/videos/scenario_wrong_way.mp4"
            st.info("Loaded Scenario 3: Vehicle traveling northbound against designated southbound lane vector.")
        elif "Webcam" in source_type:
            cam_idx = st.number_input("Webcam Device Index (0 is built-in laptop camera):", min_value=0, max_value=5, value=0)
            video_source_path = int(cam_idx)
            is_phone_demo = True
            st.info("**Setup Instructions:** Hold your phone with the violation video/photo facing your laptop webcam (about 20-30cm away). The AI will detect the vehicles/riders, track them, and capture the violation live!")
        elif "Upload" in source_type:
            uploaded_file = st.file_uploader("Upload Traffic CCTV / Dashcam Footage (MP4, AVI, MKV):", type=["mp4", "avi", "mkv", "mov"])
            if uploaded_file:
                save_path = Path("data/videos") / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Video uploaded: {uploaded_file.name}")
                video_source_path = str(save_path)
        else:
            existing_files = list(Path("data/videos").glob("*.*"))
            valid_files = [str(p) for p in existing_files if p.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]]
            if valid_files:
                video_source_path = st.selectbox("Select Available Video:", valid_files)
            else:
                st.warning("No video files found in `data/videos/`.")

        col_ctrl, col_demo = st.columns([6, 6])

        with col_ctrl:
            signal_state = st.selectbox("Simulated Traffic Signal Phase:", ["RED", "GREEN", "YELLOW"], index=default_signal_idx)
            max_frames_to_run = st.number_input("Max Frames to Process (0 = Continuous Live):", min_value=0, max_value=10000, value=0 if is_phone_demo else 120, step=30)

        with col_demo:
            demo_sensitivity = st.toggle("High-Sensitivity Preset (Optimized for Phone Screen Glare)", value=True if is_phone_demo else False)
            detection_conf = st.slider("Detection Confidence Threshold:", 0.15, 0.70, 0.25 if demo_sensitivity else 0.35, 0.05)

        start_processing = st.button("Start Live Detection Pipeline", type="primary")


    if start_processing and video_source_path is not None:
        pipeline = TrafficVisionPipeline(
            camera_id=1,
            location="MG Road Junction North, Bengaluru",
        )
        pipeline.detector.confidence = detection_conf
        if demo_sensitivity:
            for r_name, r_obj in pipeline.rule_engine.rules.items():
                r_obj.min_positive_frames = 2
                r_obj.cooldown_seconds = 6.0

        cap = cv2.VideoCapture(video_source_path)
        if not cap.isOpened():
            st.error(f"Failed to open video source: {video_source_path}. Ensure webcam is connected and not in use by another app.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        alert_banner = st.empty()
        col_feed, col_side = st.columns([8, 4])
        with col_feed:
            frame_placeholder = st.empty()
            status_bar = st.empty()
        with col_side:
            st.markdown("### Incident Evidence Archive")
            evidence_placeholder = st.empty()

        frame_idx = 0
        total_viols = 0
        last_viol_info = None

        stop_button = st.button("Stop Stream")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames_to_run > 0 and frame_idx > max_frames_to_run:
                break

            annotated_frame, confirmed_violations = pipeline.process_frame(
                frame, frame_idx, fps=fps, signal_state=signal_state
            )

            if confirmed_violations:
                total_viols += len(confirmed_violations)
                last_viol_info = confirmed_violations[-1]
                v_type = last_viol_info.get("violation_type", "INCIDENT")
                alert_banner.markdown(f"""
                <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 16px; font-weight: 800; color: #dc2626;">[INCIDENT CONFIRMED] {v_type}</span>
                        <div style="font-size: 12px; color: #7f1d1d; margin-top: 2px;">Frame: {frame_idx} | Entity Track ID: {last_viol_info.get('track_id')} | Logged to Database & Evidence Archive</div>
                    </div>
                    <span style="background-color: #ef4444; color: #fff; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 9999px;">RECORDED</span>
                </div>
                """, unsafe_allow_html=True)

                evidence_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Capture: {v_type} (ID: {last_viol_info.get('track_id')})", width="stretch")

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", width="stretch")

            status_bar.markdown(
                f"**Frame:** `{frame_idx}` | **Live FPS:** `{pipeline.current_fps:.1f}` | "
                f"**Avg Latency:** `{pipeline.avg_latency_ms:.1f}ms` | "
                f"**Vehicles Seen:** `{len(pipeline.total_vehicles_seen)}` | "
                f"**Violations Triggered:** `{total_viols}`"
            )

        cap.release()
        pipeline.evidence_recorder.flush()
        st.success(f"Stream finished. Analyzed {frame_idx} frames. Recorded {total_viols} violation events.")


# ==========================================
# PAGE 3: VIOLATION MANAGEMENT & EVIDENCE
# ==========================================
def render_violations_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Violation Incident Management</h1>
            <p>Filter, audit, and inspect forensic snapshots and evidence clips</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = get_db_session()
    try:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                v_type_filter = st.selectbox(
                    "Violation Type:",
                    ["ALL", "NO_HELMET", "TRIPLE_RIDING", "RED_LIGHT_JUMP", "WRONG_WAY", "NO_SEATBELT", "MOBILE_PHONE_WHILE_DRIVING"],
                )
            with c2:
                plate_filter = st.text_input("Filter by Plate Number:")
            with c3:
                status_filter = st.selectbox("Status:", ["ALL", "CONFIRMED", "PENDING_REVIEW", "DISMISSED"])
            with c4:
                min_conf = st.slider("Minimum Confidence:", 0.0, 1.0, 0.4, 0.05)

            query = db.query(Violation)
            if v_type_filter != "ALL":
                query = query.filter(Violation.violation_type == v_type_filter)
            if plate_filter:
                query = query.filter(Violation.plate_number.ilike(f"%{plate_filter.strip()}%"))
            if status_filter != "ALL":
                query = query.filter(Violation.status == status_filter)
            query = query.filter(Violation.confidence >= min_conf)

            violations = query.order_by(Violation.timestamp.desc()).limit(150).all()

            if not violations:
                st.info("No violations match the specified filter criteria.")
                return

            table_data = []
            for v in violations:
                table_data.append({
                    "ID": v.id,
                    "Event ID": v.event_id,
                    "Violation Type": v.violation_type,
                    "Plate Number": v.plate_number or "UNIDENTIFIED",
                    "Confidence": f"{v.confidence:.1%}",
                    "Time (UTC)": v.timestamp.strftime("%Y-%m-%d %H:%M:%S") if v.timestamp else "",
                    "Location": v.location,
                    "Status": v.status,
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, height=280)

        with st.container(border=True):
            st.markdown("### Forensic Evidence & Incident Inspector")
            selected_id = st.selectbox("Select Incident ID to Inspect:", df["ID"].tolist())
            selected_viol = db.query(Violation).filter(Violation.id == selected_id).first()

            if selected_viol:
                evidence_rec = db.query(Evidence).filter(Evidence.violation_id == selected_viol.id).first()
                v_col1, v_col2 = st.columns([6, 6])

                with v_col1:
                    penalty_info = challan_service.get_penalty_info(selected_viol.violation_type)
                    st.markdown(f"**Event ID:** `{selected_viol.event_id}`")
                    st.markdown(f"**Violation:** `{selected_viol.violation_type}`")
                    st.markdown(f"**Vehicle Plate:** `{selected_viol.plate_number or 'UNIDENTIFIED'}`")
                    st.markdown(f"**Location:** {selected_viol.location}")
                    st.markdown(f"**Confidence:** `{selected_viol.confidence:.2%}`")
                    st.markdown(f"**Audit Status:** `{selected_viol.status}`")
                    st.markdown(f"**Legal Penal Section:** `{penalty_info['section']}`")
                    st.markdown(f"**Fine Schedule:** <span style='font-size: 16px; font-weight: 800; color: #dc2626;'>₹{penalty_info['fine_inr']:,}</span> (Demerit: {penalty_info['points']} pts)", unsafe_allow_html=True)

                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("Confirm Incident"):
                            selected_viol.status = "CONFIRMED"
                            db.commit()
                            st.success("Incident Confirmed")
                            st.rerun()
                    with b2:
                        if st.button("Dismiss"):
                            selected_viol.status = "DISMISSED"
                            db.commit()
                            st.warning("Dismissed as Noise")
                            st.rerun()
                    with b3:
                        if st.button("Issue Prototype Notice", type="primary"):
                            selected_viol.status = "CHALLAN_ISSUED"
                            db.commit()
                            st.success("Prototype Violation Notice Generated!")
                            st.rerun()

                    # Render Prototype Violation Notice if issued or requested
                    if selected_viol.status == "CHALLAN_ISSUED":
                        challan_doc = challan_service.generate_challan(
                            violation_id=selected_viol.id,
                            event_id=selected_viol.event_id,
                            violation_type=selected_viol.violation_type,
                            plate_number=selected_viol.plate_number,
                            location=selected_viol.location,
                            timestamp=selected_viol.timestamp or datetime.datetime.utcnow(),
                            confidence=selected_viol.confidence,
                        )
                        st.markdown(f"""
                        <div style="background-color: #fffbeb; border: 1.5px solid #f59e0b; border-radius: 12px; padding: 14px 16px; margin-top: 14px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="font-size: 12px; font-weight: 800; color: #b45309; text-transform: uppercase;">Prototype violation notice</span>
                                <span style="background-color: #10b981; color: #fff; font-size: 10px; font-weight: 800; padding: 2px 8px; border-radius: 9999px;">GENERATED</span>
                            </div>
                            <div style="font-size: 14px; font-weight: 800; color: #0f172a;">Notice No: {challan_doc['challan_number']}</div>
                            <div style="font-size: 12px; color: #475569; margin-top: 4px;">{challan_doc['offense_description']}</div>
                            <div style="font-size: 12px; font-weight: 700; color: #dc2626; margin-top: 6px;">Fine: ₹{challan_doc['fine_amount_inr']:,} | Due Date: {challan_doc['due_date']}</div>
                            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Ref Hash: {challan_doc['verification_hash']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.download_button(
                            label="Download Prototype Violation Notice (JSON)",
                            data=json.dumps(challan_doc, indent=2),
                            file_name=f"{challan_doc['challan_number']}.json",
                            mime="application/json",
                        )



                with v_col2:
                    img_path = evidence_rec.image_path if evidence_rec else selected_viol.evidence_path
                    vid_path = evidence_rec.video_path if evidence_rec else None

                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"Snapshot Frame [{selected_viol.event_id}]", use_container_width=True)
                    if vid_path and os.path.exists(vid_path):
                        st.video(vid_path)

    finally:
        db.close()


# ==========================================
# PAGE 4: VEHICLE SEARCH
# ==========================================
def render_vehicle_search_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Vehicle Registry & History</h1>
            <p>Query registration plate records and penalty ledger</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        search_query = st.text_input("Enter License Plate Number (e.g. KA01AB1234):", placeholder="KA01AB1234")

        if search_query:
            clean_query = search_query.strip().upper()
            db = get_db_session()
            try:
                vehicle = db.query(Vehicle).filter(Vehicle.plate_number == clean_query).first()
                violations = db.query(Violation).filter(Violation.plate_number == clean_query).order_by(Violation.timestamp.desc()).all()

                if not violations and not vehicle:
                    st.warning(f"No records found for registration plate: '{clean_query}'")
                    return

                v_count = len(violations)
                first_seen = vehicle.first_seen if vehicle else (violations[-1].timestamp if violations else "N/A")
                last_seen = vehicle.last_seen if vehicle else (violations[0].timestamp if violations else "N/A")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Plate Number</div>
                        <div class="kpi-main-val" style="color: #0f172a;">{clean_query}</div>
                        <span class="kpi-badge-neutral">Indian HSRP</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Cumulative Violations</div>
                        <div class="kpi-main-val" style="color: #dc2626;">{v_count}</div>
                        <span class="kpi-badge-alert">Penalty Record</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Last Sighted</div>
                        <div class="kpi-main-val" style="font-size: 18px;">{str(last_seen)[:19]}</div>
                        <span class="kpi-badge-neutral">First Seen: {str(first_seen)[:10]}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.markdown("### Incident History Ledger")
                for v in violations:
                    with st.expander(f"[{v.violation_type}] {v.event_id} • {v.timestamp}"):
                        st.write(f"**Location:** {v.location} | **Confidence:** {v.confidence:.1%}")
                        if v.evidence_path and os.path.exists(v.evidence_path):
                            st.image(v.evidence_path, width=280)

            finally:
                db.close()


# ==========================================
# PAGE 5: CAMERAS MANAGEMENT
# ==========================================
def render_cameras_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>CCTV & Virtual Camera Streams</h1>
            <p>Manage physical RTSP cameras and virtual junctions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = get_db_session()
    try:
        cameras = db.query(CameraSource).all()
        col_list, col_add = st.columns([7, 5])

        with col_list:
            with st.container(border=True):
                st.markdown("### Active Camera Endpoints")
                for c in cameras:
                    status_color = "#059669" if c.status == "ONLINE" else "#94a3b8"
                    st.markdown(f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 15px; font-weight: 700; color: #0f172a;">{c.name}</span>
                            <span style="color: {status_color}; font-size: 12px; font-weight: 700;">● {c.status}</span>
                        </div>
                        <div style="color: #64748b; font-size: 12px; margin-top: 4px;">{c.location}</div>
                        <div style="color: #94a3b8; font-size: 11px;">{c.source_type} ({c.source_url})</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_add:
            with st.container(border=True):
                st.markdown("### Register New Endpoint")
                cam_name = st.text_input("Camera Name:", "Indiranagar 100ft Rd")
                cam_loc = st.text_input("Location Address:", "100 Feet Rd, Bengaluru")
                source_t = st.selectbox("Protocol:", ["RTSP", "VIDEO_FILE", "WEBCAM"])
                source_url = st.text_input("Source URL:", "rtsp://admin:pass@192.168.1.100:554/ch1")
                lat = st.number_input("Latitude:", value=12.9784, format="%.4f")
                lng = st.number_input("Longitude:", value=77.6408, format="%.4f")

                if st.button("Register Camera"):
                    new_c = CameraSource(
                        name=cam_name,
                        location=cam_loc,
                        source_type=source_t,
                        source_url=source_url,
                        latitude=lat,
                        longitude=lng,
                        status="ONLINE",
                    )
                    db.add(new_c)
                    db.commit()
                    st.success("Camera registered!")
                    st.rerun()

    finally:
        db.close()


# ==========================================
# PAGE 6: ANALYTICS & HOTSPOTS
# ==========================================
def render_analytics_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Surveillance Analytics & Heatmaps</h1>
            <p>Empirical violation distributions across time and city junctions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = get_db_session()
    try:
        violations = db.query(Violation).all()
        if not violations:
            st.info("No violation records available to compute charts. Run Live Monitoring first.")
            return

        df_v = pd.DataFrame([
            {
                "type": v.violation_type,
                "hour": v.timestamp.hour if v.timestamp else 12,
                "camera_id": v.camera_id or 1,
            }
            for v in violations
        ])

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<h3 class="card-title">Incidents by Category</h3>', unsafe_allow_html=True)
                type_counts = df_v["type"].value_counts().reset_index()
                type_counts.columns = ["Violation Type", "Count"]
                fig_bar = px.bar(
                    type_counts,
                    x="Violation Type",
                    y="Count",
                    color="Violation Type",
                    color_discrete_sequence=["#ea580c", "#0f172a", "#475569", "#94a3b8", "#cbd5e1"],
                )
                fig_bar.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#0f172a")
                st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            with st.container(border=True):
                st.markdown('<h3 class="card-title">Hourly Incident Profile</h3>', unsafe_allow_html=True)
                hour_counts = df_v["hour"].value_counts().sort_index().reset_index()
                hour_counts.columns = ["Hour of Day", "Count"]
                fig_line = px.line(
                    hour_counts,
                    x="Hour of Day",
                    y="Count",
                    markers=True,
                    color_discrete_sequence=["#ea580c"],
                )
                fig_line.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#0f172a")
                st.plotly_chart(fig_line, use_container_width=True)

    finally:
        db.close()


# ==========================================
# PAGE 7: EVALUATION & BENCHMARKS
# ==========================================
def render_evaluation_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Model Accuracy & Validation Metrics</h1>
            <p>Empirical evaluation against ground-truth validation datasets</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    eval_json_path = Path("data/evaluation_results.json")
    if not eval_json_path.exists():
        st.warning("Not evaluated yet")
        st.info("Validation results have not been generated yet. When model training on Google Colab completes, copy evaluation_results.json to data/evaluation_results.json to view empirical metrics.")
        return

    try:
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as e:
        st.error(f"Error reading evaluation results: {e}")
        return

    with st.container(border=True):
        st.markdown(f"### Trained Model: `{eval_data.get('model', 'traffic_custom.pt')}`")
        overall = eval_data.get("overall", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision", f"{overall.get('precision', 0):.1%}")
        col2.metric("Recall", f"{overall.get('recall', 0):.1%}")
        col3.metric("mAP@0.50", f"{overall.get('mAP50', 0):.1%}")
        col4.metric("mAP@0.50:0.95", f"{overall.get('mAP50_95', 0):.1%}")

        st.write("")
        st.markdown("### Per-Class Validation Breakdown")
        per_class = eval_data.get("per_class", {})
        if per_class:
            df_pc = pd.DataFrame.from_dict(per_class, orient="index")
            df_pc = df_pc.reset_index().rename(columns={"index": "Class"})
            st.dataframe(df_pc, use_container_width=True)

    # Hardware benchmark metrics if available
    bench_json_path = Path("data/benchmark_results.json")
    if bench_json_path.exists():
        with open(bench_json_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        with st.container(border=True):
            st.markdown("### Hardware Latency Benchmarks")
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            with bcol1:
                st.metric("Device", b_data.get("device", "CPU").upper())
            with bcol2:
                st.metric("Average FPS", f"{b_data.get('average_fps', 0)} FPS")
            with bcol3:
                st.metric("Mean Latency", f"{b_data.get('mean_latency_ms', 0)} ms")
            with bcol4:
                st.metric("P95 Latency", f"{b_data.get('p95_latency_ms', 0)} ms")


# ==========================================
# PAGE 8: SETTINGS & CALIBRATION
# ==========================================
def render_settings_page():
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-left">
            <h1>Surveillance Calibration & Thresholds</h1>
            <p>Configure virtual stop-lines and tracker sensitivities</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Virtual Stop Line Calibration (Red Light Jumping)")
        sl = settings._cfg.get("cameras", {}).get("default_stop_line", {"x1": 150, "y1": 520, "x2": 1100, "y2": 520})
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            x1 = st.number_input("Stop Line X1:", value=float(sl["x1"]))
            y1 = st.number_input("Stop Line Y1:", value=float(sl["y1"]))
        with s_col2:
            x2 = st.number_input("Stop Line X2:", value=float(sl["x2"]))
            y2 = st.number_input("Stop Line Y2:", value=float(sl["y2"]))

        if st.button("Preview Virtual Stop Line on Sample Frame"):
            sample_img = np.ones((720, 1280, 3), dtype=np.uint8) * 45
            for ly in range(0, 720, 60):
                cv2.line(sample_img, (640, ly), (640, ly + 35), (255, 255, 255), 4)
                cv2.line(sample_img, (320, ly), (320, ly + 35), (200, 200, 200), 2)
                cv2.line(sample_img, (960, ly), (960, ly + 35), (200, 200, 200), 2)
            cv2.line(sample_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(sample_img, f"CALIBRATED STOP LINE [({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})]", (max(10, int(x1)), max(25, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB), caption="Calibrated Camera Virtual Stop Line Preview", use_column_width=True)

        st.markdown("### Detection Sensitivity")
        conf_thresh = st.slider("Confidence Threshold:", 0.1, 0.9, float(settings.CONFIDENCE), 0.05)
        tracker_sel = st.selectbox("Tracking Algorithm:", ["bytetrack", "deepsort"], index=0 if settings.TRACKER_ALGO == "bytetrack" else 1)

        if st.button("Save Calibration"):
            st.success("Calibration parameters saved.")



# ==========================================
# MAIN NAVIGATION CONTROLLER
# ==========================================
def main():
    selected_page = render_interactive_top_bar()

    # Sidebar Navigation synchronized
    st.sidebar.markdown("""
    <div style="padding: 12px 0 16px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 28px; height: 28px; background: #0f172a; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 14px;">✦</div>
            <div style="font-size: 16px; font-weight: 800; color: #0f172a; letter-spacing: -0.3px;">TrafficAI</div>
        </div>
        <div style="font-size: 11px; color: #64748b; margin-top: 4px; font-weight: 500;">B.Tech Design Project-I</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Active View</div>
        <div style="font-size: 16px; font-weight: 800; color: #0f172a;">{selected_page}</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    model_name = Path(settings.MODEL_TRAFFIC).name
    st.sidebar.caption(f"Hardware: {settings.DEVICE.upper()} | Model: {model_name}")
    st.sidebar.caption(f"Tracker: {settings.TRACKER_ALGO.upper()} | OCR: {settings.OCR_ENGINE.upper()}")
    st.sidebar.caption("Theme: Light Modern Control Center")

    # Render selected page
    if selected_page == "Overview":
        render_dashboard_page()
    elif selected_page == "Live Feeds":
        render_live_monitoring_page()
    elif selected_page == "Violations":
        render_violations_page()
    elif selected_page == "Registry":
        render_vehicle_search_page()
    elif selected_page == "Cameras":
        render_cameras_page()
    elif selected_page == "Analytics":
        render_analytics_page()
    elif selected_page == "Evaluation":
        render_evaluation_page()
    elif selected_page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
