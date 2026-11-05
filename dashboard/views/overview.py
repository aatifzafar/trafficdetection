"""Overview: the old home page, kept — headline numbers, monthly trend, categories, hotspots, status and severity."""

import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import func

from backend.db_models.models import CameraSource, Vehicle, Violation
from backend.services import review_service as rs
from dashboard import ui
from dashboard.data import session

SIGN = "#1F4E8C"
CAT_COLORS = ["#C62828", "#1F4E8C", "#E0A100", "#2E7D4F", "#7FA3CF", "#5E6975", "#B0BEC5"]
LAYOUT = dict(paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=10, b=10),
              font=dict(family="Barlow, sans-serif", color="#5E6975", size=12))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEVERITY = [  # same grouping as the old overview
    ("High (red light / triple / wrong way / unsafe)", "#C62828", ["RED_LIGHT_JUMP", "TRIPLE_RIDING", "WRONG_WAY", "UNSAFE_MOVEMENT"]),
    ("Medium (helmet / parking)", "#E0A100", ["NO_HELMET", "ILLEGAL_PARKING"]),
    ("Low (seatbelt / phone)", "#2E7D4F", ["MOBILE_PHONE_WHILE_DRIVING", "NO_SEATBELT"]),
]


def _json(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _greeting():
    h = datetime.datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


def _kpi(value, label, note, tone="grey"):
    return (f'<div class="stat"><b>{ui.esc(value)}</b><span>{ui.esc(label)}</span>'
            f'<div style="margin-top:6px"><span class="tag t-{tone}">{ui.esc(note)}</span></div></div>')


def render():
    officer = st.session_state.get("officer", "Officer")
    ui.page_header("Overview", f"{_greeting()}, {officer}. Here is the city at a glance.")

    ev = _json("data/evaluation_results.json")
    bn = _json("data/benchmark_results.json")
    m = (ev.get("overall") or {}).get("mAP50")
    acc, acc_note = (f"{m:.1%}", "mAP@0.5 on validation") if isinstance(m, (int, float)) else ("N/A", "Not evaluated yet")
    lat = bn.get("mean_latency_ms")
    lat_v, lat_note = (f"{lat:.1f} ms", f"{bn.get('average_fps', 0):.1f} FPS benchmark") if isinstance(lat, (int, float)) \
        else ("N/A", "Not benchmarked yet")

    with session() as db:
        total = db.query(Violation).count()
        today0 = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today = db.query(Violation).filter(Violation.timestamp >= today0).count()
        vehicles = db.query(Vehicle).count()
        plates_seen = db.query(func.count(func.distinct(Violation.plate_number))) \
            .filter(Violation.plate_number.isnot(None), Violation.plate_number != "UNIDENTIFIED").scalar() or 0
        cams = db.query(CameraSource).all()
        online = sum(1 for c in cams if c.status == "ONLINE")
        year0 = datetime.datetime(datetime.datetime.utcnow().year, 1, 1)
        ts_year = [t for (t,) in db.query(Violation.timestamp).filter(Violation.timestamp >= year0).all() if t]
        by_type = db.query(Violation.violation_type, func.count(Violation.id)).group_by(Violation.violation_type) \
            .order_by(func.count(Violation.id).desc()).all()
        by_status = dict(db.query(Violation.status, func.count(Violation.id)).group_by(Violation.status).all())
        by_cam = dict(db.query(Violation.camera_id, func.count(Violation.id)).group_by(Violation.camera_id).all())
        cam_rows = [(c.name, c.latitude, c.longitude, int(by_cam.get(c.id, 0))) for c in cams]

    waiting = int(by_status.get("PENDING_REVIEW", 0))
    st.markdown(ui.readout([
        (f"{total:,}", "Violations recorded", f"{today} today"),
        (f"{waiting:,}", "Waiting for an officer", "Review queue", "amb" if waiting else ""),
        (f"{max(vehicles, plates_seen):,}", "Vehicles identified", "by number plate"),
        (f"{online}/{len(cams)}", "Cameras online", None, "stop" if cams and online < len(cams) / 2 else ""),
        (acc, "Detection accuracy", acc_note),
        (lat_v, "Processing time", lat_note),
    ]), unsafe_allow_html=True)

    left, right = st.columns([7, 5], gap="medium")
    with left:
        with ui.panel():
            # last 30 days, one bar per day (a year view is mostly empty months for a new deployment)
            day0 = datetime.datetime.utcnow().date() - datetime.timedelta(days=29)
            days = [day0 + datetime.timedelta(days=i) for i in range(30)]
            per_day = {d: 0 for d in days}
            for t in ts_year:
                if t.date() in per_day:
                    per_day[t.date()] += 1
            month_total = sum(per_day.values())
            st.markdown(f"### Violations in the last 30 days: {month_total:,}")
            fig = go.Figure(go.Bar(x=days, y=[per_day[d] for d in days], marker_color=SIGN,
                                   hovertemplate="%{x|%d %b}: %{y}<extra></extra>"))
            fig.update_layout(height=280, bargap=0.25, **LAYOUT)
            fig.update_xaxes(tickformat="%d %b", dtick=7 * 86400000)
            fig.update_yaxes(gridcolor="#EEF1F4", rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with ui.panel():
            st.markdown("### Violation categories")
            if by_type:
                labels = [rs.violation_label(t) for t, _ in by_type]
                vals = [n for _, n in by_type]
                colors = (CAT_COLORS * 3)[:len(labels)]
                fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.66, marker=dict(colors=colors),
                                       textinfo="none", sort=False, hovertemplate="%{label}: %{value}<extra></extra>"))
                fig.update_layout(height=190, showlegend=False,
                                  annotations=[dict(text=f"<b>Top</b><br>{ui.esc(labels[0])}", x=0.5, y=0.5,
                                                    showarrow=False, font=dict(size=13, color="#18212B"))],
                                  **{**LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                tot = max(1, sum(vals))
                st.markdown('<table class="data">' + "".join(
                    f'<tr><td><span style="color:{c}">●</span> {ui.esc(l)}</td>'
                    f'<td style="text-align:right" class="small">{100 * v / tot:.0f}%</td>'
                    f'<td style="text-align:right"><b>{v}</b></td></tr>' for l, v, c in zip(labels, vals, colors))
                            + "</table>", unsafe_allow_html=True)
            else:
                ui.empty("No violations yet", "Run a video from Live cameras.")

    b1, b2, b3 = st.columns([5, 4, 3], gap="medium")
    with b1:
        with ui.panel():
            st.markdown("### Junction hotspots")
            pts = [r for r in cam_rows if r[1] and r[2]]
            if pts:
                m_ = pd.DataFrame(pts, columns=["name", "lat", "lon", "n"])
                size = (m_.n / max(1, m_.n.max()) * 30 + 10) if m_.n.max() else [12] * len(m_)
                marker = dict(size=size, color="#C62828", opacity=0.55)
                text = m_.name + " · " + m_.n.astype(str)
                center = dict(lat=m_.lat.mean(), lon=m_.lon.mean())
                base = {k_: v for k_, v in LAYOUT.items() if k_ != "plot_bgcolor"}
                if hasattr(go, "Scattermap"):
                    fig = go.Figure(go.Scattermap(lat=m_.lat, lon=m_.lon, text=text, mode="markers", hoverinfo="text", marker=marker))
                    fig.update_layout(map=dict(style="open-street-map", center=center, zoom=11), height=220, **base)
                else:
                    fig = go.Figure(go.Scattermapbox(lat=m_.lat, lon=m_.lon, text=text, mode="markers", hoverinfo="text", marker=marker))
                    fig.update_layout(mapbox=dict(style="open-street-map", center=center, zoom=11), height=220, **base)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(f'<div class="small">{len(pts)} junctions · circle size = violations (map tiles need internet)</div>',
                            unsafe_allow_html=True)
            else:
                ui.empty("No camera locations", "Add cameras with latitude and longitude in Camera setup.")
    with b2:
        with ui.panel():
            st.markdown("### Review outcomes")
            denom = max(1, total)
            rows = [("Vehicles tracked", max(vehicles, plates_seen), None),
                    ("Approved / notice", by_status.get("APPROVED", 0), "#2E7D4F"),
                    ("Waiting for review", by_status.get("PENDING_REVIEW", 0), "#B07A00"),
                    ("Rejected (model mistakes)", by_status.get("REJECTED", 0), "#5E6975"),
                    ("Disputed by owner", by_status.get("DISPUTED", 0), "#B07A00")]
            st.markdown('<table class="data"><tr><th>Metric</th><th>Count</th><th>Share</th></tr>' + "".join(
                f'<tr><td>{ui.esc(n)}</td><td><b>{v:,}</b></td>'
                f'<td style="color:{c or "#2E7D4F"}">{"—" if c is None else f"{100 * v / denom:.1f}%"}</td></tr>'
                for n, v, c in rows) + "</table>", unsafe_allow_html=True)
    with b3:
        with ui.panel():
            st.markdown("### Severity")
            tmap = dict(by_type)
            st.markdown("".join(
                f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #EDF0F3">'
                f'<span><span style="color:{c}">●</span> {ui.esc(n)}</span><b>{sum(tmap.get(t, 0) for t in types)}</b></div>'
                for n, c, types in SEVERITY), unsafe_allow_html=True)
