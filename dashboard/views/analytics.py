"""Analytics: what happened over a period, and where the model makes mistakes."""

import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.db_models.models import CameraSource, Violation
from backend.services import review_service as rs
from dashboard import ui
from dashboard.data import session

RANGES = {"Today": 1, "7 days": 7, "This month": "month", "All time": None}
SIGN, PINK, AMBER = "#1F4E8C", "#E7A9A9", "#F2CC66"
LAYOUT = dict(paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=10, b=10),
              font=dict(family="Barlow, sans-serif", color="#5E6975", size=12))


def _since(choice):
    now = datetime.datetime.utcnow()
    v = RANGES[choice]
    if v is None:
        return None
    if v == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now - datetime.timedelta(days=v)


def render():
    head, rng = st.columns([3, 2], vertical_alignment="center")
    with rng:
        with st.container(key="chips_range"):
            choice = st.radio("Range", list(RANGES), index=3, horizontal=True, label_visibility="collapsed")
    since = _since(choice)
    with head:
        ui.page_header("Analytics", f"{choice.lower()} · all cameras")

    with session() as db:
        q = db.query(Violation)
        if since:
            q = q.filter(Violation.timestamp >= since)
        rows = q.all()
        cams = db.query(CameraSource).all()
        if not rows:
            with ui.panel():
                ui.empty("No data for this period", "Process a video or connect a camera to see analytics.")
            _evaluation()
            return

        df = pd.DataFrame([{
            "ts": r.timestamp, "type": r.violation_type, "status": r.status, "camera": r.camera_id,
            "light": r.light_condition or "day", "plate": r.plate_number, "edited": bool(r.plate_edited),
            "created": r.created_at, "reviewed": r.reviewed_at,
        } for r in rows])

        approved = int((df.status == "APPROVED").sum())
        rejected = int((df.status == "REJECTED").sum())
        decided = approved + rejected
        rev = df.dropna(subset=["reviewed"])
        avg_review = None
        if len(rev):
            secs = (rev.reviewed - rev.created).dt.total_seconds().clip(lower=0).mean()
            avg_review = f"{int(secs // 3600)} h {int(secs % 3600 // 60)} min" if secs >= 3600 else f"{int(secs // 60)} min {int(secs % 60)} s"

        k = st.columns(4)
        k[0].markdown(ui.stat(f"{len(df):,}", "Violations detected"), unsafe_allow_html=True)
        k[1].markdown(ui.stat(f"{approved:,}", "Approved by officers"), unsafe_allow_html=True)
        k[2].markdown(ui.stat(f"{100 * rejected / decided:.1f}%" if decided else "—",
                              "Rejected by officers (model mistakes)"), unsafe_allow_html=True)
        k[3].markdown(ui.stat(avg_review or "—", "Average time to review"), unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns([1.5, 1], gap="medium")
        with c1:
            with ui.panel():
                st.markdown("### Violations per day")
                df["day"] = df.ts.dt.date
                piv = df.assign(s=df.status.map({"APPROVED": "Approved", "REJECTED": "Rejected",
                                                 "PENDING_REVIEW": "Waiting", "DISPUTED": "Approved"}).fillna("Waiting")) \
                        .groupby(["day", "s"]).size().unstack(fill_value=0)
                fig = go.Figure()
                for name, col in [("Approved", SIGN), ("Rejected", PINK), ("Waiting", AMBER)]:
                    if name in piv:
                        fig.add_bar(x=piv.index, y=piv[name], name=name, marker_color=col)
                fig.update_layout(barmode="stack", height=280, legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                                  **LAYOUT)
                fig.update_yaxes(gridcolor="#EEF1F4")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            with ui.panel():
                st.markdown("### Junction hotspots")
                counts = df.groupby("camera").size()
                pts = [(c.name, c.latitude, c.longitude, int(counts.get(c.id, 0))) for c in cams
                       if c.latitude and c.longitude and counts.get(c.id, 0)]
                if pts:
                    m = pd.DataFrame(pts, columns=["name", "lat", "lon", "n"])
                    marker = dict(size=(m.n / m.n.max() * 34 + 10), color="#C62828", opacity=0.55)
                    text = m.name + " · " + m.n.astype(str)
                    center = dict(lat=m.lat.mean(), lon=m.lon.mean())
                    base = {k: v for k, v in LAYOUT.items() if k != "plot_bgcolor"}
                    if hasattr(go, "Scattermap"):  # plotly >= 5.24
                        fig = go.Figure(go.Scattermap(lat=m.lat, lon=m.lon, text=text, mode="markers",
                                                      hoverinfo="text", marker=marker))
                        fig.update_layout(map=dict(style="carto-positron", center=center, zoom=11), height=280, **base)
                    else:
                        fig = go.Figure(go.Scattermapbox(lat=m.lat, lon=m.lon, text=text, mode="markers",
                                                         hoverinfo="text", marker=marker))
                        fig.update_layout(mapbox=dict(style="carto-positron", center=center, zoom=11), height=280, **base)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('<div class="small">Circle size = violations in this period</div>', unsafe_allow_html=True)
                else:
                    ui.empty("No camera locations", "Violations here came from uploaded videos without a camera location.")

        c3, c4, c5 = st.columns([1.2, 1, 0.8], gap="medium")
        with c3:
            with ui.panel():
                st.markdown("### When violations happen")
                df["wd"] = df.ts.dt.dayofweek
                df["hr"] = df.ts.dt.hour
                grid = df.groupby(["wd", "hr"]).size().unstack(fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0)
                fig = go.Figure(go.Heatmap(z=grid.values, x=list(range(24)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                           colorscale=[[0, "#F3F6FA"], [1, SIGN]], showscale=False,
                                           hovertemplate="%{y} %{x}:00 · %{z}<extra></extra>"))
                fig.update_layout(height=230, **LAYOUT)
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown('<div class="small">Darker = more violations · weekday × hour (UTC)</div>', unsafe_allow_html=True)
        with c4:
            with ui.panel():
                st.markdown("### Where the model is wrong")
                st.markdown('<div class="small">Share of detections officers rejected</div>', unsafe_allow_html=True)
                dec = df[df.status.isin(["APPROVED", "REJECTED"])]
                if len(dec):
                    g = dec.groupby(["type", "light"]).apply(lambda x: 100 * (x.status == "REJECTED").mean(),
                                                             include_groups=False).unstack()
                    fig = go.Figure()
                    for light, col, name in [("day", "#7FA3CF", "Day"), ("night", "#1F3E66", "Night")]:
                        if light in g:
                            fig.add_bar(y=[rs.violation_label(t) for t in g.index], x=g[light], name=name,
                                        orientation="h", marker_color=col,
                                        text=[f"{v:.0f}%" if pd.notna(v) else "" for v in g[light]], textposition="outside")
                    fig.update_layout(barmode="group", height=230, legend=dict(orientation="h", y=-0.15), **LAYOUT)
                    fig.update_xaxes(range=[0, 110], visible=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    ui.empty("Not enough reviews", "This fills in as officers approve and reject detections.")
        with c5:
            with ui.panel():
                st.markdown("### Plates read automatically")
                d = df[df.status.isin(["APPROVED", "REJECTED", "DISPUTED"])]
                if len(d):
                    auto = d.plate.notna() & ~d.edited
                    def pct(mask):
                        sub = d[mask]
                        return f"{100 * (sub.plate.notna() & ~sub.edited).mean():.0f}%" if len(sub) else "—"
                    st.markdown(f'<div class="cond" style="font-size:46px;font-weight:700;line-height:1">{100 * auto.mean():.0f}%</div>'
                                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px">'
                                f'{ui.stat(pct(d.light == "day"), "Day")}{ui.stat(pct(d.light == "night"), "Night")}</div>'
                                '<div class="small" style="margin-top:8px">The rest were typed in by officers or never read.</div>',
                                unsafe_allow_html=True)
                else:
                    ui.empty("No reviewed records", "Shown after officers review detections.")
        # Charts from the first dashboard version: incidents by category, hourly profile
        c6, c7 = st.columns(2, gap="medium")
        with c6:
            with ui.panel():
                st.markdown("### Incidents by category")
                tc = df.type.map(rs.violation_label).value_counts()
                fig = go.Figure(go.Bar(x=tc.index, y=tc.values, marker_color=SIGN, text=tc.values,
                                       textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
                fig.update_layout(height=260, **LAYOUT)
                fig.update_yaxes(gridcolor="#EEF1F4", rangemode="tozero")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c7:
            with ui.panel():
                st.markdown("### Hourly incident profile")
                hc = df.ts.dt.hour.value_counts().reindex(range(24), fill_value=0)
                fig = go.Figure(go.Scatter(x=list(hc.index), y=hc.values, mode="lines+markers",
                                           line=dict(color="#C62828", width=2), hovertemplate="%{x}:00 · %{y}<extra></extra>"))
                fig.update_layout(height=260, **LAYOUT)
                fig.update_xaxes(title="Hour of day (UTC)", dtick=3)
                fig.update_yaxes(gridcolor="#EEF1F4", rangemode="tozero")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    _evaluation()


def _evaluation():
    with ui.panel():
        st.markdown("### Model evaluation")
        p = Path("data/evaluation_results.json")
        if not p.exists():
            st.markdown('<div class="note-amber">Not evaluated yet. After training on Colab, copy '
                        '<b>evaluation_results.json</b> to <b>data/</b> and the per-class precision, recall and mAP '
                        'appear here.</div>', unsafe_allow_html=True)
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not read {p}: {e}")
            return
        def _f1(m):
            p_, r_ = m.get("precision"), m.get("recall")
            return 2 * p_ * r_ / (p_ + r_) if isinstance(p_, (int, float)) and isinstance(r_, (int, float)) and p_ + r_ else None

        overall = data.get("overall") or {}
        if overall:
            tiles = [(overall.get("precision"), "precision"), (overall.get("recall"), "recall"),
                     (_f1(overall), "F1 score"), (overall.get("mAP50"), "mAP@0.50")]
            st.markdown('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">'
                        + "".join(ui.stat(f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "—", lbl)
                                  for v, lbl in tiles) + "</div>", unsafe_allow_html=True)
        per = data.get("per_class") or data.get("classes") or {}
        if isinstance(per, dict) and per:
            rows = [{"Class": str(c).replace("_", " ").capitalize(), "Precision": m.get("precision"),
                     "Recall": m.get("recall"), "F1": _f1(m), "mAP@0.50": m.get("mAP50"),
                     "mAP@0.50:0.95": m.get("mAP50_95")} for c, m in per.items() if isinstance(m, dict)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(format="%.3f")
                                        for k in ("Precision", "Recall", "F1", "mAP@0.50", "mAP@0.50:0.95")})
        st.markdown(f'<div class="small">Helmet / no-helmet / number-plate model ({ui.esc(data.get("model", ""))}), '
                    'validation set of 803 unseen images. Full comparison under System → Accuracy.</div>',
                    unsafe_allow_html=True)
        bench = Path("data/benchmark_results.json")
        if bench.exists():
            b = json.loads(bench.read_text(encoding="utf-8"))
            st.markdown(f'<div class="small">Speed benchmark: {b.get("average_fps")} FPS, '
                        f'{b.get("mean_latency_ms")} ms per frame on {b.get("device", "?")} ({b.get("model")}).</div>',
                        unsafe_allow_html=True)
