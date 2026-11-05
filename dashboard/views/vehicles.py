"""Vehicles: every number plate the system has read, built live from the violations table.

The register lists each plate with its record counts, fines, risk level and where it was last seen;
search, filters and sorting narrow it down, and picking a row opens that vehicle's full history.
Owner details are not stored here (in a real deployment they come from the transport department's VAHAN).
"""

import datetime
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
from sqlalchemy import func

from backend.db_models.models import Vehicle, Violation
from backend.services.challan_service import ChallanService
from backend.services.review_service import violation_label
from dashboard import ui
from dashboard.data import camera_names, evidence_for, session, where

TWO_WHEELER = {"NO_HELMET", "TRIPLE_RIDING"}
FOUR_WHEELER = {"NO_SEATBELT"}
SHOW = ["All vehicles", "Repeat offenders", "Waiting for review", "Disputed", "Fine due"]
SORT = ["Most records", "Last seen", "Highest fine", "Plate A-Z"]
KINDS = {"motorcycle": "Two-wheeler", "motorbike": "Two-wheeler", "scooter": "Two-wheeler", "two-wheeler": "Two-wheeler",
         "car": "Car", "bus": "Bus", "truck": "Truck", "auto": "Auto", "autorickshaw": "Auto", "—": "—"}
PERIODS = {"All time": None, "Last 7 days": 7, "Last 30 days": 30, "Today": 1}


def _fine(vtype: str) -> int:
    return int(ChallanService.get_penalty_info(vtype or "").get("fine_inr", 0))


def _points(vtype: str) -> int:
    return int(ChallanService.get_penalty_info(vtype or "").get("points", 0))


def _risk(approved: int, records: int, points: int) -> str:
    if approved >= 3 or points >= 8:
        return "High"
    if records >= 2 or points >= 4:
        return "Medium"
    return "Low"


@st.cache_data(ttl=20, show_spinner=False)
def _register(days):
    """One row per plate, aggregated in the database (grouped by plate, type and status)."""
    with session() as db:
        q = db.query(Violation.plate_number, Violation.violation_type, Violation.status,
                     func.count(Violation.id), func.min(Violation.timestamp), func.max(Violation.timestamp))
        q = q.filter(Violation.plate_number.isnot(None), Violation.plate_number != "UNIDENTIFIED",
                     Violation.plate_number != "")
        unread_q = db.query(func.count(Violation.id)).filter(
            (Violation.plate_number.is_(None)) | (Violation.plate_number == "UNIDENTIFIED") | (Violation.plate_number == ""))
        if days:
            since = datetime.datetime.utcnow() - datetime.timedelta(days=days)
            q = q.filter(Violation.timestamp >= since)
            unread_q = unread_q.filter(Violation.timestamp >= since)
        groups = q.group_by(Violation.plate_number, Violation.violation_type, Violation.status).all()
        unread = unread_q.scalar() or 0

        cams = camera_names(db)
        # where each plate was seen last: newest record per plate
        last_rows = (db.query(Violation).filter(Violation.plate_number.in_({g[0] for g in groups}))
                     .order_by(Violation.timestamp.desc()).all()) if groups else []
        last_place = {}
        for r in last_rows:
            last_place.setdefault(r.plate_number, where(r, cams))
        vtypes = dict(db.query(Vehicle.plate_number, Vehicle.vehicle_type).all())

    acc = defaultdict(lambda: {"records": 0, "APPROVED": 0, "PENDING_REVIEW": 0, "REJECTED": 0, "DISPUTED": 0,
                               "fine": 0, "points": 0, "types": Counter(), "first": None, "last": None})
    for plate, vtype, status, n, first, last in groups:
        a = acc[plate]
        a["records"] += n
        a[status] = a.get(status, 0) + n
        if status != "REJECTED":
            a["types"][vtype] += n
        if status == "APPROVED":
            a["fine"] += _fine(vtype) * n
            a["points"] += _points(vtype) * n
        a["first"] = first if a["first"] is None or (first and first < a["first"]) else a["first"]
        a["last"] = last if a["last"] is None or (last and last > a["last"]) else a["last"]

    rows = []
    for plate, a in acc.items():
        types = set(a["types"])
        kind = vtypes.get(plate) or ("Two-wheeler" if types & TWO_WHEELER else "Car" if types & FOUR_WHEELER else "—")
        main = a["types"].most_common(1)[0][0] if a["types"] else None
        rows.append({
            "Plate": plate,
            "Vehicle": KINDS.get(str(kind).lower(), str(kind).capitalize()),
            "Records": a["records"],
            "Approved": a["APPROVED"],
            "Waiting": a["PENDING_REVIEW"],
            "Disputed": a["DISPUTED"],
            "Rejected": a["REJECTED"],
            "Main offence": violation_label(main) if main else "—",
            "Fine (₹)": a["fine"],
            "Points": a["points"],
            "Risk": _risk(a["APPROVED"], a["records"] - a["REJECTED"], a["points"]),
            "Last seen": a["last"],
            "Last place": last_place.get(plate, "—"),
        })
    return pd.DataFrame(rows), unread


def _select(plate: str):
    st.session_state.veh_sel = plate


def render():
    ui.page_header("Vehicles", "Every number plate the system has read: repeat offenders, fines and full history")

    period = st.session_state.get("veh_period", "All time")
    df, unread = _register(PERIODS.get(period))

    total = len(df)
    repeat = int((df["Records"] - df["Rejected"] >= 2).sum()) if total else 0
    waiting = int((df["Waiting"] > 0).sum()) if total else 0
    fines = int(df["Fine (₹)"].sum()) if total else 0
    high = int((df["Risk"] == "High").sum()) if total else 0
    st.markdown('<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:4px 0 12px">'
                f'{ui.stat(total, "Vehicles identified")}{ui.stat(repeat, "Repeat offenders", amber=repeat > 0)}'
                f'{ui.stat(high, "High-risk vehicles", amber=high > 0)}{ui.stat(waiting, "With cases waiting")}'
                f'{ui.stat(f"₹{fines:,}", "Fines from approved notices")}</div>', unsafe_allow_html=True)

    with ui.panel():
        c1, c2, c3, c4, c5 = st.columns([2, 1.3, 1.6, 1.2, 1.2], vertical_alignment="bottom")
        q = c1.text_input("Search plate", placeholder="Type any part of a plate, e.g. KA05 or 4821",
                          key="veh_q").strip().upper().replace(" ", "")
        show = c2.selectbox("Show", SHOW, key="veh_show")
        offences = sorted(df["Main offence"].unique().tolist()) if total else []
        offence = c3.selectbox("Main offence", ["Any offence"] + offences, key="veh_off")
        c4.selectbox("Period", list(PERIODS), key="veh_period")
        sort = c5.selectbox("Sort by", SORT, key="veh_sort")

        view = df
        if total:
            if q:
                view = view[view["Plate"].str.replace(" ", "").str.contains(q, regex=False)]
            if show == "Repeat offenders":
                view = view[view["Records"] - view["Rejected"] >= 2]
            elif show == "Waiting for review":
                view = view[view["Waiting"] > 0]
            elif show == "Disputed":
                view = view[view["Disputed"] > 0]
            elif show == "Fine due":
                view = view[view["Fine (₹)"] > 0]
            if offence != "Any offence":
                view = view[view["Main offence"] == offence]
            by = {"Most records": (["Records", "Last seen"], False), "Last seen": (["Last seen"], False),
                  "Highest fine": (["Fine (₹)", "Records"], False), "Plate A-Z": (["Plate"], True)}[sort]
            view = view.sort_values(by[0], ascending=by[1], na_position="last").reset_index(drop=True)

        if not total:
            ui.empty("No plates read yet", "Vehicles appear here once the system reads a number plate from a camera "
                                           "or a citizen report.")
            return
        st.markdown(f'<div class="small" style="margin:2px 0 6px">{len(view)} of {total} vehicles · '
                    f'{unread} detections without a readable plate are not listed · select a row to open its record'
                    '</div>', unsafe_allow_html=True)
        event = st.dataframe(
            view, hide_index=True, use_container_width=True, height=min(560, 38 + 35 * max(1, len(view))),
            on_select="rerun", selection_mode="single-row", key="veh_table",
            column_order=["Plate", "Vehicle", "Records", "Approved", "Waiting", "Disputed", "Main offence",
                          "Fine (₹)", "Points", "Risk", "Last seen", "Last place"],
            column_config={
                "Records": st.column_config.NumberColumn(help="All detections for this plate"),
                "Approved": st.column_config.NumberColumn(help="Notices approved by an officer"),
                "Waiting": st.column_config.NumberColumn(help="Cases waiting in the review queue"),
                "Fine (₹)": st.column_config.NumberColumn(format="₹%d", help="Sum of fines for approved notices"),
                "Points": st.column_config.NumberColumn(help="Penalty points from approved notices"),
                "Last seen": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
            })

    picked = None
    sel = getattr(getattr(event, "selection", None), "rows", None) or []
    if sel and sel[0] < len(view):
        picked = view.iloc[sel[0]]["Plate"]
    elif len(view) == 1:
        picked = view.iloc[0]["Plate"]
    elif q and q in set(df["Plate"]):
        picked = q
    if picked:
        _detail(picked, df[df["Plate"] == picked].iloc[0])


def _detail(plate: str, summary):
    with session() as db:
        cams = camera_names(db)
        rows = db.query(Violation).filter(Violation.plate_number == plate).order_by(Violation.timestamp.desc()).all()
        if not rows:
            return
        left, right = st.columns([1.05, 2.2], gap="medium")
        status = Counter(r.status for r in rows)
        risk = summary["Risk"]
        risk_cls = {"High": "t-stop", "Medium": "t-amber"}.get(risk, "t-go")
        with left:
            with ui.panel():
                first, last = rows[-1].timestamp, rows[0].timestamp
                st.markdown(
                    f'{ui.plate_html(plate, large=True)}'
                    f'<div style="margin-top:10px"><span class="tag {risk_cls}">{risk} risk</span> '
                    f'<span class="tag t-grey">{ui.esc(summary["Vehicle"])}</span></div>'
                    f'<dl class="kv" style="margin-top:12px"><dt>First seen</dt><dd>{ui.fmt_time(first)}</dd>'
                    f'<dt>Last seen</dt><dd>{ui.fmt_time(last)}</dd>'
                    f'<dt>Last place</dt><dd>{ui.esc(where(rows[0], cams))}</dd>'
                    f'<dt>Fine (approved)</dt><dd>₹{int(summary["Fine (₹)"]):,}</dd>'
                    f'<dt>Penalty points</dt><dd>{int(summary["Points"])}</dd></dl>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0">'
                    f'{ui.stat(len(rows), "Records in total")}{ui.stat(status.get("APPROVED", 0), "Approved notices")}'
                    f'{ui.stat(status.get("PENDING_REVIEW", 0), "Waiting for review", amber=status.get("PENDING_REVIEW", 0) > 0)}'
                    f'{ui.stat(status.get("REJECTED", 0), "Rejected by officers")}</div>',
                    unsafe_allow_html=True)
                if status.get("PENDING_REVIEW"):
                    st.button("Open review queue", key="veh_to_review", use_container_width=True,
                              on_click=lambda: st.session_state.update(nav="Review queue"))
                st.markdown("**Offences**")
                st.markdown(ui.bars(Counter(violation_label(r.violation_type) for r in rows
                                            if r.status != "REJECTED").most_common(6)), unsafe_allow_html=True)
                st.markdown("**Seen at**")
                st.markdown(ui.bars(Counter(where(r, cams) for r in rows).most_common(6)), unsafe_allow_html=True)
                st.markdown('<div class="note-blue" style="margin-top:12px">Owner name, address and phone are not stored '
                            "in TrafficAI. In a real deployment they come from the transport department's VAHAN system.</div>",
                            unsafe_allow_html=True)
        with right:
            with ui.panel():
                st.markdown(f"### History, newest first ({len(rows)})")
                for r in rows[:25]:
                    ev = evidence_for(db, r)
                    a, b = st.columns([1, 3], vertical_alignment="center")
                    with a:
                        if ev["image"]:
                            st.image(ev["image"], use_container_width=True)
                    with b:
                        outcome = ""
                        if r.status == "APPROVED":
                            outcome = f"Approved by {r.reviewed_by or 'an officer'} · fine ₹{_fine(r.violation_type):,}"
                        elif r.status == "REJECTED":
                            outcome = f"Rejected by {r.reviewed_by or 'an officer'}: {r.reject_reason or 'no reason saved'}"
                        elif r.status == "PENDING_REVIEW":
                            outcome = "Waiting for an officer"
                        elif r.status == "DISPUTED":
                            outcome = "The owner disputed this notice. An officer needs to re-check the evidence."
                        st.markdown(
                            f'<div style="display:flex;gap:8px">{ui.violation_tag(r.violation_type, r.source)}'
                            f'{ui.status_tag(r.status)}</div>'
                            f'<div class="small" style="margin-top:6px">{ui.fmt_time(r.timestamp)} · {ui.esc(where(r, cams))}'
                            f' · {ui.esc(r.event_id)} · AI confidence {r.confidence:.0%}</div>'
                            f'<div style="margin-top:4px">{ui.esc(outcome)}</div>', unsafe_allow_html=True)
                    st.markdown('<hr style="margin:6px 0;border:0;border-top:1px solid #EDF0F3">', unsafe_allow_html=True)
                if len(rows) > 25:
                    st.caption(f"Showing the newest 25 of {len(rows)} records. The Violations tab lists them all.")
