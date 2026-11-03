"""Vehicles: the full record of one number plate."""

from collections import Counter

import streamlit as st
from sqlalchemy import desc

from backend.db_models.models import Violation
from backend.services import review_service as rs
from dashboard import ui
from dashboard.data import camera_names, evidence_for, session, where


def render():
    head, search = st.columns([3, 1.4], vertical_alignment="center")
    with head:
        ui.page_header("Vehicles", "Look up the full record of one number plate")
    with search:
        plate = st.text_input("Number plate", placeholder="e.g. KA05MX4821", label_visibility="collapsed",
                              key="veh_q").strip().upper().replace(" ", "")

    with session() as db:
        cams = camera_names(db)
        if not plate:
            recent = (db.query(Violation.plate_number).filter(Violation.plate_number.isnot(None))
                      .order_by(desc(Violation.timestamp)).limit(40).all())
            seen = list(dict.fromkeys(p for (p,) in recent if p and p != "UNIDENTIFIED"))[:12]
            with ui.panel():
                if seen:
                    st.markdown("**Recently seen plates**")
                    cols = st.columns(4)
                    for i, p in enumerate(seen):
                        cols[i % 4].button(p, key=f"veh_{p}", use_container_width=True,
                                           on_click=lambda p=p: st.session_state.update(veh_q=p))
                else:
                    ui.empty("No plates read yet", "Plates appear here once the system reads them from video.")
            return

        rows = db.query(Violation).filter(Violation.plate_number == plate).order_by(desc(Violation.timestamp)).all()
        if not rows:
            with ui.panel():
                ui.empty(f"No records for {plate}", "Check the plate, or it has not been seen by any camera yet.")
            return

        left, right = st.columns([1.05, 2.2], gap="medium")
        status = Counter(r.status for r in rows)
        with left:
            with ui.panel():
                first, last = rows[-1].timestamp, rows[0].timestamp
                st.markdown(
                    f'{ui.plate_html(plate, large=True)}'
                    f'<dl class="kv" style="margin-top:12px"><dt>First seen</dt><dd>{ui.fmt_time(first)}</dd>'
                    f'<dt>Last seen</dt><dd>{ui.fmt_time(last)}</dd></dl>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0">'
                    f'{ui.stat(len(rows), "Records in total")}{ui.stat(status.get("APPROVED", 0), "Approved notices")}'
                    f'{ui.stat(status.get("REJECTED", 0), "Rejected by officers")}{ui.stat(status.get("DISPUTED", 0), "Disputed by owner")}</div>',
                    unsafe_allow_html=True)
                st.markdown("**Seen at**")
                st.markdown(ui.bars(Counter(where(r, cams) for r in rows).most_common(6)), unsafe_allow_html=True)
                st.markdown('<div class="note-blue" style="margin-top:12px">Owner name, address and phone are not stored '
                            "in TrafficAI. In a real deployment they stay in the transport department's VAHAN system.</div>",
                            unsafe_allow_html=True)
        with right:
            with ui.panel():
                st.markdown("### History, newest first")
                for r in rows[:15]:
                    ev = evidence_for(db, r)
                    a, b = st.columns([1, 3], vertical_alignment="center")
                    with a:
                        if ev["image"]:
                            st.image(ev["image"], use_container_width=True)
                    with b:
                        outcome = ""
                        if r.status == "APPROVED" and r.reviewed_by:
                            outcome = f"Approved by {r.reviewed_by}"
                        elif r.status == "REJECTED":
                            outcome = f"Rejected by {r.reviewed_by or 'an officer'}: {r.reject_reason or 'no reason saved'}"
                        elif r.status == "PENDING_REVIEW":
                            outcome = f"AI confidence {r.confidence:.2f} · waiting for an officer"
                        elif r.status == "DISPUTED":
                            outcome = "The owner disputed this notice. An officer needs to re-check the clip."
                        st.markdown(
                            f'<div style="display:flex;gap:8px">{ui.violation_tag(r.violation_type, r.source)}'
                            f'{ui.status_tag(r.status)}</div>'
                            f'<div class="small" style="margin-top:6px">{ui.fmt_time(r.timestamp)} · {ui.esc(where(r, cams))}'
                            f' · {ui.esc(r.event_id)} · AI confidence {r.confidence:.0%}</div>'
                            f'<div style="margin-top:4px">{ui.esc(outcome)}</div>', unsafe_allow_html=True)
                    st.markdown('<hr style="margin:6px 0;border:0;border-top:1px solid #EDF0F3">', unsafe_allow_html=True)
