"""Violations: every detection and citizen report with its review outcome; search, filter, export."""

import base64
import datetime
import io
import json
import os

import pandas as pd
import streamlit as st
from PIL import Image
from sqlalchemy import desc

from backend.db_models.models import Violation
from backend.services import review_service as rs
from backend.services.challan_service import challan_service
from dashboard import ui
from dashboard.data import camera_names, evidence_for, session, where

PAGE_SIZE = 12
DATES = {"Today": 1, "Last 7 days": 7, "Last 30 days": 30, "All time": None}
STATUSES = [("All", None), ("Waiting", "PENDING_REVIEW"), ("Approved", "APPROVED"),
            ("Rejected", "REJECTED"), ("Disputed", "DISPUTED")]


@st.cache_data(show_spinner=False, max_entries=500)
def _thumb(path: str, mtime: float) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((160, 100))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def _thumb_html(path):
    if not path:
        return '<div style="width:96px;height:56px;background:#EEF1F4;border-radius:4px"></div>'
    b64 = _thumb(path, os.path.getmtime(path))
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:96px;height:56px;object-fit:cover;border-radius:4px">'


def render():
    ui.page_header("Violations", "Every detection and citizen report, with its review outcome")
    with session() as db:
        cams = camera_names(db)
        with ui.panel():
            c = st.columns([1.3, 0.9, 1, 1.1, 1, 1.1])
            plate_q = c[0].text_input("Search plate", placeholder="e.g. KA05MX4821").strip().upper().replace(" ", "")
            date_k = c[1].selectbox("Date", list(DATES), index=3)
            type_opts = ["All types"] + list(rs.VIOLATION_LABELS)
            type_k = c[2].selectbox("Violation", type_opts,
                                    format_func=lambda x: x if x == "All types" else rs.violation_label(x))
            cam_opts = ["All cameras"] + list(cams)
            cam_k = c[3].selectbox("Camera", cam_opts, format_func=lambda x: x if x == "All cameras" else cams[x])
            src_k = c[4].selectbox("Source", ["CCTV and citizen", "CCTV only", "Citizen only"])
            min_conf = c[5].slider("Minimum AI confidence", 0.0, 1.0, 0.0, 0.05)

            q = db.query(Violation)
            if plate_q:
                q = q.filter(Violation.plate_number.ilike(f"%{plate_q}%"))
            if DATES[date_k]:
                q = q.filter(Violation.timestamp >= datetime.datetime.utcnow() - datetime.timedelta(days=DATES[date_k]))
            if type_k != "All types":
                q = q.filter(Violation.violation_type == type_k)
            if cam_k != "All cameras":
                q = q.filter(Violation.camera_id == cam_k)
            if src_k == "CCTV only":
                q = q.filter(Violation.source != "CITIZEN")
            elif src_k == "Citizen only":
                q = q.filter(Violation.source == "CITIZEN")
            if min_conf > 0:
                q = q.filter(Violation.confidence >= min_conf)

            counts = {code: q.filter(Violation.status == code).count() for _, code in STATUSES if code}
            total = q.count()
            labels = [f"{n} {total if code is None else counts[code]}" for n, code in STATUSES]
            with st.container(key="chips_status"):
                pick = st.radio("Status", labels, horizontal=True, label_visibility="collapsed")
            status_code = dict(zip(labels, [c for _, c in STATUSES]))[pick]
            if status_code:
                q = q.filter(Violation.status == status_code)

        rows = q.order_by(desc(Violation.timestamp)).all()
        with ui.panel():
            if not rows:
                ui.empty("No violations match", "Change the filters above, or process a video from Live cameras.")
                return
            pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
            page_no = st.session_state.get("viol_page", 1)
            page_no = min(max(1, page_no), pages)
            chunk = rows[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]
            body = ""
            for v in chunk:
                ev = evidence_for(db, v)
                body += (f"<tr><td>{_thumb_html(ev['image'])}</td><td>{ui.fmt_time(v.timestamp)}</td>"
                         f"<td>{ui.violation_tag(v.violation_type, v.source)}</td><td>{ui.plate_html(v.plate_number)}</td>"
                         f"<td>{ui.esc(where(v, cams))}</td><td class='conf'>{v.confidence:.2f}</td>"
                         f"<td>{ui.status_tag(v.status, v.reject_reason)}"
                         f"{' <span class=small>+' + str(v.merged_count) + ' merged</span>' if v.merged_count else ''}</td>"
                         f"<td class='small'>{ui.esc(v.reviewed_by or '—')}</td></tr>")
            st.markdown('<table class="data"><tr><th>Evidence</th><th>When</th><th>Violation</th><th>Number plate</th>'
                        '<th>Where</th><th>AI confidence</th><th>Status</th><th>Reviewed by</th></tr>'
                        f"{body}</table>", unsafe_allow_html=True)

            f1, f2, f3, f4 = st.columns([4, 1, 1, 1.6], vertical_alignment="center")
            f1.markdown(f'<div class="small">Showing {(page_no - 1) * PAGE_SIZE + 1}–{(page_no - 1) * PAGE_SIZE + len(chunk)} '
                        f'of {len(rows)} · newest first</div>', unsafe_allow_html=True)
            if f2.button("Previous", disabled=page_no <= 1):
                st.session_state.viol_page = page_no - 1
                st.rerun()
            if f3.button("Next", disabled=page_no >= pages):
                st.session_state.viol_page = page_no + 1
                st.rerun()
            export = pd.DataFrame([{
                "event_id": v.event_id, "time_utc": v.timestamp, "violation": rs.violation_label(v.violation_type),
                "plate": v.plate_number or "", "where": where(v, cams), "ai_confidence": round(v.confidence, 3),
                "status": rs.status_label(v.status), "reject_reason": v.reject_reason or "",
                "reviewed_by": v.reviewed_by or "", "merged_duplicates": v.merged_count or 0,
                "source": v.source or "CCTV", "light": v.light_condition or "",
            } for v in rows])
            f4.download_button("Export CSV", export.to_csv(index=False).encode("utf-8"),
                               file_name=f"violations_{datetime.date.today()}.csv", mime="text/csv",
                               use_container_width=True)

        _inspector(db, rows, chunk, cams)


def _act(action, viol_id, reason=None):
    officer = st.session_state.get("officer", "Officer")
    with session() as db:
        try:
            if action == "approve":
                rs.approve(db, viol_id, officer)
                st.toast("Approved. Prototype notice generated.")
            else:
                rs.reject(db, viol_id, officer, reason or "")
                st.toast("Dismissed as a false detection (kept as training data).")
        except ValueError as e:
            st.session_state.insp_error = str(e)


def _inspector(db, rows, chunk, cams):
    """Forensic evidence & incident inspector from the first dashboard version."""
    with ui.panel():
        st.markdown("### Evidence & incident inspector")
        options = [v.id for v in chunk] + [v.id for v in rows if v not in chunk][:150]
        by_id = {v.id: v for v in rows}
        sel = st.selectbox("Record to inspect", options,
                           format_func=lambda i: f"#{i} · {by_id[i].event_id} · {rs.violation_label(by_id[i].violation_type)}"
                                                 f" · {by_id[i].plate_number or 'plate not read'}")
        v = by_id[sel]
        ev = evidence_for(db, v)
        pen = challan_service.get_penalty_info(v.violation_type or "")
        left, right = st.columns([1, 1.15], gap="medium")
        with left:
            st.markdown(
                f'<dl class="kv"><dt>Event ID</dt><dd><code>{ui.esc(v.event_id)}</code></dd>'
                f'<dt>Violation</dt><dd>{ui.violation_tag(v.violation_type, v.source)}</dd>'
                f'<dt>Number plate</dt><dd>{ui.plate_html(v.plate_number)}</dd>'
                f'<dt>Location</dt><dd>{ui.esc(where(v, cams))}</dd>'
                f'<dt>Time</dt><dd>{ui.fmt_time(v.timestamp)} UTC</dd>'
                f'<dt>AI confidence</dt><dd>{v.confidence:.1%}</dd>'
                f'<dt>Status</dt><dd>{ui.status_tag(v.status, v.reject_reason)}</dd>'
                f'<dt>Reviewed by</dt><dd>{ui.esc(v.reviewed_by or "—")}</dd>'
                f'<dt>Legal section</dt><dd>{ui.esc(pen.get("section", "—"))}</dd>'
                f'<dt>Fine schedule</dt><dd><b style="color:var(--stop);font-size:16px">₹{pen.get("fine_inr", 0):,}</b>'
                f' · demerit {ui.esc(pen.get("points", 0))} pts</dd></dl>', unsafe_allow_html=True)
            if st.session_state.get("insp_error"):
                st.error(st.session_state.pop("insp_error"))
            if v.status in ("PENDING_REVIEW", "DISPUTED"):
                b1, b2 = st.columns(2)
                b1.button("Confirm and issue notice", type="primary", use_container_width=True, key=f"insp_ok_{v.id}",
                          on_click=_act, args=("approve", v.id))
                with b2.popover("Dismiss", use_container_width=True):
                    reason = st.selectbox("Reason", rs.REJECT_REASONS, key=f"insp_reason_{v.id}")
                    st.button("Dismiss this detection", key=f"insp_rej_{v.id}", on_click=_act,
                              args=("reject", v.id, reason))
            if v.status == "APPROVED":
                doc = challan_service.generate_challan(
                    violation_id=v.id, event_id=v.event_id, violation_type=v.violation_type,
                    plate_number=v.plate_number, location=where(v, cams),
                    timestamp=v.timestamp or datetime.datetime.utcnow(), confidence=v.confidence,
                    evidence_image_path=ev["image"])
                doc["approved_by"] = v.reviewed_by
                st.markdown(
                    f'<div class="note-amber" style="margin-top:10px"><div style="display:flex;justify-content:space-between">'
                    f'<b style="text-transform:uppercase;font-size:12px">Prototype violation notice</b>'
                    f'<span class="tag t-go">Generated</span></div>'
                    f'<div style="font-size:15px;font-weight:700;margin-top:4px">Notice no: {ui.esc(doc["challan_number"])}</div>'
                    f'<div class="small">{ui.esc(doc["offense_description"])}</div>'
                    f'<div style="color:var(--stop);font-weight:700;margin-top:4px">Fine: ₹{doc["fine_amount_inr"]:,} · '
                    f'due {ui.esc(doc["due_date"])}</div>'
                    f'<div class="small">Approved by {ui.esc(v.reviewed_by or "—")} · ref {ui.esc(doc["verification_hash"])}</div>'
                    f'</div>', unsafe_allow_html=True)
                st.download_button("Download prototype notice (JSON)", json.dumps(doc, indent=2),
                                   file_name=f"{doc['challan_number']}.json", mime="application/json",
                                   key=f"insp_dl_{v.id}")
        with right:
            if ev["image"]:
                st.image(ev["image"], caption=f"Snapshot · {v.event_id}", use_container_width=True)
            else:
                ui.empty("Snapshot missing", "The evidence image for this record was not found on disk.")
            if ev["video"]:
                st.video(ev["video"], key=f"vio_video_{v.id}")
