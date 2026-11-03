"""Upload proof of violation: a citizen's video or photo goes through detection, then to the review queue."""

import datetime

import streamlit as st
from sqlalchemy import desc

from backend.db_models.models import CameraSource, Violation
from backend.services import citizen_service as cs
from backend.services import review_service as rs
from backend.services import geo_service
from dashboard import ui
from dashboard.data import session

TYPES = ["NO_HELMET", "TRIPLE_RIDING", "RED_LIGHT_JUMP", "WRONG_WAY", "NO_SEATBELT",
         "MOBILE_PHONE_WHILE_DRIVING", "CITIZEN_REPORT"]
TYPE_LABEL = {**{t: rs.violation_label(t) for t in TYPES}, "CITIZEN_REPORT": "Something else / not sure"}


def _local_to_utc(d: datetime.date, t: datetime.time) -> datetime.datetime:
    local = datetime.datetime.combine(d, t).astimezone()  # this computer's time zone (IST on the project PC)
    return local.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def render():
    ui.page_header("Upload proof of violation",
                   "Citizens can send a video or photo of a traffic violation. An officer checks every upload.")
    with session() as db:
        cams = db.query(CameraSource).order_by(CameraSource.location).all()
        cam_opts = {None: "Not near a camera / other place"} | {c.id: f"{c.location} ({c.name})" for c in cams}

    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        with ui.panel():
            with st.form("citizen_upload", clear_on_submit=False):
                st.markdown("### 1. The proof")
                up = st.file_uploader("Video or photo", type=["mp4", "avi", "mkv", "mov", "jpg", "jpeg", "png"],
                                      help="Videos are checked by the AI (first 30 seconds). Photos go straight to an officer.")
                st.markdown("### 2. What happened")
                c1, c2 = st.columns(2)
                vtype = c1.selectbox("Violation you saw", TYPES, format_func=TYPE_LABEL.get)
                plate = c2.text_input("Vehicle number (if you could read it)", placeholder="e.g. KA05MX4821")
                cam_default = st.session_state.get("cit_cam")
                cam_index = list(cam_opts).index(cam_default) if cam_default in cam_opts else 0
                cam = st.selectbox("Nearest junction", list(cam_opts), index=cam_index, format_func=cam_opts.get)
                place = st.text_input("Place / landmark", value=st.session_state.get("cit_place", ""), placeholder="e.g. Silk Board junction, towards Hosur Rd")
                c3, c4 = st.columns(2)
                day = c3.date_input("Date", value=datetime.date.today(), max_value=datetime.date.today())
                tm = c4.time_input("Time (approx.)", value=datetime.datetime.now().time().replace(second=0, microsecond=0))
                note = st.text_area("Anything else the officer should know", max_chars=500, height=80)
                st.markdown("### 3. Your details (optional)")
                c5, c6 = st.columns(2)
                name = c5.text_input("Name")
                phone = c6.text_input("Phone", help="Used only if the officer needs to ask you about this upload.")
                agree = st.checkbox("I recorded this myself and it is a genuine, unedited record of what happened.")
                send = st.form_submit_button("Submit proof", type="primary", use_container_width=True)

        if send:
            if up is None:
                st.error("Please add a video or photo.")
            elif not agree:
                st.error("Please confirm that the upload is genuine.")
            elif not (place.strip() or cam):
                st.error("Please give the place or pick the nearest junction.")
            else:
                try:
                    path = cs.save_upload(up.name, up.getvalue())
                except ValueError as e:
                    st.error(str(e))
                    return
                location = place.strip() or cam_opts[cam].split(" (")[0]
                info = {"violation_type": vtype, "plate": plate, "camera_id": cam, "location": location,
                        "when_utc": _local_to_utc(day, tm), "note": note.strip(), "name": name.strip(),
                        "phone": phone.strip()}
                bar = st.progress(0.0, "Starting…")
                result = cs.process_upload(path, info, progress=lambda f, msg: bar.progress(min(f, 1.0), msg))
                bar.empty()
                st.session_state.citizen_result = result

        res = st.session_state.get("citizen_result")
        if res:
            with ui.panel():
                st.markdown("### Thank you. Your proof was received.")
                rows = "".join(
                    f'<tr><td><code>{ui.esc(r["event_id"])}</code></td><td>{ui.violation_tag(r["type"], "CITIZEN")}</td>'
                    f'<td>{ui.plate_html(r["plate"])}</td><td class="small">'
                    f'{"Found by the AI (confidence " + format(r["confidence"], ".2f") + ")" if r["auto"] else "Your report, for an officer to check"}'
                    f'</td></tr>' for r in res["records"])
                st.markdown(f'<table class="data"><tr><th>Reference</th><th>Violation</th><th>Plate</th><th>How</th></tr>'
                            f'{rows}</table>', unsafe_allow_html=True)
                checked = f"The AI checked {res['frames']} frames and found {res['detected']} violation(s). " if res["frames"] else ""
                st.markdown(f'<div class="note-blue" style="margin-top:8px">{checked}Every record is now waiting in the '
                            'officer review queue. No notice is sent until an officer approves it.</div>',
                            unsafe_allow_html=True)

    with right:
        with ui.panel():
            st.markdown("### Recent citizen uploads")
            with session() as db:
                recent = (db.query(Violation).filter(Violation.source == "CITIZEN")
                          .order_by(desc(Violation.created_at)).limit(12).all())
                if not recent:
                    ui.empty("No uploads yet", "Uploads from citizens appear here with their review status.")
                for v in recent:
                    st.markdown(
                        f'<div style="padding:8px 0;border-bottom:1px solid #EDF0F3">'
                        f'<div style="display:flex;gap:8px;flex-wrap:wrap">{ui.violation_tag(v.violation_type, "CITIZEN")}'
                        f'{ui.status_tag(v.status, v.reject_reason)}</div>'
                        f'<div class="small" style="margin-top:4px">{ui.fmt_time(v.timestamp)} · {ui.esc(v.location)} · '
                        f'{ui.esc(v.plate_number or "plate not given")}'
                        f'{" · by " + ui.esc(v.reporter_name) if v.reporter_name else ""}</div></div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="note-amber">Tips for good proof: keep the phone steady, film the number plate '
                    'clearly, and include a few seconds before and after the violation. Do not take risks to record.</div>',
                    unsafe_allow_html=True)
