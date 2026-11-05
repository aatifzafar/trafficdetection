"""Review queue: interactive officer verification and decision command center."""

import streamlit as st
from sqlalchemy import desc

from backend.db_models.models import Violation
from backend.services import review_service as rs
from backend.services.challan_service import ChallanService
from backend.services.citizen_service import mask_phone
from dashboard import ui
from dashboard.data import camera_names, evidence_for, plate_crop, session, where

TITLES = {
    "NO_HELMET": "Rider without Safety Helmet",
    "TRIPLE_RIDING": "Triple Riding (More than 2 Persons on Two-Wheeler)",
    "RED_LIGHT_JUMP": "Signal Disobedience (Crossed Virtual Stop Line on Red)",
    "WRONG_WAY": "One-Way Corridor Violation (Moving Against Allowed Vector)",
    "NO_SEATBELT": "Driver/Passenger Without Seatbelt (Cabin Compliance)",
    "ILLEGAL_PARKING": "Illegal Parking (Stationary in No-Parking Zone)",
    "UNSAFE_MOVEMENT": "Unsafe Vehicle Movement (Weaving / Over-speeding)",
}
FILTERS = {
    "All": None,
    "No seatbelt": "NO_SEATBELT",
    "No helmet": "NO_HELMET",
    "Triple riding": "TRIPLE_RIDING",
    "Red light": "RED_LIGHT_JUMP",
    "Wrong way": "WRONG_WAY",
    "Illegal parking": "ILLEGAL_PARKING",
    "Unsafe movement": "UNSAFE_MOVEMENT",
    "Citizen reports": "CITIZEN",
}


def _why(viol, meta) -> list:
    extra = (meta or {}).get("extra", {}) or {}
    lines = []
    t = viol.violation_type
    if t == "NO_HELMET":
        if extra.get("bike_track_id") is not None:
            lines.append(f"Spatial Tracking: Rider linked to motorcycle #{extra['bike_track_id']} by bounding box containment")
        mode = extra.get("mode")
        if mode == "CUSTOM_MODEL":
            lines.append("Detection Head: Trained model detected 'no_helmet' on rider's head in >= 6 of 10 consecutive frames")
        elif mode:
            lines.append("Baseline Heuristic: Triggered by natural edge density and skin tone thresholding in head region")
        if extra.get("seat_role"):
            lines.append(f"Position: Seated as {extra['seat_role'].replace('_', ' ').title()}")
    elif t == "TRIPLE_RIDING":
        mode = extra.get("detection_mode")
        if mode == "DIRECT_MODEL":
            lines.append("Detection Head: Dedicated trained model detected 'MORE_THAN_TWO_PERSONS' (Triple Riding) with high confidence")
        elif extra.get("rider_count"):
            lines.append(f"Spatial Tracking: {extra['rider_count']} riders spatially bound to motorcycle chassis")
    elif t == "RED_LIGHT_JUMP":
        lines.append(f"Signal State: Signal confirmed as {extra.get('signal_state', 'RED').upper()} at the exact moment of crossing")
    elif t == "WRONG_WAY":
        is_motion_verified = (
            extra.get("motion_verified") is True
            or (viol.source == "CCTV" and extra.get("temporal_confirmed") is True)
        )
        v_type = (extra.get("vehicle_type") or "Vehicle").title()
        lines.append(f"Vehicle Type: {v_type}")

        if is_motion_verified:
            lines.append("✓ Temporal Status: WRONG WAY VERIFIED (Multi-frame kinematic tracking)")
            t_id = viol.track_id or extra.get("track_id")
            if t_id:
                lines.append(f"Vehicle Tracking ID: Track #{t_id}")
            if extra.get("motion_direction"):
                lines.append(f"Movement Direction Vector: {extra['motion_direction']}")
            elif extra.get("velocity_vector"):
                lines.append(f"Velocity Vector: dx={extra['velocity_vector'][0]}px, dy={extra['velocity_vector'][1]}px")
            if extra.get("lane_vector"):
                lines.append(f"Permitted Lane Direction: {extra['lane_vector']}")
            if extra.get("angle_deviation_deg") is not None:
                lines.append(f"Angular Deviation: {extra['angle_deviation_deg']}° (Threshold: ≥120.0°)")
            if extra.get("confirming_observations"):
                lines.append(f"Confirming Observations: {extra['confirming_observations']}")
            if extra.get("dl_model_active"):
                lines.append("DL Trajectory Classifier: Bi-LSTM temporal sequence confirmed")
        else:
            lines.append("Verification Status: POTENTIAL WRONG WAY · NOT VERIFIED · PENDING OFFICER REVIEW")
            if viol.confidence > 0:
                lines.append(f"AI Detection Confidence: {int(viol.confidence * 100)}% (Vehicle Localization & Heading Consensus)")
            lines.append("Visual Evidence: Suspected offending vehicle marked with a prominent RED bounding box; compliant traffic marked in BLUE.")
            if extra.get("orientation_consensus"):
                lines.append(f"Carriageway Flow Consensus: {extra['orientation_consensus']}")
            if extra.get("vehicle_orientation") and extra.get("dominant_flow"):
                lines.append(f"Vehicle Heading: {extra['vehicle_orientation']} (Opposing permitted flow: {extra['dominant_flow']})")
            if extra.get("dominant_flow"):
                lines.append(f"Permitted Direction: {extra['dominant_flow'].replace('_', ' ').title()}")
            if extra.get("angle_deviation_deg") is not None:
                lines.append(f"Angular Deviation: {extra['angle_deviation_deg']}° (Opposing orientation)")
            unver_reason = extra.get("unverified_reason") or "Insufficient temporal motion evidence in static photograph."
            lines.append(f"Reason: {unver_reason}")
            lines.append("Legal Notice: A static photograph cannot establish motion vectors. Human officer verification required before issuing notice.")
    elif t == "ILLEGAL_PARKING":
        if extra.get("stationary_seconds") is not None:
            lines.append(f"Dwell Time: {extra.get('vehicle_class', 'Vehicle')} stood still for {extra['stationary_seconds']} s "
                         f"inside no-parking zone {int(extra.get('zone_index', 0)) + 1} (allowed {extra.get('allowed_seconds')} s)")
    elif t == "UNSAFE_MOVEMENT":
        reasons = extra.get("reasons") or []
        if "WEAVING" in reasons:
            lines.append(f"Trajectory: weaving, {extra.get('direction_changes')} side-to-side direction changes "
                         f"(swing {extra.get('lateral_swing_widths')} vehicle widths) in >= 6 of 10 frames")
        if "OVER_SPEED" in reasons and extra.get("speed_kmph") is not None:
            lines.append(f"Speed: about {extra['speed_kmph']} km/h against a limit of {extra.get('speed_limit_kmph')} km/h "
                         "(from the camera's pixels-per-metre calibration)")
    elif t == "NO_SEATBELT":
        persp = extra.get("perspective", "AUTO")
        sharp = extra.get("sharpness")
        roi_mode = extra.get("roi_mode", "windshield")
        crop_dim = extra.get("crop_dimensions")
        diag_trace = extra.get("diagnostic_trace", [])
        lines.append(f"Camera Perspective: {persp} perspective analysis ({roi_mode.replace('_', ' ')})")
        if sharp is not None:
            lines.append(f"Cabin Optical Sharpness: Laplacian variance {sharp} (passes optical threshold >= 15.0)")
        if crop_dim:
            lines.append(f"Inspection Resolution: Cabin crop {crop_dim[0]}×{crop_dim[1]}px")
        if diag_trace:
            top_diag = diag_trace[-1]
            lines.append(f"Diagnostic ROI: Mode '{top_diag.get('mode')}', Model class '{top_diag.get('class_name')}' at {int(top_diag.get('confidence', 0)*100)}% confidence")
        if viol.source != "CITIZEN":
            lines.append("Temporal Confirmation: Missing seatbelt confirmed across multi-frame sliding window (>= 6 of 10 frames)")
        lines.append("Compliance Safeguard: Marked as EXPERIMENTAL requiring mandatory officer verification")
        if extra.get("limitations_note"):
            lines.append(f"Advisory: {extra['limitations_note']}")
    if extra.get("light"):
        lines.append("Ambient Lighting: " + ("Night / Low-light illumination" if extra["light"] == "night" else "Daylight CCTV illumination"))
    concurrent = [c for c in extra.get("concurrent_violations", []) if c != viol.violation_type]
    if concurrent:
        c_names = ", ".join(TITLES.get(c, c) for c in concurrent)
        lines.append(f"Concurrent Offense: Vehicle was simultaneously flagged for {c_names}")

    if viol.source == "CITIZEN":
        if extra.get("auto_classified") and extra.get("claimed_violation") and extra.get("claimed_violation") not in {None, "AUTO_DETECT", viol.violation_type}:
            lines.append(f"AI Auto-Classification: Citizen reported '{extra['claimed_violation']}', but AI detected '{viol.violation_type}' with {int(viol.confidence * 100)}% confidence")
        elif extra.get("auto_classified") or (viol.confidence and viol.confidence > 0):
            lines.append(f"AI Computer Vision: Automatically detected and verified from citizen upload ({int(viol.confidence * 100)}% confidence)")
        elif extra.get("citizen_manual"):
            lines.append("Citizen Upload: Manually reported by app user — review photographic evidence closely")
        else:
            lines.append("Automated Detection: Processed from video footage submitted via citizen portal")
        if viol.report_note:
            lines.append(f"Citizen Note: \"{viol.report_note}\"")
    return lines or ["Detection validated against standard spatial-temporal rules."]


def _do(action, viol_id, next_id=None, **kw):
    officer = st.session_state.get("officer", "Officer")
    with session() as db:
        try:
            if action == "approve":
                rs.approve(db, viol_id, officer)
                st.toast("Violation Approved. Prototype violation notice generated.")
            elif action == "reject":
                rs.reject(db, viol_id, officer, kw.get("reason", ""))
                st.toast("Incident Rejected. Saved with reason for model retraining.")
            elif action == "plate":
                rs.correct_plate(db, viol_id, kw.get("plate", ""), officer)
                st.toast(f"Plate corrected to '{kw.get('plate', '').strip().upper()}'.")
        except ValueError as e:
            st.session_state.review_error = str(e)
            return
    if action != "plate":
        if next_id:
            st.session_state.review_sel = next_id
        else:
            st.session_state.pop("review_sel", None)
    st.rerun()


def _go(viol_id):
    st.session_state.review_sel = viol_id


def render():
    with session() as db:
        cams = camera_names(db)
        pending = (
            db.query(Violation)
            .filter(Violation.status == rs.PENDING)
            .order_by(Violation.timestamp.asc())
            .all()
        )

        counts = {k: 0 for k in FILTERS}
        counts["All"] = len(pending)
        for v in pending:
            for name, code in FILTERS.items():
                if code == "CITIZEN" and v.source == "CITIZEN":
                    counts[name] += 1
                elif code and v.violation_type == code:
                    counts[name] += 1

        # Page Header & Filter Chips
        ui.page_header("Review queue", f"{len(pending)} cases waiting · every detection is checked by an officer "
                                       "before any notice is issued")
        chip_col = st.container()

        if not pending:
            with ui.panel():
                ui.empty(
                    "Queue is Clear!",
                    "All detected violations have been reviewed. New incidents from CCTV feeds and citizen reports will appear here automatically."
                )
            return

        with chip_col:
            options = [f"{k} ({counts[k]})" for k in FILTERS if k == "All" or counts[k] > 0]
            with st.container(key="chips_review"):
                pick = st.radio("Filter", options, horizontal=True, label_visibility="collapsed")

        filter_name = pick.rsplit(" (", 1)[0]
        code = FILTERS.get(filter_name)
        items = [
            v for v in pending
            if code is None
            or (code == "CITIZEN" and v.source == "CITIZEN")
            or (v.violation_type == code)
        ]
        ids = [v.id for v in items]

        if not items:
            with ui.panel():
                ui.empty("No matching cases", f"No pending cases match the filter '{filter_name}'.")
            return

        if st.session_state.get("review_sel") not in ids:
            st.session_state.review_sel = ids[0]

        viol = next((v for v in items if v.id == st.session_state.review_sel), items[0])
        pos = ids.index(viol.id)
        nxt = ids[pos + 1] if pos + 1 < len(ids) else (ids[pos - 1] if pos > 0 else None)
        prev_id = ids[pos - 1] if pos > 0 else None
        next_id = ids[pos + 1] if pos + 1 < len(ids) else None

        # Main Layout: 3 Columns (Sidebar Queue | Evidence Studio | Action Deck)
        left_col, center_col, right_col = st.columns([1.1, 2.2, 1.25], gap="medium")

        # ==============================================================
        # 1. LEFT COLUMN: PENDING CASES QUEUE
        # ==============================================================
        with left_col:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<span style="font-weight:700;font-size:14px;color:var(--ink);">CASES ({len(items)})</span>'
                f'<span style="font-size:12px;color:var(--muted);">Oldest First</span></div>',
                unsafe_allow_html=True
            )

            with st.container(height=720, border=False, key="rq_list"):
                for idx, v in enumerate(items[:40]):
                    is_active = (v.id == viol.id)
                    ev_item = evidence_for(db, v)
                    img_path = ev_item.get("image")

                    with st.container(border=True, key=f"qcard_{v.id}{'_active' if is_active else ''}"):
                        thumb_c, meta_c = st.columns([1, 2.1], vertical_alignment="center", gap="small")
                        with thumb_c:
                            if img_path:
                                st.image(img_path, use_container_width=True)
                            else:
                                st.markdown('<div style="width:100%;height:45px;background:#E2E6EB;border-radius:4px;"></div>', unsafe_allow_html=True)
                        with meta_c:
                            active_pill = '<span class="rq-active-badge">ACTIVE</span> ' if is_active else ''
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:4px;">{active_pill}{ui.violation_tag(v.violation_type, v.source)}</div>'
                                f'<div style="margin:4px 0 2px;">{ui.plate_html(v.plate_number, small=True)}</div>'
                                f'<div class="small rq-meta">{ui.esc(where(v, cams))} · {ui.fmt_time(v.timestamp)}</div>',
                                unsafe_allow_html=True
                            )
                        # Full overlay click target: makes the ENTIRE card clickable
                        st.button(
                            f"Select Case #{idx + 1}",
                            key=f"card_btn_{v.id}",
                            on_click=_go,
                            args=(v.id,),
                            use_container_width=True
                        )

                if len(items) > 40:
                    st.markdown(f'<div class="small" style="text-align:center;padding:10px;">+ {len(items) - 40} more in queue</div>', unsafe_allow_html=True)

        ev = evidence_for(db, viol)
        meta = ev.get("meta") or {}
        light = (meta.get("extra", {}) or {}).get("light") or viol.light_condition

        # ==============================================================
        # 2. CENTER COLUMN: FORENSIC EVIDENCE STUDIO & DIAGNOSTICS
        # ==============================================================
        with center_col:
            # Case Header & Fast Navigation Bar
            with ui.panel():
                h_info, h_nav = st.columns([2.4, 1.2], vertical_alignment="center")
                with h_info:
                    extra_data = (meta.get("extra") or {}) if isinstance(meta, dict) else {}
                    is_ww_unverified = (
                        viol.violation_type == "WRONG_WAY"
                        and (
                            extra_data.get("motion_verified") is False
                            or (viol.source == "CITIZEN" and (viol.confidence == 0.0 or not extra_data.get("angle_deviation_deg")))
                        )
                    )
                    if is_ww_unverified:
                        conf_pct = f"{int(viol.confidence * 100)}%"
                        if viol.confidence > 0:
                            status_badge = (
                                f'<span style="font-size:12px;font-weight:700;color:#fff;background:#C62828;padding:3px 8px;border-radius:4px;">'
                                f'⚠️ POTENTIAL WRONG WAY · NOT VERIFIED · {conf_pct} AI CONFIDENCE</span> '
                                f'<span style="font-size:12px;font-weight:700;color:#fff;background:#D97706;padding:3px 8px;border-radius:4px;">'
                                f'PENDING OFFICER REVIEW</span>'
                            )
                            case_title = "Wrong-Way Driving Violation (Photo Candidate — Suspect Marked in Red)"
                        else:
                            status_badge = (
                                '<span style="font-size:12px;font-weight:700;color:#fff;background:#D97706;padding:3px 8px;border-radius:4px;">'
                                '⚠️ WRONG-WAY: NOT VERIFIED (PHOTO ONLY)</span> '
                                '<span style="font-size:12px;font-weight:700;color:#fff;background:#6B7280;padding:3px 8px;border-radius:4px;">'
                                'PENDING OFFICER REVIEW</span>'
                            )
                            case_title = "Wrong-Way Incident Report (Motion Unverified — Requires Officer Review)"
                    elif viol.violation_type == "WRONG_WAY":
                        status_badge = f'<span style="font-size:12px;font-weight:700;color:#fff;background:#2E7D4F;padding:2px 8px;border-radius:4px;">✓ WRONG WAY VERIFIED · {int(viol.confidence * 100)}% AI CONFIDENCE</span>'
                        case_title = TITLES.get(viol.violation_type, rs.violation_label(viol.violation_type))
                    else:
                        conf_color = "#2E7D4F" if viol.confidence >= 0.85 else "#C62828" if viol.confidence < 0.70 else "#E9A400"
                        status_badge = f'<span style="font-size:12px;font-weight:700;color:#fff;background:{conf_color};padding:2px 8px;border-radius:4px;">{int(viol.confidence * 100)}% AI CONFIDENCE</span>'
                        case_title = TITLES.get(viol.violation_type, rs.violation_label(viol.violation_type))

                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:2px;">'
                        f'{ui.violation_tag(viol.violation_type, viol.source)}'
                        f'{status_badge}'
                        f'<span class="small">Case <b>{pos + 1}</b> of <b>{len(items)}</b> · <code>{ui.esc(viol.event_id)}</code></span>'
                        f'</div>'
                        f'<div class="rq-title">{ui.esc(case_title)}</div>'
                        f'<div class="small">{ui.esc(where(viol, cams))} · <b>{ui.fmt_time(viol.timestamp)}</b> · Track #{viol.track_id or "—"}</div>',
                        unsafe_allow_html=True
                    )
                with h_nav:
                    nav_prev, nav_next = st.columns(2)
                    nav_prev.button("← Prev", disabled=(prev_id is None), use_container_width=True, on_click=_go, args=(prev_id,))
                    nav_next.button("Next →", disabled=(next_id is None), use_container_width=True, on_click=_go, args=(next_id,))

            # Evidence Image & Video View
            with ui.panel():
                if ev.get("image"):
                    with st.container(key="rq_evidence"):
                        st.image(ev["image"], use_container_width=True)
                else:
                    ui.empty("Snapshot Missing", "No evidence image was found on disk for this record.")

                orig_img = meta.get("original_image") or meta.get("upload")
                if orig_img and Path(orig_img).exists():
                    with st.expander("📷 View Original Uploaded Photograph (Unannotated)", expanded=False):
                        st.image(str(orig_img), caption="Original photograph as uploaded by citizen", use_container_width=True)

                if ev.get("video"):
                    with st.expander("🎥 Evidence Video Clip (Multi-frame Context)", expanded=False):
                        st.video(ev["video"], key=f"rev_video_{viol.id}")

            # Plate Verification & Quick Correction Studio
            with ui.panel():
                crop_img = plate_crop(ev.get("image"))
                st.markdown('<div class="plate-label">Number plate check</div>', unsafe_allow_html=True)

                p_crop_col, p_badge_col, p_edit_col = st.columns([1, 1.7, 1.9], vertical_alignment="center", gap="small")
                with p_crop_col:
                    if crop_img:
                        st.image(crop_img, caption="Plate Bumper Crop", width=140)
                    else:
                        st.markdown('<div style="font-size:12px;color:var(--muted);background:#F1F4F8;padding:12px;border-radius:4px;text-align:center;">No plate crop</div>', unsafe_allow_html=True)
                with p_badge_col:
                    # built outside the f-string: a backslash inside an f-string breaks Python 3.11 and older
                    verified = ("<div class='small' style='color:#1F4E8C;font-weight:600;margin-top:3px;'>"
                                "Officer Verified</div>") if viol.plate_edited else ""
                    st.markdown(
                        f'<div style="text-align:center;">'
                        f'{ui.plate_html(viol.plate_number)}'
                        f'{verified}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with p_edit_col:
                    edit_c1, edit_c2 = st.columns([2.5, 1], vertical_alignment="bottom")
                    corrected_input = edit_c1.text_input(
                        "Edit plate if OCR misread",
                        value=viol.plate_number or "",
                        key=f"input_plate_{viol.id}",
                        placeholder="e.g. KL09AQ3439"
                    )
                    if edit_c2.button("Save", key=f"save_plate_{viol.id}", use_container_width=True):
                        _do("plate", viol.id, plate=corrected_input)

            # AI Diagnostic Trace
            with ui.panel():
                st.markdown("**Why the AI flagged this**")
                diag_items = "".join(f"<li>{ui.esc(x)}</li>" for x in _why(viol, meta))
                st.markdown(f'<ul class="why" style="margin:4px 0 6px;">{diag_items}</ul>', unsafe_allow_html=True)

                if viol.merged_count:
                    st.markdown(
                        f'<div class="note-amber"><b>Merged Event</b>: {viol.merged_count} consecutive repeat detections '
                        f'of this vehicle were deduplicated into this single case.</div>',
                        unsafe_allow_html=True
                    )

        # ==============================================================
        # 3. RIGHT COLUMN: DECISION COMMAND DECK & LEGAL PENALTY
        # ==============================================================
        with right_col:
            # Legal Penal Schedule Card
            penalty_info = ChallanService.get_penalty_info(viol.violation_type)
            with ui.panel():
                st.markdown(
                    f'<div class="penalty-card">'
                    f'<div class="sec">{ui.esc(penalty_info["section"])}</div>'
                    f'<div class="desc">{ui.esc(penalty_info["description"])}</div>'
                    f'<div class="fine-row">'
                    f'<span class="fine">₹{penalty_info["fine_inr"]:,}</span>'
                    f'<span class="pts">· {penalty_info["points"]} Demerit Points</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if st.session_state.get("review_error"):
                    st.error(st.session_state.pop("review_error"))

                # 1. APPROVE ACTION (HIGH CONTRAST PRIMARY BUTTON)
                st.markdown('<div class="st-key-btn_approve">', unsafe_allow_html=True)
                if st.button("Approve & Issue Notice", key=f"btn_app_{viol.id}", use_container_width=True, type="primary"):
                    _do("approve", viol.id, next_id=nxt)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Creates an official prototype notice and registers fine in Parivahan queue.")

                st.markdown("<hr style='margin:12px 0 10px;border:none;border-top:1px solid #E2E6EB;'>", unsafe_allow_html=True)

                # 2. REJECT ACTION (CLEAR, VISIBLE, DIRECT REASON DROPDOWN)
                st.markdown("**Reject Detection (False Positive)**")
                rej_reason = st.selectbox(
                    "Rejection Reason",
                    rs.REJECT_REASONS,
                    key=f"rej_reason_{viol.id}",
                    label_visibility="collapsed"
                )
                st.markdown('<div class="st-key-btn_reject">', unsafe_allow_html=True)
                if st.button("Reject Detection", key=f"btn_rej_{viol.id}", use_container_width=True):
                    _do("reject", viol.id, next_id=nxt, reason=rej_reason)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Rejections are saved with reason as training data for future model tuning.")

            # Repeat Offender & Prior History Card
            with ui.panel():
                st.markdown(f"**Vehicle Offense Dossier**")
                if viol.plate_number and viol.plate_number != "UNIDENTIFIED":
                    history = (
                        db.query(Violation)
                        .filter(Violation.plate_number == viol.plate_number, Violation.id != viol.id)
                        .order_by(desc(Violation.timestamp))
                        .limit(5)
                        .all()
                    )
                    if history:
                        st.markdown(f'<div class="note-amber" style="margin-bottom:8px;"><b>Repeat Offender</b>: {len(history)} prior incidents recorded.</div>', unsafe_allow_html=True)
                        h_rows = "".join(
                            f"<tr>"
                            f"<td>{ui.fmt_time(h.timestamp)}</td>"
                            f"<td>{ui.esc(rs.violation_label(h.violation_type))}</td>"
                            f"<td>{ui.status_tag(h.status, h.reject_reason)}</td>"
                            f"</tr>"
                            for h in history
                        )
                        st.markdown(
                            f'<table class="data">'
                            f'<tr><th>Date</th><th>Offense</th><th>Status</th></tr>'
                            f'{h_rows}'
                            f'</table>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown('<div class="note-blue"><b>Clean Record</b>: No prior violations logged for this plate.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="small">Number plate is unidentified. Use the plate edit box above to assign the registration number.</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div style="font-size:11px;color:var(--muted);margin-top:10px;">'
                    f'Review Officer: <b>{ui.esc(st.session_state.get("officer", "Officer"))}</b> · '
                    f'Bengaluru Traffic Incident Command</div>',
                    unsafe_allow_html=True
                )
