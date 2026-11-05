"""Live cameras: run the detection pipeline on up to four sources side by side."""

import datetime
import time
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st

from algorithms.calibration.camera_calibration import load_config_fresh
from algorithms.detection import model_registry as mr
from backend.config.settings import settings
from backend.db_models.models import CameraSource, Violation
from backend.services import review_service as rs
from dashboard import ui
from dashboard.data import session
from dashboard.views.cameras_setup import _add_camera_form

WEBCAM_TIP = ("Phone demo: hold your phone with the violation video or photo facing the laptop webcam, about 20–30 cm "
              "away. The AI detects the vehicles and riders, tracks them and captures the violation live. "
              "Turn on the high-sensitivity preset under Detection options for phone-screen glare.")
ALL = "All locations"



def _for_screen(frame_bgr, max_w: int = 960):
    """Frame shown in the browser: at most 960 px wide, RGB. Sending full 1080p frames every update is slow."""
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        frame_bgr = cv2.resize(frame_bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def _matches(cams, loc, q, only):
    return [c for c in cams
            if (loc == ALL or c["location"] == loc)
            and (not q or q in c["name"].lower() or q in c["location"].lower())
            and (only == "All" or (c["status"] == "ONLINE") == (only == "Online"))]


def _refilter(cams):
    """When the filter changes, show the first four cameras that match it."""
    ss = st.session_state
    shown = _matches(cams, ss.get("live_loc", ALL), (ss.get("live_q") or "").strip().lower(), ss.get("live_status", "All"))
    ss.live_sel = [c["id"] for c in shown][:4]


def _cameras(db):
    """Every registered camera: (id, name, location, type, source, status)."""
    out = []
    for c in db.query(CameraSource).order_by(CameraSource.location, CameraSource.name).all():
        src = c.source_url
        if c.source_type == "WEBCAM":
            src = int(src) if str(src).isdigit() else 0
        out.append({"id": c.id, "name": c.name, "location": c.location or "—", "type": c.source_type,
                    "src": src, "status": c.status or "OFFLINE", "lat": float(c.latitude or 12.9716), "lon": float(c.longitude or 77.5946)})
    return out


def _set_status(cam_id, status):
    with session() as db:
        c = db.query(CameraSource).filter(CameraSource.id == cam_id).first()
        if c is not None and c.status != status:
            c.status = status
            db.commit()


def _side_panel():
    today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    with session() as db:
        rows = db.query(Violation).filter(Violation.timestamp >= today).all()
    st_counts = Counter(r.status for r in rows)
    with ui.panel():
        st.markdown("### Today so far")
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
            f'{ui.stat(st_counts.get("PENDING_REVIEW", 0), "Waiting for review", amber=True)}'
            f'{ui.stat(st_counts.get("APPROVED", 0), "Approved")}'
            f'{ui.stat(st_counts.get("REJECTED", 0), "Rejected by officers")}'
            f'{ui.stat(sum(1 for r in rows if r.source == "CITIZEN"), "Citizen reports")}</div>'
            '<div class="small" style="margin-top:8px">Rejected detections show where the model is wrong. '
            'They are saved for the next training round.</div>', unsafe_allow_html=True)
    with ui.panel():
        st.markdown("### By violation, today")
        st.markdown(ui.bars(Counter(rs.violation_label(r.violation_type) for r in rows).most_common()),
                    unsafe_allow_html=True)
    with ui.panel():
        st.markdown("### By hour, today")
        hours = Counter(r.timestamp.hour for r in rows if r.timestamp)
        if hours:
            top = max(hours.values())
            cells = "".join(
                f'<i title="{h}:00 · {hours.get(h, 0)}" style="flex:1;display:block;border-radius:2px 2px 0 0;'
                f'background:{"#1F4E8C" if hours.get(h, 0) == top else "#AFC3DB"};height:{100 * hours.get(h, 0) / top:.0f}%"></i>'
                for h in range(24))
            st.markdown(f'<div style="display:flex;align-items:flex-end;gap:2px;height:60px">{cells}</div>'
                        '<div class="small" style="display:flex;justify-content:space-between"><span>0</span>'
                        '<span>6</span><span>12</span><span>18</span><span>23 h</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small">No violations today yet.</div>', unsafe_allow_html=True)
    with ui.panel():
        st.markdown("### Detection models in use")
        custom = Path(settings.MODEL_CUSTOM)
        helmet = (f"{custom.name}, trained {datetime.datetime.fromtimestamp(custom.stat().st_mtime):%d %b %Y}"
                  if custom.exists() else "Not trained yet: helmet rule is off")
        cur = (load_config_fresh().get("detection") or {}).get("model_traffic", settings.MODEL_TRAFFIC)
        st.markdown(f'<dl class="kv"><dt>Vehicles, people</dt><dd>{ui.esc(mr.label(cur))}</dd>'
                    f'<dt>Helmet, plate</dt><dd>{ui.esc(helmet)}</dd>'
                    f'<dt>Plate reading</dt><dd>{ui.esc(settings.OCR_ENGINE)} + multi-frame voting</dd>'
                    f'<dt>Tracker</dt><dd>ByteTrack</dd></dl>', unsafe_allow_html=True)


def render():
    with session() as db:
        cams = _cameras(db)
    running = st.session_state.get("live_running", False)

    main, side = st.columns([2.6, 1], gap="medium")
    with main:
        head, add = st.columns([4, 1], vertical_alignment="center")
        with head:
            ui.page_header("Live cameras", "Pick a location, choose up to four cameras and press Start")
        with add.popover("＋ Add camera", use_container_width=True, disabled=running):
            _add_camera_form()
        with ui.panel():
            if not cams:
                ui.empty("No cameras yet", "Use “＋ Add camera” to register a CCTV stream (RTSP), a webcam or a video file.")
                return
            locations = sorted({c["location"] for c in cams})
            f1, f2, f3 = st.columns([1.6, 1.4, 1])
            loc = f1.selectbox("Filter by location", [ALL] + locations, key="live_loc", disabled=running,
                               on_change=_refilter, args=(cams,))
            q = f2.text_input("Search camera", placeholder="name or junction", key="live_q", disabled=running,
                              on_change=_refilter, args=(cams,)).strip().lower()
            only = f3.selectbox("Status", ["All", "Online", "Offline"], key="live_status", disabled=running,
                                on_change=_refilter, args=(cams,))
            shown = _matches(cams, loc, q, only)
            by_id = {c["id"]: c for c in cams}
            prev = [i for i in st.session_state.get("live_sel", []) if i in by_id]
            options = list(dict.fromkeys([c["id"] for c in shown] + prev))
            if "live_sel" not in st.session_state:
                st.session_state.live_sel = [c["id"] for c in shown][:1]
            sel = st.multiselect(f"Cameras to show ({len(shown)} match the filter, up to 4 at a time)", options,
                                 key="live_sel", max_selections=4, disabled=running,
                                 format_func=lambda i: f"{by_id[i]['name']} · {by_id[i]['location']}"
                                                       f"{'' if by_id[i]['status'] == 'ONLINE' else ' (offline)'}")
            a, d, e = st.columns([2, 1, 1], vertical_alignment="bottom")
            signal = a.selectbox("Signal", ["AUTO", "RED", "GREEN", "YELLOW"], disabled=running,
                                 help="AUTO reads the lamp inside the signal area drawn in Camera setup. "
                                      "RED / GREEN / YELLOW force a simulated signal phase for testing.")
            d.button("Start", type="primary", disabled=running or not sel, use_container_width=True,
                     on_click=lambda: st.session_state.update(live_running=True))
            e.button("Stop", disabled=not running, use_container_width=True,
                     on_click=lambda: st.session_state.update(live_running=False))
            webcam_chosen = any(by_id[i]["type"] == "WEBCAM" for i in sel)
            conf_default = float((load_config_fresh().get("detection") or {}).get("confidence", settings.CONFIDENCE))
            with st.expander("Detection options", expanded=False):
                o1, o2, o3 = st.columns([1.2, 1.2, 1.1])
                conf = o1.slider("Detection confidence threshold", 0.15, 0.70,
                                 round(min(max(0.25 if webcam_chosen else conf_default, 0.15), 0.70), 2), 0.05,
                                 disabled=running)
                high_sens = o2.toggle("High-sensitivity preset", value=webcam_chosen, disabled=running,
                                      help="Optimised for phone-screen glare: a violation is confirmed in 2 frames "
                                           "instead of 6, with a 6-second cooldown.")
                max_frames = o3.number_input("Stop after N frames per camera (0 = keep running)", 0, 100000, 0,
                                             step=100, disabled=running)
                m1, m2 = st.columns([1.2, 2.3])
                keys = mr.available()
                cur = mr.key_for_path((load_config_fresh().get("detection") or {}).get("model_traffic", settings.MODEL_TRAFFIC))
                model_key = m1.selectbox("Detection model", keys, index=keys.index(cur) if cur in keys else 0,
                                         format_func=mr.label, disabled=running, key="live_model")
                others = ["none"] + [k for k in keys if k != model_key]
                cmp_key = m2.selectbox("Compare side by side with (practice run on the first selected camera, nothing saved)",
                                       others, format_func=lambda k: "— no comparison —" if k == "none" else mr.label(k),
                                       key="live_compare_with", disabled=running)
                compare = cmp_key != "none"
            if webcam_chosen:
                st.markdown(f'<div class="note-blue" style="margin-top:6px">{ui.esc(WEBCAM_TIP)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="small" style="margin-top:6px">To check a recorded video or a citizen\'s clip, '
                        'use the <b>Upload proof</b> page.</div>', unsafe_allow_html=True)
        alert_banner = st.empty()

        if compare and sel:
            active = [(0, {**by_id[sel[0]], "model": model_key}), (1, {**by_id[sel[0]], "model": cmp_key})]
        else:
            active = [(i, {**by_id[cid], "model": model_key}) for i, cid in enumerate(sel[:4])]
        grid = [st.columns(2), st.columns(2)]
        slots = []
        for i in range(4):
            with grid[i // 2][i % 2]:
                slots.append((st.empty(), st.empty()))
        with ui.panel():
            st.markdown("### Incident evidence archive")
            evidence_slot = st.empty()
            evidence_slot.markdown('<div class="small">The frame of the latest confirmed violation appears here '
                                   'during a run.</div>', unsafe_allow_html=True)
        for i, (img, cap) in enumerate(slots):
            cam = next((c for j, c in active if j == i), None)
            if cam is None:
                img.markdown('<div style="background:#2A3038;border-radius:8px;height:240px;display:grid;place-items:center;'
                             'color:#AAB4BE">No camera selected</div>', unsafe_allow_html=True)
            elif not running:
                dot = "#4CC38A" if cam["status"] == "ONLINE" else "#AAB4BE"
                lat_c = cam.get("lat", 12.9716)
                lon_c = cam.get("lon", 77.5946)
                gmaps = f"https://www.google.com/maps?q={lat_c},{lon_c}"
                img.markdown(f'<div style="background:#2A3038;border-radius:8px;height:240px;display:grid;place-items:center;'
                             f'color:#E6EAEE;text-align:center"><div><b class="cond" style="font-size:20px">{ui.esc(cam["name"])}</b>'
                             f'<br><span style="color:#AAB4BE">{ui.esc(cam["location"])}</span>'
                             f'<br><span style="color:{dot}">● {ui.esc(cam["status"].title())}</span>'
                             f'<span style="color:#AAB4BE"> · {ui.esc(cam["type"])}</span>'
                             f'<br><a href="{gmaps}" target="_blank" style="text-decoration:none;">'
                             f'<span style="display:inline-block;margin-top:6px;background:#1F4E8C;color:#fff;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600;">'
                             f'1-Click Location ({lat_c:.4f}° N, {lon_c:.4f}° E)</span></a>'
                             f'</div></div>',
                             unsafe_allow_html=True)

    with side:
        _side_panel()

    if not running or not active:
        if running and not active:
            st.session_state.live_running = False
        return

    from pipeline.traffic_pipeline import TrafficVisionPipeline

    streams = []
    for i, cam in active:
        label, src = f'{cam["name"]} · {mr.short(cam["model"])}', cam["src"]
        placeholder_url = isinstance(src, str) and (".mock" in src or "demo:demo@" in src)
        capture = None if placeholder_url else cv2.VideoCapture(src)
        if capture is None or not capture.isOpened():
            _set_status(cam["id"], "OFFLINE")
            slots[i][0].markdown(f'<div style="background:#2A3038;border-radius:8px;height:220px;display:grid;'
                                 f'place-items:center;color:#E6EAEE;text-align:center"><div><b class="cond" style="font-size:20px">'
                                 f'{ui.esc(label)} is offline</b><br><span class="small">Could not open the stream. '
                                 f'Check the address in Camera setup.</span></div></div>', unsafe_allow_html=True)
            continue
        _set_status(cam["id"], "ONLINE")
        if not mr.is_downloaded(cam["model"]):
            st.toast(f"Downloading {mr.short(cam['model'])} weights (first use, needs internet)…")
        pipe = TrafficVisionPipeline(camera_id=cam["id"], location=cam["location"],
                                     model_traffic=mr.path(cam["model"]), persist=not compare)
        pipe.detector.confidence = conf
        if high_sens:
            for rule in pipe.rule_engine.rules.values():
                rule.min_positive_frames = 2
                rule.cooldown_seconds = 6.0
        sig = "RED" if signal == "AUTO" and "scenario_red_light" in str(src) else signal
        streams.append({"i": i, "label": label, "cap": capture, "pipe": pipe, "n": 0, "sig": sig,
                        "fps": capture.get(cv2.CAP_PROP_FPS) or 25.0, "new": 0, "done": False})
    try:
        while st.session_state.get("live_running") and any(not s["done"] for s in streams):
            for s in streams:
                if s["done"]:
                    continue
                ok, frame = s["cap"].read()
                if not ok or (max_frames and s["n"] >= max_frames):
                    s["done"] = True
                    slots[s["i"]][1].markdown(f'<div class="small">{ui.esc(s["label"])} · finished after {s["n"]} frames · '
                                              f'{s["new"]} new detections · {s["pipe"].merged_duplicates} merged</div>',
                                              unsafe_allow_html=True)
                    continue
                s["n"] += 1
                t0 = time.perf_counter()
                annotated, new = s["pipe"].process_frame(frame, s["n"], fps=s["fps"], signal_state=s["sig"])
                s["t"] = s.get("t", 0.0) + time.perf_counter() - t0
                s["new"] += len(new)
                slots[s["i"]][0].image(_for_screen(annotated), use_container_width=True, output_format="JPEG")
                slots[s["i"]][1].markdown(
                    f'<div class="small"><b style="color:#18212B">{ui.esc(s["label"])}</b> · frame {s["n"]} · '
                    f'{s["pipe"].current_fps:.1f} fps · {s["pipe"].avg_latency_ms:.0f} ms latency · '
                    f'{len(s["pipe"].total_vehicles_seen)} vehicles · '
                    f'{s["new"]} new detections · light: {s["pipe"].last_light}</div>', unsafe_allow_html=True)
                if new and compare:
                    st.toast(f"{s['label']}: would flag {rs.violation_label(new[-1].get('violation_type'))} "
                             "(practice run, not saved)")
                elif new:
                    v = new[-1]
                    vlabel = rs.violation_label(v.get("violation_type"))
                    alert_banner.markdown(
                        f'<div style="background:var(--stop-tint);border:2px solid var(--stop);border-radius:8px;padding:10px 16px;'
                        f'margin:8px 0;display:flex;justify-content:space-between;align-items:center"><div>'
                        f'<b style="color:var(--stop);font-size:17px">Violation confirmed: {ui.esc(vlabel)}</b>'
                        f'<div class="small">{ui.esc(s["label"])} · frame {s["n"]} · track ID {ui.esc(v.get("track_id"))} · '
                        f'saved with evidence, waiting in the review queue</div></div>'
                        f'<span class="tag t-stop">RECORDED</span></div>', unsafe_allow_html=True)
                    evidence_slot.image(_for_screen(annotated), use_container_width=True, output_format="JPEG",
                                        caption=f"{vlabel} · track ID {v.get('track_id')} · {s['label']} · frame {s['n']}")
                    st.toast(f"New: {rs.violation_label(new[-1].get('violation_type'))} on {s['label']}. "
                             "It is waiting in the review queue.")
            time.sleep(0.001)
    finally:
        for s in streams:
            s["cap"].release()
            s["pipe"].evidence_recorder.flush()
    st.session_state.live_running = False
    if compare and streams:
        rows = ""
        for s in streams:
            d = s["pipe"].detector.stats
            calls = max(1, d["calls"])
            rows += (f'<tr><td><b>{ui.esc(s["label"].split(" · ")[-1])}</b></td><td>{s["n"]}</td>'
                     f'<td>{s["n"] / max(1e-6, s.get("t", 0.0)):.1f}</td>'
                     f'<td>{d["ms"] / calls:.0f} ms</td><td>{d["boxes"] / calls:.1f}</td>'
                     f'<td>{(d["conf_sum"] / d["boxes"]) if d["boxes"] else 0:.2f}</td>'
                     f'<td>{len(s["pipe"].total_vehicles_seen)}</td><td>{s["new"]}</td></tr>')
        with ui.panel():
            st.markdown("### Side-by-side result (same camera, same frames)")
            st.markdown('<table class="data"><tr><th>Model</th><th>Frames</th><th>Pipeline FPS</th><th>Detection time / frame</th>'
                        '<th>Objects / frame</th><th>Mean confidence</th><th>Vehicles tracked</th><th>Violations it would flag</th></tr>'
                        f'{rows}</table>', unsafe_allow_html=True)
            st.markdown('<div class="small">Practice run: nothing was saved. Both models ran one after the other on this computer, '
                        'so the timings are directly comparable. For a saved report use the Evaluation page.</div>',
                        unsafe_allow_html=True)
        return
    total_new = sum(s["new"] for s in streams)
    total_frames = sum(s["n"] for s in streams)
    st.success(f"Stream finished. Analysed {total_frames} frames and recorded {total_new} violation events. "
               "New detections are waiting in the review queue. Press Stop, then Start to run again.")
