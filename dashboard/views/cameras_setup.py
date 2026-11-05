"""Camera setup: map of all camera locations, 1-click GPS detection, and calibration."""

import math
import time
import urllib.parse
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.calibration.camera_calibration import (
    RULE_KEYS,
    get_camera_calibration,
    load_config_fresh,
    save_camera_calibration,
)
from backend.db_models.models import CameraSource
from backend.services import geo_service
from dashboard import ui
from dashboard.data import session

RULE_NAMES = {
    "helmet": ("No helmet", "Needs the trained helmet model"),
    "triple_riding": ("Triple riding", ""),
    "red_light": ("Red light jump", "Needs the stop line and the signal area"),
    "wrong_way": ("Wrong way", "Needs the lane direction"),
    "seatbelt": ("Seatbelt compliance", "Evaluates driver/passenger front/side cabin with YOLO26n"),
    "illegal_parking": ("Illegal parking", "Needs a no-parking zone"),
    "unsafe_movement": ("Unsafe movement", "Weaving from the tracked path; over-speeding needs pixels per metre"),
}


def _parse_zones(text: str, w: int, h: int):
    """'x,y; x,y; x,y' per line -> list of polygons (pixels). Lines with fewer than 3 points are skipped."""
    zones = []
    for line in (text or "").splitlines():
        pts = []
        for part in line.replace("|", ";").split(";"):
            bits = [b for b in part.replace(" ", "").split(",") if b]
            if len(bits) == 2:
                try:
                    x, y = float(bits[0]), float(bits[1])
                except ValueError:
                    continue
                pts.append([int(min(max(x, 0), w)), int(min(max(y, 0), h))])
        if len(pts) >= 3:
            zones.append(pts)
    return zones


def _zones_text(zones) -> str:
    return "\n".join("; ".join(f"{int(p[0])},{int(p[1])}" for p in z) for z in (zones or []))


def _is_placeholder(src) -> bool:
    return isinstance(src, str) and (".mock" in src or "demo:demo@" in src)


def _source(cam):
    if cam.source_type == "WEBCAM":
        return int(cam.source_url) if str(cam.source_url).isdigit() else 0
    return cam.source_url


@st.cache_data(show_spinner=False, ttl=600)
def _grab_frame(src, nonce: int = 0):
    if _is_placeholder(src):
        return None
    if isinstance(src, str) and "://" in src:
        # network stream (CCTV / phone): give up after 4 s so an unreachable camera never freezes the page
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000,
                                                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000])
    else:
        cap = cv2.VideoCapture(src)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    for _ in range(10):  # skip black first frames of some encoders
        if ok and frame is not None and frame.mean() > 8:
            break
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _mask(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://••••••@{rest.split('@', 1)[1]}"
    return url


def _draw(frame, cal):
    img = frame.copy()
    h, w = img.shape[:2]
    sl = cal.get("stop_line")
    if sl:
        p1, p2 = (int(sl["x1"]), int(sl["y1"])), (int(sl["x2"]), int(sl["y2"]))
        cv2.line(img, p1, p2, (40, 40, 220), 4)
        for p in (p1, p2):
            cv2.circle(img, p, 9, (255, 255, 255), -1)
            cv2.circle(img, p, 9, (40, 40, 220), 3)
        cv2.putText(img, "Stop line", (p1[0] + 12, p1[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 220), 2)
    lane = cal.get("lane_direction")
    if lane:
        dx, dy = lane
        n = math.hypot(dx, dy) or 1
        for fx in (0.3, 0.5, 0.7):
            cx, cy = int(w * fx), int(h * 0.55)
            L = min(w, h) * 0.18
            cv2.arrowedLine(img, (int(cx - dx / n * L), int(cy - dy / n * L)),
                            (int(cx + dx / n * L), int(cy + dy / n * L)),
                            (255, 200, 90), 4, tipLength=0.25)
        cv2.putText(img, "Allowed direction", (int(w * 0.3) - 60, int(h * 0.55) - int(min(w, h) * 0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 90), 2)
    for i, z in enumerate(cal.get("no_parking_zones") or []):
        pts = np.array(z, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (60, 60, 230))
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [pts], True, (60, 60, 230), 3)
        cv2.putText(img, f"No parking {i + 1}", (int(pts[:, 0].min()) + 6, int(pts[:, 1].min()) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 230), 2)
    roi = cal.get("signal_roi")
    if roi:
        x1, y1, x2, y2 = [int(v) for v in roi]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 195, 255), 3)
        cv2.putText(img, "Signal", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 195, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _status(cal) -> str:
    if not cal:
        return '<span class="tag t-amber">Not calibrated</span>'
    if not cal.get("stop_line"):
        return '<span class="tag t-amber">Stop line missing</span>'
    return '<span class="tag t-go">Calibrated</span>'


def _add_camera_form():
    st.markdown("### Add New Camera")
    
    # 1-Click Location Helpers
    hlp_col1, hlp_col2 = st.columns(2)
    if hlp_col1.button("1-Click Detect GPS Location", help="Use current device network/GPS location", use_container_width=True):
        loc = geo_service.get_current_location()
        if loc:
            st.session_state.add_cam_lat = round(loc["lat"], 4)
            st.session_state.add_cam_lon = round(loc["lon"], 4)
            st.session_state.add_cam_loc = loc["address"]
            st.toast(f"GPS Location detected: {loc['address']}")
            st.rerun()

    loc_candidate = st.session_state.get("add_cam_loc_query", "")
    if hlp_col2.button("Auto-Geocode from Landmark", help="Find coordinates from landmark name", use_container_width=True):
        if loc_candidate:
            res = geo_service.geocode_address(loc_candidate)
            if res:
                st.session_state.add_cam_lat = round(res[0], 4)
                st.session_state.add_cam_lon = round(res[1], 4)
                st.session_state.add_cam_loc = res[2].split(",")[0] + ", Bengaluru"
                st.toast(f"Coordinates found for {loc_candidate}")
                st.rerun()
            else:
                st.warning(f"Could not find coordinates for '{loc_candidate}'.")
        else:
            st.info("Enter a landmark in the Location field first to auto-geocode.")

    with st.form("add_cam", clear_on_submit=False):
        name = st.text_input("Name", placeholder="e.g. Indiranagar 100ft Rd · Cam 1")
        loc_val = st.text_input(
            "Location / Landmark",
            value=st.session_state.get("add_cam_loc", ""),
            placeholder="e.g. Indiranagar 100ft Road, Bengaluru",
            key="add_cam_loc_query"
        )
        kind = st.selectbox("Type", ["RTSP", "VIDEO_FILE", "WEBCAM"])
        url = st.text_input("Stream address or file path", placeholder="rtsp://user:pass@10.0.0.5:554/stream1")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=float(st.session_state.get("add_cam_lat", 12.9716)), format="%.4f")
        lon = c2.number_input("Longitude", value=float(st.session_state.get("add_cam_lon", 77.5946)), format="%.4f")
        
        submitted = st.form_submit_button("Save & Register Camera", type="primary", use_container_width=True)
        if submitted:
            if not name.strip() or not url.strip():
                st.error("Please enter a camera name and stream address/path.")
            else:
                with session() as db:
                    new_cam = CameraSource(
                        name=name.strip(),
                        location=loc_val.strip() or name.strip(),
                        source_type=kind,
                        source_url=url.strip(),
                        latitude=lat,
                        longitude=lon,
                        status="ONLINE" if kind == "VIDEO_FILE" else "OFFLINE"
                    )
                    db.add(new_cam)
                    db.commit()
                st.toast(f"Camera '{name}' registered with location ({lat}, {lon})!")
                st.session_state.show_add_cam = False
                st.rerun()


def _render_all_cameras_map(raw_cams):
    """Interactive GIS map plotting ALL registered cameras with 1-click GPS detection."""
    pts = [c for c in raw_cams if c.latitude and c.longitude]
    
    # Top Action Bar
    bar_c1, bar_c2, bar_c3 = st.columns([2.2, 1.8, 1.2], vertical_alignment="center")
    with bar_c1:
        online_cnt = sum(1 for c in raw_cams if c.status == "ONLINE")
        st.markdown(f"**{len(raw_cams)} Cameras** · <span style='color:#2E7D4F;'>● {online_cnt} Online</span> · {len(pts)} Plotted on Map", unsafe_allow_html=True)
    with bar_c2:
        if st.button("1-Click Get Current Location", type="primary", use_container_width=True):
            loc_data = geo_service.get_current_location()
            if loc_data:
                st.session_state.user_loc = loc_data
                st.toast(f"Detected: {loc_data['address']} ({loc_data['lat']:.4f}, {loc_data['lon']:.4f})")
                st.rerun()
    with bar_c3:
        if st.button("Add Camera", use_container_width=True):
            st.session_state.show_add_cam = not st.session_state.get("show_add_cam", False)
            st.rerun()

    if st.session_state.get("show_add_cam"):
        with ui.panel():
            _add_camera_form()

    user_loc = st.session_state.get("user_loc")
    if user_loc:
        nearest_cam, dist_km = geo_service.find_nearest_camera(user_loc["lat"], user_loc["lon"], raw_cams)
        near_text = f" · Nearest CCTV Junction: <b>{nearest_cam.name}</b> ({dist_km} km away)" if nearest_cam else ""
        st.markdown(
            f'<div class="note-blue" style="margin-bottom:12px;">'
            f'<b>Your Current Location:</b> {user_loc["address"]} (Lat: {user_loc["lat"]:.4f}, Lon: {user_loc["lon"]:.4f}){near_text}'
            f'</div>',
            unsafe_allow_html=True
        )

    # Render Plotly Map
    if pts:
        df = pd.DataFrame([
            {
                "id": c.id,
                "name": c.name,
                "location": c.location,
                "lat": float(c.latitude),
                "lon": float(c.longitude),
                "status": c.status or "OFFLINE",
                "type": c.source_type,
            }
            for c in pts
        ])
        colors = ["#2E7D4F" if s == "ONLINE" else "#8A949E" for s in df["status"]]
        hover = (
            "<b>" + df["name"] + "</b><br>"
            + df["location"] + "<br>"
            + "Status: " + df["status"] + " (" + df["type"] + ")<br>"
            + "GPS: " + df["lat"].astype(str) + ", " + df["lon"].astype(str)
        )

        fig = go.Figure()

        ScatterClass = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
        if ScatterClass:
            fig.add_trace(ScatterClass(
                lat=df["lat"],
                lon=df["lon"],
                mode="markers+text",
                text=df["name"].apply(lambda n: n.split(" · ")[0][:20]),
                textposition="top center",
                textfont=dict(size=11, color="#1F4E8C", family="Barlow, sans-serif"),
                hoverinfo="text",
                hovertext=hover,
                marker=dict(size=14, color=colors, opacity=0.92),
                name="CCTV Cameras"
            ))

            if user_loc:
                fig.add_trace(ScatterClass(
                    lat=[user_loc["lat"]],
                    lon=[user_loc["lon"]],
                    mode="markers+text",
                    text=["YOU ARE HERE"],
                    textposition="bottom center",
                    textfont=dict(size=12, color="#C62828", family="Barlow, sans-serif"),
                    hoverinfo="text",
                    hovertext=[f"Your Location: {user_loc['address']}"],
                    marker=dict(size=16, color="#C62828"),
                    name="Your Location"
                ))

            center_lat = user_loc["lat"] if user_loc else df["lat"].mean()
            center_lon = user_loc["lon"] if user_loc else df["lon"].mean()
            
            map_kwargs = dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11 if not user_loc else 12
            )
            if hasattr(go, "Scattermap"):
                fig.update_layout(map=map_kwargs, height=440, margin=dict(l=0, r=0, t=0, b=0))
            else:
                fig.update_layout(mapbox=map_kwargs, height=440, margin=dict(l=0, r=0, t=0, b=0))

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        ui.empty("No geolocated cameras", "Cameras added without coordinates will not appear on the map.")

    # Camera Directory Cards
    st.markdown("### Camera Locations & Quick Actions")
    cols = st.columns(2, gap="medium")
    for i, c in enumerate(raw_cams):
        with cols[i % 2]:
            with ui.panel():
                online = c.status == "ONLINE"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<b>{ui.esc(c.name)}</b>'
                    f'<span class="tag t-{"go" if online else "grey"}">● {c.status or "OFFLINE"}</span>'
                    f'</div>'
                    f'<div class="small" style="color:var(--muted);margin-top:2px;">{ui.esc(c.location)}</div>'
                    f'<div class="small" style="margin-top:4px;">'
                    f'<b>GPS:</b> <code>{c.latitude:.4f}, {c.longitude:.4f}</code> · <b>Type:</b> {c.source_type}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                ac1, ac2, ac3 = st.columns(3)
                if ac1.button("Auto-Locate", key=f"geofix_{c.id}", help="Fetch exact GPS from location text", use_container_width=True):
                    res = geo_service.geocode_address(c.location)
                    if res:
                        with session() as db:
                            db_c = db.query(CameraSource).filter(CameraSource.id == c.id).first()
                            if db_c:
                                db_c.latitude = round(res[0], 4)
                                db_c.longitude = round(res[1], 4)
                                db_c.commit()
                        st.toast(f"Updated {c.name} to ({res[0]:.4f}, {res[1]:.4f})")
                        st.rerun()
                    else:
                        st.warning("Could not auto-locate this landmark.")

                maps_url = f"https://www.google.com/maps/search/?api=1&query={c.latitude},{c.longitude}"
                ac2.markdown(
                    f'<a href="{maps_url}" target="_blank" style="text-decoration:none;">'
                    f'<button style="width:100%;height:38px;background:#F1F4F8;border:1px solid #D8E0EA;border-radius:6px;'
                    f'cursor:pointer;font-weight:600;font-size:12px;color:#1F4E8C;">Maps</button></a>',
                    unsafe_allow_html=True
                )

                if ac3.button("Calibrate", key=f"caljump_{c.id}", use_container_width=True):
                    st.session_state.cam_sel = c.id
                    st.session_state.cam_tab = "Calibration"
                    st.session_state.cam_mode = "Camera Calibration & Virtual Lines"
                    st.rerun()


def _render_calibration_view(cams, status, cfg):
    """The interactive stop line, lane direction and signal area calibration studio."""
    left, mid, right = st.columns([1, 2.4, 1.3], gap="medium")
    with left:
        with ui.panel():
            if not cams:
                ui.empty("No cameras", "Add a camera to start.")
            ids = [c[0] for c in cams]
            sel = st.session_state.get("cam_sel")
            if sel not in ids:
                sel = ids[0] if ids else None
            for cid, name, loc, kind_, url_ in cams:
                cal = get_camera_calibration(cfg, cid)
                online = status.get(cid) == "ONLINE"
                with ui.panel(selected=True) if cid == sel else st.container():
                    st.markdown(f"**{ui.esc(name)}**<br><span class='small'>{ui.esc(loc)}</span><br>"
                                f"<span class='small' style='color:{'#2E7D4F' if online else '#8A949E'}'>● "
                                f"{ui.esc(status.get(cid) or 'UNKNOWN')}</span> "
                                f"<span class='small'>· {ui.esc(kind_)} · {ui.esc(_mask(str(url_)))}</span><br>{_status(cal)}",
                                unsafe_allow_html=True)
                    if cid != sel and st.button("Select", key=f"camsel_{cid}"):
                        st.session_state.cam_sel = cid
                        st.rerun()
        if st.session_state.get("show_add_cam"):
            with ui.panel():
                _add_camera_form()
    if not cams:
        return
    cam = next(c for c in cams if c[0] == sel)
    cid, name, loc, kind, url = cam
    src = int(url) if kind == "WEBCAM" and str(url).isdigit() else url
    saved = get_camera_calibration(cfg, cid)
    frame = _grab_frame(src, st.session_state.get(f"nonce_{cid}", 0))

    with mid:
        with ui.panel():
            if frame is None:
                ui.empty("No frame from this camera",
                         "The stream could not be opened. Check the address, or use a recorded video for calibration.")
                h, w = 720, 1280
            else:
                h, w = frame.shape[:2]
            with st.container(key="chips_tool"):
                tool = st.radio("Tool", ["Stop line", "Lane direction", "Signal light area", "No-parking zone",
                                         "Speed"], horizontal=True, label_visibility="collapsed")
            sl = saved.get("stop_line") or {"x1": w * 0.12, "y1": h * 0.72, "x2": w * 0.88, "y2": h * 0.72}
            lane = saved.get("lane_direction") or [0, 1]
            roi = saved.get("signal_roi") or [w * 0.85, h * 0.05, w * 0.95, h * 0.30]
            if tool == "Stop line":
                c = st.columns(4)
                sl = {"x1": c[0].number_input("Left x", 0, w, int(sl["x1"])), "y1": c[1].number_input("Left y", 0, h, int(sl["y1"])),
                       "x2": c[2].number_input("Right x", 0, w, int(sl["x2"])), "y2": c[3].number_input("Right y", 0, h, int(sl["y2"]))}
                st.session_state[f"sl_{cid}"] = sl
            elif tool == "Lane direction":
                ang0 = int(round(math.degrees(math.atan2(lane[1], lane[0])))) % 360
                ang = st.slider("Direction traffic is allowed to move (0° = right, 90° = down the image, 270° = up)",
                                0, 359, ang0)
                lane = [round(math.cos(math.radians(ang)), 3), round(math.sin(math.radians(ang)), 3)]
                st.session_state[f"lane_{cid}"] = lane
            elif tool == "No-parking zone":
                tk = f"npz_txt_{cid}"
                if tk not in st.session_state:
                    st.session_state[tk] = _zones_text(st.session_state.get(f"npz_{cid}", saved.get("no_parking_zones") or []))
                zt = st.text_area(
                    "One zone per line, as corner points x,y separated by ';' (at least 3 points)",
                    key=tk,
                    placeholder=f"e.g. {int(w * 0.05)},{int(h * 0.6)}; {int(w * 0.3)},{int(h * 0.6)}; "
                                f"{int(w * 0.3)},{int(h * 0.95)}; {int(w * 0.05)},{int(h * 0.95)}",
                    height=90)
                st.session_state[f"npz_{cid}"] = _parse_zones(zt, w, h)
                st.markdown('<div class="small">A vehicle whose wheels stay inside a zone without moving for longer '
                            'than the allowed time (config.yaml: parking_max_seconds) is flagged.</div>',
                            unsafe_allow_html=True)
            elif tool == "Speed":
                c = st.columns(2)
                for k, v in ((f"ppm_in_{cid}", float(st.session_state.get(f"ppm_{cid}", saved.get("pixels_per_meter") or 0.0))),
                             (f"lim_in_{cid}", int(st.session_state.get(f"lim_{cid}", saved.get("speed_limit_kmph") or 40)))):
                    if k not in st.session_state:
                        st.session_state[k] = v
                ppm = c[0].number_input("Pixels per metre on the road (0 = off)", 0.0, 500.0, key=f"ppm_in_{cid}",
                                        step=1.0, help="Measure a known length on the road in the image, e.g. a 3 m "
                                                       "lane marking that is 60 px long gives 20 px per metre.")
                lim = c[1].number_input("Speed limit (km/h)", 10, 120, key=f"lim_in_{cid}")
                st.session_state[f"ppm_{cid}"] = ppm
                st.session_state[f"lim_{cid}"] = lim
                st.markdown('<div class="small">Without pixels per metre only weaving (zig-zag) is checked. The speed '
                            'is approximate: one scale for the whole image ignores perspective.</div>',
                            unsafe_allow_html=True)
            else:
                c = st.columns(4)
                roi = [c[0].number_input("Left", 0, w, int(roi[0])), c[1].number_input("Top", 0, h, int(roi[1])),
                       c[2].number_input("Right", 0, w, int(roi[2])), c[3].number_input("Bottom", 0, h, int(roi[3]))]
                st.session_state[f"roi_{cid}"] = roi
            cal_now = {"stop_line": st.session_state.get(f"sl_{cid}", saved.get("stop_line") or sl),
                       "lane_direction": st.session_state.get(f"lane_{cid}", saved.get("lane_direction") or lane),
                       "signal_roi": st.session_state.get(f"roi_{cid}", saved.get("signal_roi") or roi),
                       "no_parking_zones": st.session_state.get(f"npz_{cid}", saved.get("no_parking_zones") or []),
                       "pixels_per_meter": st.session_state.get(f"ppm_{cid}", saved.get("pixels_per_meter") or 0) or None,
                       "speed_limit_kmph": st.session_state.get(f"lim_{cid}", saved.get("speed_limit_kmph") or 40)}
            if frame is not None:
                st.image(_draw(frame, cal_now), use_container_width=True)
                if st.button("Take a fresh frame"):
                    st.session_state[f"nonce_{cid}"] = st.session_state.get(f"nonce_{cid}", 0) + 1
                    st.rerun()
            st.markdown('<div class="small">Image size {}×{} px. A vehicle moving more than 120° against the allowed '
                        'direction is flagged as wrong way. Red-light jumps are flagged only when the lamp inside the '
                        'signal area is read as red.</div>'.format(w, h), unsafe_allow_html=True)
            test = st.session_state.get(f"test_{cid}")
            if test:
                st.markdown(f'<div class="lbl" style="margin-top:10px">Last test · {test["when"]} · {test["frames"]} frames '
                            f'(nothing was saved)</div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:6px">'
                            f'{ui.stat(test["vehicles"], "vehicles tracked")}{ui.stat(test["red"], "red light jumps")}'
                            f'{ui.stat(test["wrong"], "wrong-way movements")}{ui.stat(test.get("seatbelt", 0), "no seatbelt")}'
                            f'{ui.stat(test.get("parking", 0), "illegal parking")}{ui.stat(test.get("unsafe", 0), "unsafe movement")}'
                            f'{ui.stat(test["fps"], "frames/sec")}</div>',
                            unsafe_allow_html=True)

    with right:
        with ui.panel():
            st.markdown(f"### {ui.esc(name)}")
            st.markdown(f'<div class="lbl">Stream address</div><div style="margin-bottom:8px">{ui.esc(_mask(str(url)))}</div>'
                        f'<div class="lbl">Location</div><div style="margin-bottom:8px">{ui.esc(loc)}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="lbl">Rules on this camera</div>', unsafe_allow_html=True)
            rules_saved = saved.get("rules") or {}
            rules = {}
            for k in RULE_KEYS:
                title, hint = RULE_NAMES.get(k, (k.replace('_', ' ').title(), ""))
                rules[k] = st.toggle(title, value=bool(rules_saved.get(k, True)), help=hint or None, key=f"rule_{cid}_{k}")

            st.markdown('<div class="lbl" style="margin-top:10px">Seatbelt CCTV Perspective</div>', unsafe_allow_html=True)
            persp_opts = ["AUTO", "FRONT", "SIDE"]
            persp_saved = (saved.get("perspective") or saved.get("seatbelt_perspective") or "AUTO").upper()
            persp_idx = persp_opts.index(persp_saved) if persp_saved in persp_opts else 0
            persp_val = st.selectbox(
                "Perspective Mode",
                persp_opts,
                index=persp_idx,
                key=f"persp_{cid}",
                help="FRONT: oncoming/overhead CCTV; SIDE: lateral street angle; AUTO: aspect-ratio detection",
            )

            a = b = st.container()
            if a.button("Test stream", use_container_width=True, disabled=frame is None):
                _run_test(cid, src, cal_now, rules)
            if b.button("Save calibration", type="primary", use_container_width=True):
                save_camera_calibration(cid, {
                    **cal_now,
                    "rules": rules,
                    "perspective": persp_val,
                    "seatbelt_perspective": persp_val,
                })
                st.toast(f"Saved calibration for {name} to data/camera_calibration.yaml")
                st.rerun()


def _run_test(cid, src, cal, rules):
    """Run 300 frames through the pipeline without saving anything; show what the rules would catch."""
    from pipeline.traffic_pipeline import TrafficVisionPipeline
    from algorithms.calibration.camera_calibration import config_for_camera

    cfg = load_config_fresh()
    cfg.setdefault("camera_calibration", {})[str(cid)] = {**cal, "rules": rules}
    pipe = TrafficVisionPipeline(config=config_for_camera(cfg, cid), camera_id=cid, location="test", persist=False)
    pipe.calibration = {**cal, "rules": rules}
    cap = cv2.VideoCapture(src)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    bar = st.progress(0.0, text="Testing on the stream…")
    n, t0 = 0, time.time()
    while n < 300:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        pipe.process_frame(frame, n, fps=fps_src, signal_state="AUTO")
        if n % 10 == 0:
            bar.progress(n / 300, text=f"Testing on the stream… {n} frames")
    cap.release()
    bar.empty()
    st.session_state[f"test_{cid}"] = {
        "when": time.strftime("%H:%M"), "frames": n, "vehicles": len(pipe.total_vehicles_seen),
        "red": pipe.event_counts.get("RED_LIGHT_JUMP", 0), "wrong": pipe.event_counts.get("WRONG_WAY", 0),
        "seatbelt": pipe.event_counts.get("NO_SEATBELT", 0),
        "parking": pipe.event_counts.get("ILLEGAL_PARKING", 0),
        "unsafe": pipe.event_counts.get("UNSAFE_MOVEMENT", 0),
        "fps": f"{n / max(0.001, time.time() - t0):.1f}",
    }
    st.rerun()


def render():
    ui.page_header("Camera Setup & Locations",
                   "Map of all CCTV cameras, 1-click GPS location detection, and stop-line calibration")

    cfg = load_config_fresh()
    with session() as db:
        raw_cams = db.query(CameraSource).order_by(CameraSource.id).all()
        status = {c.id: c.status for c in raw_cams}
        cams_tuple = [(c.id, c.name, c.location, c.source_type, c.source_url) for c in raw_cams]

    # Mode Selector
    options = ["All Camera Locations (Map & GPS)", "Camera Calibration & Virtual Lines"]
    current_mode = st.session_state.get("cam_mode", options[0])
    mode = st.radio("Mode", options, index=options.index(current_mode) if current_mode in options else 0,
                    horizontal=True, label_visibility="collapsed")
    st.session_state.cam_mode = mode

    if mode == "All Camera Locations (Map & GPS)":
        _render_all_cameras_map(raw_cams)
    else:
        _render_calibration_view(cams_tuple, status, cfg)
