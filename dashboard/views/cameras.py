"""
Unified Cameras Command Center — Bengaluru, Karnataka.
Consolidates all camera operations into a single master hub:
  1. All Cameras Video Wall (Small screen preview cards for each camera with Setup & Location buttons)
  2. Live Camera Feeds (Realtime AI detection, multi-camera grid monitoring)
  3. Virtual Lines & Calibration (Stop-line, allowed lane direction, signal ROI, rule toggles)
  4. Register Camera (Bengaluru junction presets & stream registration)
"""

import datetime
import os
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.calibration.camera_calibration import load_config_fresh
from backend.db_models.models import CameraSource
from backend.services import geo_service
from dashboard import ui
from dashboard.data import session
from dashboard.views import cameras_setup, live

TABS = [
    "Video wall",
    "Live detection",
    "Calibration",
    "Register camera",
]
WALL, LIVE, CALIB, REGISTER = TABS
PAGE_SIZE = 9


@st.cache_data(show_spinner=False, ttl=300, max_entries=64)
def _thumb(src, max_w: int = 480):
    """Small snapshot for the wall (cached 5 min); full-size frames make the page slow to send."""
    frame = cameras_setup._grab_frame(src)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _render_camera_cards_wall(raw_cams):
    """
    Video Wall Grid of Cards:
    Each camera is rendered as a sleek CCTV monitor card with:
      - Live moving CCTV video footage (or snapshot toggle)
      - CCTV HUD overlay with timestamp & Bengaluru camera ID
      - Verified Bengaluru physical junction & GPS coordinates
      - 1-click action buttons:
          * Location (Shows verified Bengaluru physical location, coordinates & Google Maps)
          * Setup (Configures stop-line, allowed lane direction, and rules)
          * Run AI (Runs realtime AI detection on this camera)
    """
    if not raw_cams:
        ui.empty("No cameras registered", "Add a camera using the 'Register Camera' tab.")
        return

    cams_by_id = {c.id: c for c in raw_cams}
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary + toolbar
    online_count = sum(1 for c in raw_cams if c.status == "ONLINE")
    offline = len(raw_cams) - online_count
    st.markdown(ui.readout([
        (len(raw_cams), "Cameras registered"),
        (online_count, "Online"),
        (offline, "Offline", None, "amb" if offline else ""),
        (len({(c.location or "").split(",")[0] for c in raw_cams}), "Junctions covered"),
    ]), unsafe_allow_html=True)
    with st.container(key="wall_toolbar"):
        t1, t2, t3, t4, t5 = st.columns([2.2, 1.2, 1.5, 0.9, 0.9], vertical_alignment="bottom")
        query = t1.text_input("Search", placeholder="Camera name or junction", key="wall_q")
        status_f = t2.selectbox("Status", ["All", "Online", "Offline"], key="wall_status")
        feed_mode = t3.radio("Preview", ["Snapshots", "Live video"], horizontal=True, key="cctv_feed_mode",
                             help="Live video plays every camera on this page at once; snapshots load much faster.")
        if t4.button("City map", use_container_width=True):
            st.session_state.show_city_map = not st.session_state.get("show_city_map", False)
            st.rerun()
        if t5.button("Add camera", use_container_width=True):
            st.session_state.cam_tab = REGISTER
            st.rerun()

    # =========================================================================
    # 1. EXPANDED PHYSICAL LOCATION DOSSIER (TRIGGERED BY 1-CLICK LOCATION)
    # =========================================================================
    active_loc_id = st.session_state.get("active_location_cam")
    if active_loc_id and active_loc_id in cams_by_id:
        target_cam = cams_by_id[active_loc_id]
        dossier = geo_service.get_camera_original_location(target_cam)
        with ui.panel():
            header_col, close_col = st.columns([4, 1], vertical_alignment="center")
            with header_col:
                st.markdown(
                    f'<div style="font-size:18px;font-weight:700;color:#1F4E8C;">'
                    f'Original Physical Location: {ui.esc(dossier["name"])}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with close_col:
                if st.button("Close Location", key="close_loc_dossier", use_container_width=True):
                    st.session_state.active_location_cam = None
                    st.rerun()

            doss_left, doss_right = st.columns([1.5, 1.5], gap="medium")
            with doss_left:
                st.markdown(
                    f'<table class="data" style="margin:0;">'
                    f'<tr><td style="width:140px;font-weight:600;color:var(--muted);">City & State</td>'
                    f'<td><b>Bengaluru, Karnataka (India)</b></td></tr>'
                    f'<tr><td style="font-weight:600;color:var(--muted);">Physical Junction</td>'
                    f'<td><b>{ui.esc(dossier["location"])}</b></td></tr>'
                    f'<tr><td style="font-weight:600;color:var(--muted);">Landmark</td>'
                    f'<td>{ui.esc(dossier["landmark"])}</td></tr>'
                    f'<tr><td style="font-weight:600;color:var(--muted);">Traffic Division</td>'
                    f'<td>{ui.esc(dossier["zone"])} (PIN {dossier["pin_code"]})</td></tr>'
                    f'<tr><td style="font-weight:600;color:var(--muted);">GPS Coordinates</td>'
                    f'<td><code>{dossier["latitude"]:.4f}° N, {dossier["longitude"]:.4f}° E</code></td></tr>'
                    f'<tr><td style="font-weight:600;color:var(--muted);">Stream Source</td>'
                    f'<td>{target_cam.source_type} · <code>{ui.esc(cameras_setup._mask(str(target_cam.source_url)))}</code></td></tr>'
                    f'</table>',
                    unsafe_allow_html=True,
                )

                # Action Links
                act1, act2 = st.columns(2)
                with act1:
                    st.markdown(
                        f'<a href="{dossier["google_maps_url"]}" target="_blank" style="text-decoration:none;">'
                        f'<button style="width:100%;height:38px;background:#1F4E8C;color:#fff;border:none;'
                        f'border-radius:6px;font-weight:600;font-size:12px;cursor:pointer;">Open in Google Maps</button></a>',
                        unsafe_allow_html=True,
                    )
                with act2:
                    if st.button("Configure Setup for this Cam", key=f"doss_setup_{target_cam.id}", use_container_width=True):
                        st.session_state.cam_sel = target_cam.id
                        st.session_state.cam_tab = CALIB
                        st.rerun()

            with doss_right:
                # Small map centered on this camera in Bengaluru
                ScatterClass = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
                if ScatterClass:
                    fig_mini = go.Figure(ScatterClass(
                        lat=[dossier["latitude"]],
                        lon=[dossier["longitude"]],
                        mode="markers+text",
                        text=[dossier["name"]],
                        textposition="top center",
                        textfont=dict(size=12, color="#1F4E8C", family="Barlow, sans-serif"),
                        marker=dict(size=18, color="#C62828"),
                    ))
                    map_cfg = dict(
                        style="open-street-map",
                        center=dict(lat=dossier["latitude"], lon=dossier["longitude"]),
                        zoom=13.5,
                    )
                    if hasattr(go, "Scattermap"):
                        fig_mini.update_layout(map=map_cfg, height=220, margin=dict(l=0, r=0, t=0, b=0))
                    else:
                        fig_mini.update_layout(mapbox=map_cfg, height=220, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_mini, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # 2. FULL BENGALURU MAP (OPTIONAL TOGGLE)
    # =========================================================================
    if st.session_state.get("show_city_map", False):
        with ui.panel():
            st.markdown("### Bengaluru City-Wide CCTV Camera Map")
            df_all = pd.DataFrame([
                {
                    "id": c.id,
                    "name": c.name,
                    "location": c.location,
                    "lat": float(c.latitude or 12.9716),
                    "lon": float(c.longitude or 77.5946),
                    "status": c.status or "OFFLINE",
                }
                for c in raw_cams
            ])
            colors = ["#2E7D4F" if s == "ONLINE" else "#8A949E" for s in df_all["status"]]
            hover = (
                "<b>" + df_all["name"] + "</b><br>"
                + df_all["location"] + "<br>"
                + "GPS: " + df_all["lat"].round(4).astype(str) + "° N, " + df_all["lon"].round(4).astype(str) + "° E"
            )

            ScatterClass = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
            if ScatterClass:
                fig_city = go.Figure(ScatterClass(
                    lat=df_all["lat"],
                    lon=df_all["lon"],
                    mode="markers+text",
                    text=df_all["name"].apply(lambda n: n.split(" - ")[0][:16]),
                    textposition="top center",
                    textfont=dict(size=11, color="#1F4E8C", family="Barlow, sans-serif"),
                    hoverinfo="text",
                    hovertext=hover,
                    marker=dict(size=14, color=colors, opacity=0.9),
                ))
                city_cfg = dict(style="open-street-map", center=dict(lat=12.9716, lon=77.5946), zoom=11.3)
                if hasattr(go, "Scattermap"):
                    fig_city.update_layout(map=city_cfg, height=360, margin=dict(l=0, r=0, t=0, b=0))
                else:
                    fig_city.update_layout(mapbox=city_cfg, height=360, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # 3. VIDEO WALL GRID OF CARDS (EACH WITH SMALL SCREEN + BUTTONS)
    # =========================================================================
    q = (query or "").strip().lower()
    shown = [c for c in raw_cams
             if (not q or q in (c.name or "").lower() or q in (c.location or "").lower())
             and (status_f == "All" or (c.status == "ONLINE") == (status_f == "Online"))]
    pages = max(1, -(-len(shown) // PAGE_SIZE))
    page = min(int(st.session_state.get("wall_page", 1)), pages)
    head_l, head_r = st.columns([3, 2], vertical_alignment="center")
    head_l.markdown(f"<div class='small' style='padding-top:10px'>Showing {len(shown)} of {len(raw_cams)} cameras</div>",
                    unsafe_allow_html=True)
    if pages > 1:
        with head_r:
            p1, p2, p3 = st.columns([1, 1.2, 1], vertical_alignment="center")
            if p1.button("Previous", disabled=page <= 1, use_container_width=True, key="wall_prev"):
                st.session_state.wall_page = page - 1
                st.rerun()
            p2.markdown(f"<div class='small' style='text-align:center'>Page {page} of {pages}</div>",
                        unsafe_allow_html=True)
            if p3.button("Next", disabled=page >= pages, use_container_width=True, key="wall_next"):
                st.session_state.wall_page = page + 1
                st.rerun()
    if not shown:
        ui.empty("No cameras match", "Change the search or status filter.")
        return
    num_cols = 3
    cols = st.columns(num_cols, gap="medium")

    for idx, c in enumerate(shown[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]):
        col = cols[idx % num_cols]
        with col:
            with ui.panel():
                is_on = c.status == "ONLINE"
                is_active = c.id == active_loc_id

                # Card Header
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<b style="font-size:14px;color:#18212B;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'{ui.esc(c.name)}</b>'
                    f'<span class="tag t-{"go" if is_on else "grey"}">{"Online" if is_on else "Offline"}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # SMALL SCREEN: CCTV Live Footage / Video Screen
                src = cameras_setup._source(c)
                is_video_file = c.source_type == "VIDEO_FILE" and isinstance(src, str) and os.path.exists(src)

                play_video = feed_mode == "Live video" and is_video_file
                thumb = None if play_video else _thumb(src)
                has_feed = play_video or thumb is not None
                signal = ('<span style="color:#e03131;font-weight:700;">● LIVE</span>' if play_video else
                          '<span style="color:#51cf66;font-weight:700;">● SNAPSHOT</span>' if has_feed else
                          '<span style="color:#adb5bd;font-weight:700;">○ NO SIGNAL</span>')
                # CCTV Screen Header Bezel
                st.markdown(
                    f'<div style="background:#11161d;padding:4px 8px;border-top-left-radius:6px;border-top-right-radius:6px;'
                    f'display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #232b36;'
                    f'font-family:monospace;font-size:11px;color:#f8f9fa;">'
                    f'<span>{signal}</span>'
                    f'<span style="color:#adb5bd;">CAM-{c.id:02d} {ui.esc(KIND_LABEL.get(c.source_type, c.source_type or ""))}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if play_video:
                    st.video(src, format="video/mp4", loop=True, autoplay=True, muted=True)
                else:
                    if thumb is not None:
                        st.image(thumb, use_container_width=True, output_format="JPEG")
                    else:
                        st.markdown(
                            f'<div style="background:#2A3038;height:160px;display:grid;'
                            f'place-items:center;color:#AAB4BE;font-size:13px;text-align:center;">'
                            f'<div><b>No signal</b><br><span style="font-size:11px;">'
                            f'{"Stream address is a placeholder" if cameras_setup._is_placeholder(src) else "Source could not be opened"}'
                            f'</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                loc_short = (c.location or "Junction").split(",")[0]
                lat_val = float(c.latitude or 12.9716)
                lon_val = float(c.longitude or 77.5946)

                # CCTV Screen Footer Bezel
                st.markdown(
                    f'<div style="background:#11161d;padding:3px 8px;border-bottom-left-radius:6px;border-bottom-right-radius:6px;'
                    f'font-family:monospace;font-size:10px;color:#9098a1;display:flex;justify-content:space-between;margin-bottom:8px;">'
                    f'<span>{ui.esc(loc_short)}</span>'
                    f'<span>{now_str} IST</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Physical Location Details in Bengaluru
                st.markdown(
                    f'<div class="small" style="color:var(--muted);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'<b>{ui.esc(c.location or "—")}</b>'
                    f'</div>'
                    f'<div class="small" style="color:var(--muted);margin-bottom:10px;">'
                    f'{lat_val:.4f}° N, {lon_val:.4f}° E'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # BUTTON TOOLBAR: Location, Setup, Live
                btn1, btn2, btn3 = st.columns(3)
                # Button 1: Location
                if btn1.button("Location", key=f"wall_loc_{c.id}", help="View verified Bengaluru location & Google Maps", use_container_width=True):
                    st.session_state.active_location_cam = c.id
                    st.rerun()

                # Button 2: Setup
                if btn2.button("Setup", key=f"wall_setup_{c.id}", help="Configure virtual stop-line & lane direction", use_container_width=True):
                    st.session_state.cam_sel = c.id
                    st.session_state.cam_tab = CALIB
                    st.rerun()

                # Button 3: run AI detection
                if btn3.button("Run AI", key=f"wall_live_{c.id}", help="Run realtime AI detection pipeline on this feed", use_container_width=True):
                    st.session_state.live_sel = [c.id]
                    st.session_state.cam_tab = LIVE
                    st.rerun()


SOURCE_KINDS = {
    "Mobile phone (IP Webcam app)": "MOBILE",
    "CCTV / IP camera (RTSP)": "RTSP",
    "USB or laptop webcam": "WEBCAM",
    "Recorded video file": "VIDEO_FILE",
}
HANDHELD_KINDS = {"MOBILE", "WEBCAM"}
KIND_LABEL = {"VIDEO_FILE": "VIDEO", "MOBILE": "PHONE", "RTSP": "RTSP", "WEBCAM": "WEBCAM"}
HANDHELD_RULES = {"helmet": True, "triple_riding": True, "seatbelt": True, "red_light": False, "wrong_way": False,
                  "illegal_parking": False, "unsafe_movement": False}
JUNCTIONS = {
    "Custom location": ("", 12.9716, 77.5946),
    "Central Silk Board Junction (Hosur Rd / ORR)": ("Silk Board Junction, Hosur Road, Bengaluru, Karnataka", 12.9176, 77.6238),
    "MG Road - Brigade Road Crossing": ("MG Road - Brigade Rd Crossing, Bengaluru, Karnataka", 12.9754, 77.6067),
    "Hebbal Flyover (Bellary Rd / Airport Corridor)": ("Hebbal Flyover, Bellary Road, Bengaluru, Karnataka", 13.0358, 77.5970),
    "Sony World Junction, Koramangala 80ft Rd": ("80 Feet Rd, Sony World Signal, Koramangala, Bengaluru, Karnataka", 12.9345, 77.6256),
    "Indiranagar 100 Feet Road Junction": ("100 Feet Road, Indiranagar, Bengaluru, Karnataka", 12.9719, 77.6412),
    "Electronic City Toll Junction": ("Hosur Elevated Tollway, Electronic City, Bengaluru, Karnataka", 12.8452, 77.6602),
    "Majestic Kempegowda Interchange": ("Kempegowda Bus Station, Majestic, Bengaluru, Karnataka", 12.9767, 77.5713),
    "Alliance University, Anekal": ("Alliance University, Chikkahagade Cross, Anekal, Bengaluru, Karnataka", 12.7286, 77.7071),
}


def _phone_url(addr: str) -> str:
    """Turn what the IP Webcam app shows (e.g. 192.168.43.1:8080) into its video stream address."""
    a = addr.strip().rstrip("/")
    if not a:
        return ""
    if not a.startswith(("http://", "https://", "rtsp://")):
        a = "http://" + a
    host = a.split("://", 1)[1]
    if ":" not in host.split("/")[0]:
        a = a.replace(host.split("/")[0], host.split("/")[0] + ":8080", 1)
    if not a.split("://", 1)[1].count("/"):
        a += "/video"
    return a


def _probe(src, timeout_s: float = 8.0):
    """Open the source and read one frame, giving up after timeout_s so a wrong address never freezes the page.
    Returns (frame or None, message)."""
    import threading
    out = {}

    def work():
        try:
            if isinstance(src, str) and "://" in src:
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_s * 1000),
                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_s * 1000)])
            else:
                cap = cv2.VideoCapture(src)
            out["opened"] = cap.isOpened()
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            out["fps"] = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
            cap.release()
            out["frame"] = frame if ok else None
        except Exception as e:  # noqa: BLE001
            out["err"] = str(e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s + 2)
    if t.is_alive():
        return None, "No reply from the camera. Check the address and that both devices are on the same network."
    if out.get("err"):
        return None, f"Could not open the source: {out['err']}"
    f = out.get("frame")
    if f is None and not out.get("opened"):
        return None, ("Could not connect. Check the address, that the camera app is running, and that the laptop "
                      "is on the same network (the phone's hotspot).")
    if f is None:
        return None, "Connected, but no picture arrived. Check the address (a phone stream ends with /video)."
    h, w = f.shape[:2]
    fps = out.get("fps") or 0
    return f, f"Connected · {w}×{h}" + (f" · {fps:.0f} FPS" if 0 < fps < 120 else "")


def _render_add_camera():
    """Register a camera: pick the kind of source, test the connection, place it on the map, save."""
    ss = st.session_state
    st.markdown("### Register a camera")
    st.markdown('<div class="small">Choose the kind of camera, test that the picture comes through, then save. '
                'The camera then appears on the Video wall and can run live AI detection.</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("**1. Camera source**")
        kind_label = st.radio("Kind of camera", list(SOURCE_KINDS), key="reg_kind", horizontal=False,
                              label_visibility="collapsed")
        kind = SOURCE_KINDS[kind_label]
        src = None
        if kind == "MOBILE":
            st.markdown('<div class="note-blue">On the phone: install <b>IP Webcam</b> (Android), set video '
                        'resolution to 640×480 or 1280×720 and tap <b>Start server</b>. Connect this laptop to the '
                        "phone's hotspot (college Wi-Fi usually blocks phone-to-laptop streams). Type the address "
                        'shown at the bottom of the phone screen.</div>', unsafe_allow_html=True)
            addr = st.text_input("Address shown on the phone", placeholder="192.168.43.1:8080", key="reg_phone")
            src = _phone_url(addr) if addr.strip() else None
            if src:
                st.caption(f"Stream address: {src}")
        elif kind == "RTSP":
            src = st.text_input("Stream address", placeholder="rtsp://user:password@10.0.0.5:554/stream1",
                                key="reg_rtsp").strip() or None
            st.caption("From the camera's or NVR's settings. HTTP/MJPEG addresses (http://…) also work.")
        elif kind == "WEBCAM":
            idx = st.number_input("Device number", min_value=0, max_value=9, value=0, step=1, key="reg_idx",
                                  help="0 is the laptop's built-in camera. A phone connected with DroidCam or "
                                       "Iriun Webcam is usually 1.")
            src = int(idx)
            st.caption("0 = built-in camera · 1 = second camera (USB camera, DroidCam or Iriun on a phone)")
        else:
            vids = sorted(str(p).replace("\\", "/") for p in Path("data/videos").glob("*.mp4"))
            pick = st.selectbox("Video in data/videos", vids + ["Other path…"], key="reg_vid") if vids else "Other path…"
            src = (st.text_input("Video path", placeholder="data/videos/my_clip.mp4", key="reg_vpath").strip() or None) \
                if pick == "Other path…" else pick

        tested = ss.get("reg_test")
        if st.button("Test connection", key="reg_test_btn", disabled=src is None, use_container_width=True):
            with st.spinner("Connecting…"):
                frame, msg = _probe(src)
            ss.reg_test = {"src": src, "ok": frame is not None, "msg": msg,
                           "jpg": cv2.imencode(".jpg", cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1]))))[1].tobytes()
                           if frame is not None else None}
            tested = ss.reg_test
        if tested and tested.get("src") == src:
            if tested["ok"]:
                st.markdown(f'<div class="note-blue"><b>{ui.esc(tested["msg"])}</b></div>', unsafe_allow_html=True)
                st.image(tested["jpg"], caption="Picture from the camera", width=360)
            else:
                st.markdown(f'<div class="note-amber"><b>Not connected.</b> {ui.esc(tested["msg"])}</div>',
                            unsafe_allow_html=True)

    with right:
        st.markdown("**2. Name and place**")
        default_name = {"MOBILE": "Demo - Mobile camera", "WEBCAM": "Demo - Webcam"}.get(kind, "")
        name = st.text_input("Camera name", value=default_name, placeholder="e.g. Indiranagar 100ft Rd - Cam 1",
                             key=f"reg_name_{kind}")
        junction = st.selectbox("Junction", list(JUNCTIONS), key="reg_junction",
                                help="Sets the address and map position. Pick Custom location to type your own.")
        j_loc, j_lat, j_lon = JUNCTIONS[junction]
        loc = st.text_input("Location", value=j_loc or "Bengaluru, Karnataka", key=f"reg_loc_{junction}")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude (°N)", value=j_lat, format="%.4f", key=f"reg_lat_{junction}")
        lon = c2.number_input("Longitude (°E)", value=j_lon, format="%.4f", key=f"reg_lon_{junction}")

        with session() as db:
            taken = {n.lower() for (n,) in db.query(CameraSource.name).all()}
        problem = None
        if not name.strip():
            problem = "Enter a camera name."
        elif name.strip().lower() in taken:
            problem = "A camera with this name already exists. Use a different name."
        elif src is None or src == "":
            problem = "Enter the camera address first."
        ok_conn = bool(tested and tested.get("src") == src and tested.get("ok"))
        if kind in HANDHELD_KINDS:
            st.caption("Phone and webcam cameras start with helmet, triple-riding and seatbelt checks only. Red light, "
                       "wrong way, parking and unsafe movement need a fixed camera with a drawn stop line or lane; "
                       "switch them on in Calibration once the camera is mounted still.")
        if not problem and not ok_conn:
            st.caption("Tip: test the connection first. You can still save a camera that is not reachable now.")
        if problem:
            st.caption(problem)

        st.markdown("**3. Save**")
        b1, b2 = st.columns(2)
        save = b1.button("Save camera", type="primary", disabled=problem is not None, use_container_width=True,
                         key="reg_save")
        save_run = b2.button("Save and open live detection", disabled=problem is not None, use_container_width=True,
                             key="reg_save_run")
        if save or save_run:
            with session() as db:
                cam = CameraSource(name=name.strip(), location=loc.strip() or "Bengaluru, Karnataka",
                                   source_type=kind, source_url=str(src), latitude=lat, longitude=lon,
                                   status="ONLINE" if (kind == "VIDEO_FILE" or ok_conn) else "OFFLINE")
                db.add(cam)
                db.commit()
                new_id = cam.id
            if kind in HANDHELD_KINDS:
                # a phone or webcam has no fixed stop line, lane or parking zone: keep only the rules that work
                # on any picture; they can be switched on later in Calibration once the camera is mounted still
                cameras_setup.save_camera_calibration(new_id, {"rules": dict(HANDHELD_RULES)})
            for k in ("reg_test", "reg_phone", "reg_rtsp"):
                ss.pop(k, None)
            st.toast(f"Camera '{name.strip()}' registered")
            if save_run:
                ss.live_sel = [new_id]
                ss.cam_tab = LIVE
            else:
                ss.cam_tab = WALL
            st.rerun()


def render():
    ui.page_header("Cameras", "Every junction camera, live AI detection, calibration and registration")

    current_tab = st.session_state.get("cam_tab", TABS[0])
    if current_tab not in TABS:
        current_tab = TABS[0]
    # buttons elsewhere set cam_tab; carry it into the radio before the radio is drawn
    if st.session_state.get("cam_tab_radio") != current_tab:
        st.session_state["cam_tab_radio"] = current_tab

    with st.container(key="cam_nav_chips"):
        selected_tab = st.radio(
            "Camera View",
            TABS,
            horizontal=True,
            label_visibility="collapsed",
            key="cam_tab_radio",
            on_change=lambda: st.session_state.update(cam_tab=st.session_state.cam_tab_radio),
        )
    st.session_state.cam_tab = selected_tab

    cfg = load_config_fresh()
    with session() as db:
        raw_cams = db.query(CameraSource).order_by(CameraSource.id).all()
        status = {c.id: c.status for c in raw_cams}
        cams_tuple = [(c.id, c.name, c.location, c.source_type, c.source_url) for c in raw_cams]

    # Sub-views
    if selected_tab == WALL:
        _render_camera_cards_wall(raw_cams)
    elif selected_tab == LIVE:
        live.render()
    elif selected_tab == CALIB:
        cameras_setup._render_calibration_view(cams_tuple, status, cfg)
    elif selected_tab == REGISTER:
        with ui.panel():
            _render_add_camera()
