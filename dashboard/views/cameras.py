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
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">'
        + ui.stat(len(raw_cams), "cameras registered")
        + ui.stat(online_count, "online")
        + ui.stat(len(raw_cams) - online_count, "offline", amber=len(raw_cams) > online_count)
        + ui.stat(len({(c.location or "").split(",")[0] for c in raw_cams}), "junctions")
        + "</div>", unsafe_allow_html=True)
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
    head_l.markdown(f"### Cameras <span class='small'>· {len(shown)} of {len(raw_cams)}</span>", unsafe_allow_html=True)
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
                    f'<span class="tag t-{"go" if is_on else "grey"}">● {c.status or "OFFLINE"}</span>'
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
                    f'<span style="color:#adb5bd;">CAM-{c.id:02d} · {ui.esc(c.source_type or "")}</span>'
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
                    f'GPS: <code>{lat_val:.4f}° N, {lon_val:.4f}° E</code> · Bengaluru'
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


def _render_add_camera():
    """Form to register a new camera with verified Bengaluru junction presets."""
    st.markdown("### Register New CCTV Camera — Bengaluru")
    st.caption("Select a popular Bengaluru junction preset or enter custom stream parameters.")

    presets = {
        "Custom Location": ("", 12.9716, 77.5946),
        "Central Silk Board Junction (Hosur Rd / ORR)": ("Silk Board Junction, Hosur Road, Bengaluru, Karnataka", 12.9176, 77.6238),
        "MG Road - Brigade Road Crossing": ("MG Road - Brigade Rd Crossing, Bengaluru, Karnataka", 12.9754, 77.6067),
        "Hebbal Flyover (Bellary Rd / Airport Corridor)": ("Hebbal Flyover, Bellary Road, Bengaluru, Karnataka", 13.0358, 77.5970),
        "Sony World Junction, Koramangala 80ft Rd": ("80 Feet Rd, Sony World Signal, Koramangala, Bengaluru, Karnataka", 12.9345, 77.6256),
        "Indiranagar 100 Feet Road Junction": ("100 Feet Road, Indiranagar, Bengaluru, Karnataka", 12.9719, 77.6412),
        "Electronic City Toll Junction": ("Hosur Elevated Tollway, Electronic City, Bengaluru, Karnataka", 12.8452, 77.6602),
        "Majestic Kempegowda Interchange": ("Kempegowda Bus Station, Majestic, Bengaluru, Karnataka", 12.9767, 77.5713),
    }

    choice = st.selectbox("1-Click Bengaluru Junction Preset:", list(presets.keys()))
    preset_loc, preset_lat, preset_lon = presets[choice]

    with st.form("add_cam_form", clear_on_submit=False):
        name = st.text_input("Camera Name", placeholder="e.g. Indiranagar 100ft Rd - Cam 1")
        loc_val = st.text_input("Junction Location (Bengaluru, Karnataka)", value=preset_loc or "Bengaluru, Karnataka")
        kind = st.selectbox("Source Type", ["RTSP", "VIDEO_FILE", "WEBCAM"])
        url = st.text_input("Stream Address or Video Path", placeholder="rtsp://user:pass@10.0.0.5:554/stream1 or data/videos/...")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude (°N)", value=preset_lat, format="%.4f")
        lon = c2.number_input("Longitude (°E)", value=preset_lon, format="%.4f")

        submitted = st.form_submit_button("Save & Register Camera", type="primary", use_container_width=True)
        if submitted:
            if not name.strip() or not url.strip():
                st.error("Please enter a camera name and stream address/path.")
            else:
                with session() as db:
                    new_cam = CameraSource(
                        name=name.strip(),
                        location=loc_val.strip(),
                        source_type=kind,
                        source_url=url.strip(),
                        latitude=lat,
                        longitude=lon,
                        status="ONLINE" if kind == "VIDEO_FILE" else "OFFLINE",
                    )
                    db.add(new_cam)
                    db.commit()
                st.toast(f"Camera '{name}' registered in Bengaluru ({lat:.4f}, {lon:.4f})!")
                st.session_state.cam_tab = WALL
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
