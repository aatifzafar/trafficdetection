"""Settings: detection sensitivity, tracker and the default stop line — the old Settings page, kept.

Saving writes data/app_settings.yaml; every new pipeline run (Live cameras, run_video.py, API uploads) reads it.
Per-camera stop lines from Camera setup still take priority over the default stop line.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from algorithms.calibration.camera_calibration import TRACKERS, load_app_settings, load_config_fresh, save_app_settings
from algorithms.detection import model_registry as mr
from backend.config.settings import settings
from dashboard import ui


def _preview(x1, y1, x2, y2):
    img = np.full((720, 1280, 3), 45, dtype=np.uint8)
    for ly in range(0, 720, 60):
        cv2.line(img, (640, ly), (640, ly + 35), (255, 255, 255), 4)
        cv2.line(img, (320, ly), (320, ly + 35), (200, 200, 200), 2)
        cv2.line(img, (960, ly), (960, ly + 35), (200, 200, 200), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
    cv2.putText(img, f"STOP LINE ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})",
                (max(10, int(x1)), max(25, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render():
    ui.page_header("Settings", "Detection sensitivity, tracker and the default stop line")
    cfg = load_config_fresh()
    saved = load_app_settings()
    conf0 = float((cfg.get("detection") or {}).get("confidence", settings.CONFIDENCE))
    trk0 = (cfg.get("tracking") or {}).get("algorithm", settings.TRACKER_ALGO)
    model0 = mr.key_for_path((cfg.get("detection") or {}).get("model_traffic", settings.MODEL_TRAFFIC))
    sl0 = (cfg.get("cameras") or {}).get("default_stop_line") or {"x1": 150, "y1": 520, "x2": 1100, "y2": 520}

    left, right = st.columns([1, 1.25], gap="medium")
    with left:
        with ui.panel():
            st.markdown("### Detection model and sensitivity")
            keys = mr.available()
            model_key = st.selectbox("Vehicle / person detection model", keys,
                                     index=keys.index(model0) if model0 in keys else 0, format_func=mr.label,
                                     help="YOLO26n is the project model. YOLOv8n was the Review 1 plan; keep it to compare. "
                                          "The Evaluation page compares both on the same video.")
            conf = st.slider("Confidence threshold", 0.10, 0.90, round(conf0, 2), 0.05,
                             help="Lower finds more objects but adds false boxes. 0.35 is the project default.")
            trk = st.selectbox("Tracking algorithm", TRACKERS, index=TRACKERS.index(trk0) if trk0 in TRACKERS else 0,
                               format_func=lambda t: {"bytetrack": "ByteTrack (recommended)", "deepsort": "DeepSORT"}[t])
            st.markdown(f'<dl class="kv"><dt>Plate reader</dt><dd>{ui.esc(settings.OCR_ENGINE)}</dd>'
                        f'<dt>Device</dt><dd>{ui.esc(settings.DEVICE)}</dd>'
                        f'<dt>Model file</dt><dd>{ui.esc(mr.path(model_key))}'
                        f'{"" if mr.is_downloaded(model_key) else " (downloads on first use)"}</dd>'
                        f'<dt>Helmet / plate model</dt><dd>{ui.esc(Path(settings.MODEL_CUSTOM).name)}'
                        f'{"" if Path(settings.MODEL_CUSTOM).exists() else " (not trained yet)"}</dd></dl>',
                        unsafe_allow_html=True)
        with ui.panel():
            st.markdown("### Default stop line")
            st.markdown('<div class="small">Used by cameras that have no stop line of their own. '
                        'Set per-camera lines on the Camera setup page.</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            x1 = c1.number_input("Stop line X1", value=float(sl0["x1"]), step=10.0)
            y1 = c1.number_input("Stop line Y1", value=float(sl0["y1"]), step=10.0)
            x2 = c2.number_input("Stop line X2", value=float(sl0["x2"]), step=10.0)
            y2 = c2.number_input("Stop line Y2", value=float(sl0["y2"]), step=10.0)
        if st.button("Save settings", type="primary", use_container_width=True):
            save_app_settings({"model_traffic": mr.path(model_key), "confidence": conf, "tracker": trk,
                               "default_stop_line": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
            st.success("Saved to data/app_settings.yaml. New runs on Live cameras use these settings.")
        if saved:
            st.markdown('<div class="small">Currently saved overrides: '
                        + ", ".join(f"{ui.esc(k)}" for k in saved) + "</div>", unsafe_allow_html=True)
    with right:
        with ui.panel():
            st.markdown("### Stop line preview")
            st.image(_preview(x1, y1, x2, y2), caption="Default stop line on a sample 1280×720 frame",
                     use_container_width=True)
