"""
Per-camera calibration (stop line, lane direction, signal-light area, rule switches, perspective)
and small helpers the pipeline uses with it.

Calibration is written by the dashboard's Camera setup page to data/camera_calibration.yaml
(config.yaml is never rewritten, so its comments stay). An optional `camera_calibration:` block
in config.yaml is still read; the file written by the dashboard wins.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from backend.config.settings import CONFIG_PATH

RULE_KEYS = ["helmet", "triple_riding", "red_light", "wrong_way", "seatbelt"]
CALIBRATION_PATH = Path("data/camera_calibration.yaml")
APP_SETTINGS_PATH = Path("data/app_settings.yaml")   # written by the dashboard's Settings page
TRACKERS = ["bytetrack", "deepsort"]


def _load_calibration_file() -> Dict[str, Any]:
    if not CALIBRATION_PATH.exists():
        return {}
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_app_settings() -> Dict[str, Any]:
    """Settings saved from the dashboard (detection model, confidence, tracker, default stop line)."""
    if not APP_SETTINGS_PATH.exists():
        return {}
    with open(APP_SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_app_settings(values: Dict[str, Any]) -> None:
    current = load_app_settings()
    current.update({k: _clean(v) for k, v in values.items()})
    APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(APP_SETTINGS_PATH, "w", encoding="utf-8") as f:
        f.write("# Written by the dashboard's Settings page. Overrides config.yaml for these keys.\n")
        yaml.safe_dump(current, f, sort_keys=False, allow_unicode=True)


def _apply_app_settings(cfg: Dict[str, Any]) -> None:
    app = load_app_settings()
    if "confidence" in app:
        cfg.setdefault("detection", {})["confidence"] = float(app["confidence"])
    if app.get("model_traffic"):
        cfg.setdefault("detection", {})["model_traffic"] = app["model_traffic"]
    if app.get("tracker") in TRACKERS:
        cfg.setdefault("tracking", {})["algorithm"] = app["tracker"]
    if app.get("default_stop_line"):
        cfg.setdefault("cameras", {})["default_stop_line"] = app["default_stop_line"]


def load_config_fresh() -> Dict[str, Any]:
    """Read config.yaml from disk (settings.CONFIG_RAW is only read once at import),
    with the dashboard's per-camera calibration merged in."""
    cfg: Dict[str, Any] = {}
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cal = dict(cfg.get("camera_calibration") or {})
    cal.update({str(k): v for k, v in _load_calibration_file().items()})
    cfg["camera_calibration"] = cal
    _apply_app_settings(cfg)
    return cfg


def get_camera_calibration(cfg: Dict[str, Any], camera_id: Optional[int]) -> Dict[str, Any]:
    if camera_id is None:
        return {}
    cal = (cfg.get("camera_calibration") or {})
    return dict(cal.get(str(camera_id)) or cal.get(camera_id) or {})


def _clean(v):
    if isinstance(v, float):
        return round(v, 3) if abs(v) < 10 else int(round(v))
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def save_camera_calibration(camera_id: int, calibration: Dict[str, Any]) -> None:
    """Write one camera's calibration to data/camera_calibration.yaml (config.yaml is left untouched)."""
    cal = _load_calibration_file()
    cal[str(camera_id)] = _clean(calibration)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        f.write("# Written by the dashboard's Camera setup page. Keys are camera ids.\n")
        yaml.safe_dump(cal, f, sort_keys=False, allow_unicode=True)


def config_for_camera(base_cfg: Dict[str, Any], camera_id: Optional[int]) -> Dict[str, Any]:
    """Return a copy of the config with this camera's calibration applied to the rule settings."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (base_cfg or {}).items()}
    cal = get_camera_calibration(cfg, camera_id)
    if not cal:
        return cfg
    cams = dict(cfg.get("cameras") or {})
    if cal.get("stop_line"):
        cams["default_stop_line"] = cal["stop_line"]
    if cal.get("lane_direction"):
        cams["default_lane_direction"] = cal["lane_direction"]
    if cal.get("perspective"):
        cams["default_perspective"] = cal["perspective"]
    if cal.get("seatbelt_perspective"):
        cams["seatbelt_perspective"] = cal["seatbelt_perspective"]
    cfg["cameras"] = cams
    rules = cal.get("rules") or {}
    if rules:
        viol = dict(cfg.get("violations") or {})
        for k in RULE_KEYS:
            if k in rules:
                viol[k] = bool(rules[k])
        cfg["violations"] = viol
    return cfg


def signal_state_from_roi(frame: np.ndarray, roi: Optional[List[float]]) -> str:
    """Classify the traffic light inside the calibrated signal area as RED / GREEN / UNKNOWN."""
    if frame is None or not roi or len(roi) != 4:
        return "UNKNOWN"
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN"
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 2] > 150) & (hsv[..., 1] > 90)  # lit lamp pixels only
    hue = hsv[..., 0]
    red = int(np.count_nonzero(bright & ((hue <= 10) | (hue >= 170))))
    green = int(np.count_nonzero(bright & (hue >= 40) & (hue <= 90)))
    if max(red, green) < 15:
        return "UNKNOWN"
    return "RED" if red > green else "GREEN"


def light_condition(frame: np.ndarray) -> str:
    """'night' when the scene is dark (mean brightness below ~70/255), else 'day'."""
    if frame is None or frame.size == 0:
        return "day"
    small = cv2.resize(frame, (64, 36))
    v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].mean()
    return "night" if v < 70 else "day"
