import os
import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_cfg = load_config()


class Settings:
    _cfg: Dict[str, Any] = _cfg
    CONFIG_RAW: Dict[str, Any] = _cfg
    APP_NAME: str = _cfg.get("app", {}).get("name", "Traffic Violation System")
    DEMO_MODE: bool = os.getenv("DEMO_MODE", str(_cfg.get("app", {}).get("demo_mode", True))).lower() in ("true", "1", "yes")
    DEVICE: str = _cfg.get("app", {}).get("device", "auto")
    DATABASE_URL: str = os.getenv("DATABASE_URL", _cfg.get("database", {}).get("url", "sqlite:///./data/traffic_system.db"))
    
    # Models
    MODEL_TRAFFIC: str = _cfg.get("detection", {}).get("model_traffic", "models/yolov8n.pt")
    MODEL_HELMET: str = _cfg.get("detection", {}).get("model_helmet", "models/helmet_model.pt")
    MODEL_PLATE: str = _cfg.get("detection", {}).get("model_plate", "models/plate_model.pt")
    CONFIDENCE: float = float(_cfg.get("detection", {}).get("confidence", 0.35))
    IOU: float = float(_cfg.get("detection", {}).get("iou", 0.45))
    
    # Tracker
    TRACKER_ALGO: str = _cfg.get("tracking", {}).get("algorithm", "bytetrack")
    
    # Association
    SPATIAL_IOU: float = float(_cfg.get("association", {}).get("spatial_iou_threshold", 0.15))
    MAX_RIDER_DISTANCE: float = float(_cfg.get("association", {}).get("max_rider_bike_distance", 120))
    
    # Evidence
    PRE_SECONDS: int = int(_cfg.get("evidence", {}).get("pre_seconds", 4))
    POST_SECONDS: int = int(_cfg.get("evidence", {}).get("post_seconds", 4))
    EVIDENCE_DIR: str = _cfg.get("evidence", {}).get("save_dir", "data/evidence")
    
    # OCR
    OCR_ENGINE: str = _cfg.get("ocr", {}).get("engine", "easyocr")
    PLATE_REGEX: str = _cfg.get("ocr", {}).get("plate_regex", "^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
    
    # Violations enabled
    VIOLATIONS_CONFIG: Dict[str, bool] = _cfg.get("violations", {
        "helmet": True,
        "triple_riding": True,
        "red_light": True,
        "wrong_way": True,
        "seatbelt": True,
        "mobile_phone": True,
    })
    
    # Temporal confirmation
    TEMPORAL_WINDOW: int = int(_cfg.get("temporal_confirmation", {}).get("window", 10))
    MIN_POSITIVE_FRAMES: int = int(_cfg.get("temporal_confirmation", {}).get("minimum_positive_frames", 6))
    COOLDOWN_SECONDS: int = int(_cfg.get("temporal_confirmation", {}).get("violation_cooldown_seconds", 10))


settings = Settings()
