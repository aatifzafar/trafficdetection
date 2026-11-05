"""
Catalogue of the vehicle/person detection models the dashboard can compare and put live.

Review 1 planned YOLOv8; the project moved to YOLO26 (released January 2026). The Models page lets the
team compare any of these on the same video and choose which one runs live.

Published numbers: Ultralytics model pages, COCO val2017, 640 px (checked 2 Nov 2026).
GPU times were measured by Ultralytics on different GPUs for different families (T4 or A100), so compare
GPU times only within the same hardware. YOLOv10 has no published CPU time.
Weights (.pt) are downloaded automatically by Ultralytics into models/ the first time a model is used.
"""

from pathlib import Path
from typing import Dict, List, Optional

SOURCES = {
    "YOLO26": "https://docs.ultralytics.com/models/yolo26/",
    "YOLO11": "https://docs.ultralytics.com/models/yolo11/",
    "YOLOv10": "https://docs.ultralytics.com/models/yolov10/",
    "YOLOv8": "https://docs.ultralytics.com/models/yolov8/",
    "YOLOv5": "https://docs.ultralytics.com/models/yolov5/",
}
SOURCE_URL = "https://docs.ultralytics.com/compare/yolov8-vs-yolo26/"

FAMILIES: Dict[str, Dict] = {
    "YOLO26": {"year": 2026, "note": "Project model. End-to-end and NMS-free, no DFL head, ProgLoss + STAL for small "
                                     "objects, MuSGD optimizer; built for CPU and edge devices."},
    "YOLO11": {"year": 2024, "note": "Successor of YOLOv8 with a lighter backbone (C3k2, C2PSA attention); "
                                     "needs NMS."},
    "YOLOv10": {"year": 2024, "note": "First NMS-free YOLO (Tsinghua University, dual label assignment); "
                                      "no published CPU time."},
    "YOLOv8": {"year": 2023, "note": "Review 1 plan. Anchor-free head with DFL; needs NMS; very widely used."},
    "YOLOv5": {"year": 2020, "note": "Older baseline (Ultralytics 'u' version with an anchor-free head); needs NMS."},
}

# key -> family, size, published {mAP50_95, cpu_onnx_ms, gpu_ms, gpu_hw, params_m, flops_b}
_TABLE = [
    ("yolo26n", "YOLO26", "n", 40.9, 38.9, 1.7, "T4", 2.4, 5.5),
    ("yolo26s", "YOLO26", "s", 48.6, 87.2, 2.5, "T4", 9.5, 20.9),
    ("yolo26m", "YOLO26", "m", 53.1, 220.0, 4.7, "T4", 20.4, 68.4),
    ("yolo11n", "YOLO11", "n", 39.5, 56.1, 1.5, "T4", 2.6, 6.5),
    ("yolo11s", "YOLO11", "s", 47.0, 90.0, 2.5, "T4", 9.4, 21.6),
    ("yolo11m", "YOLO11", "m", 51.5, 183.2, 4.7, "T4", 20.1, 68.1),
    ("yolov10n", "YOLOv10", "n", 38.5, None, 1.84, "T4", 2.3, 6.7),
    ("yolov10s", "YOLOv10", "s", 46.3, None, 2.49, "T4", 7.2, 21.6),
    ("yolov8n", "YOLOv8", "n", 37.3, 80.4, 0.99, "A100", 3.2, 8.7),
    ("yolov8s", "YOLOv8", "s", 44.9, 128.4, 1.20, "A100", 11.2, 28.6),
    ("yolov8m", "YOLOv8", "m", 50.2, 234.7, 1.83, "A100", 25.9, 78.9),
    ("yolov5nu", "YOLOv5", "nu", 34.3, 73.6, 1.06, "A100", 2.6, 7.7),
    ("yolov5su", "YOLOv5", "su", 43.0, 120.7, 1.27, "A100", 9.1, 24.0),
]
SIZE_NAMES = {"n": "nano", "s": "small", "m": "medium", "nu": "nano", "su": "small"}

MODELS: Dict[str, Dict] = {}
for key, fam, size, m, cpu, gpu, hw, params, flops in _TABLE:
    short_name = f"{fam}{size}" if fam != "YOLOv10" else f"YOLOv10{size}"
    MODELS[key] = {
        "path": f"models/{key}.pt",
        "family": fam,
        "size": SIZE_NAMES[size],
        "year": FAMILIES[fam]["year"],
        "short": short_name,
        "label": f"{short_name} ({FAMILIES[fam]['year']}, {SIZE_NAMES[size]})",
        "published": {"mAP50_95": m, "cpu_onnx_ms": cpu, "gpu_ms": gpu, "gpu_hw": hw, "params_m": params, "flops_b": flops},
        "note": FAMILIES[fam]["note"],
        "source": SOURCES[fam],
    }
MODELS["yolo26n"]["label"] = "YOLO26n (2026, nano) — project model"
MODELS["yolov8n"]["label"] = "YOLOv8n (2023, nano) — Review 1 plan"


def key_for_path(path: str) -> str:
    return Path(str(path)).stem.lower()


def available() -> List[str]:
    """Every catalogue model. A missing .pt file is downloaded by Ultralytics the first time it is used."""
    return list(MODELS)


def is_downloaded(key: str) -> bool:
    return key in MODELS and Path(MODELS[key]["path"]).exists()


def label(key_or_path: str) -> str:
    k = key_for_path(key_or_path)
    return MODELS[k]["label"] if k in MODELS else Path(str(key_or_path)).name


def short(key_or_path: str) -> str:
    k = key_for_path(key_or_path)
    if "yolo26n_v2_832" in k:
        return "YOLO26n run at 832 px (same model)"
    if "seatbelt" in k:
        return "YOLO26n (Seatbelt)"
    if "wrong_way_yolo" in k:
        return "YOLO (Wrong Way)"
    if "wrong_way" in k:
        return "Bi-LSTM (Wrong Way)"
    return MODELS[k]["short"] if k in MODELS else Path(str(key_or_path)).stem


def path(key: str) -> str:
    return MODELS[key]["path"] if key in MODELS else key


def download(key: str) -> Optional[str]:
    """Fetch the weights into models/ (needs internet). Returns an error message, or None when ready."""
    if is_downloaded(key):
        return None
    try:
        from ultralytics import YOLO
        Path("models").mkdir(exist_ok=True)
        YOLO(path(key))
        return None if is_downloaded(key) else "download did not create the file"
    except Exception as e:  # no internet, blocked site, etc.
        return str(e)
