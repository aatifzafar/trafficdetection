"""
FastAPI Router for Seatbelt Violation Detection & Model Status.
Endpoint style strictly matches existing backend/api routes.
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/seatbelt", tags=["Seatbelt Enforcement"])

MODEL_PATHS = [
    PROJECT_ROOT / "models" / "seatbelt_best.pt",
    PROJECT_ROOT / "models" / "seatbelt_model.pt",
]

_loaded_seatbelt_model = None


def get_seatbelt_model():
    global _loaded_seatbelt_model
    if _loaded_seatbelt_model is not None:
        return _loaded_seatbelt_model

    for p in MODEL_PATHS:
        if p.exists():
            try:
                from ultralytics import YOLO
                _loaded_seatbelt_model = YOLO(str(p))
                logger.info(f"Loaded seatbelt model from {p}")
                return _loaded_seatbelt_model
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
    return None


class SeatbeltDetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class SeatbeltInferenceResponse(BaseModel):
    status: str
    experimental: bool = True
    model_configured: bool
    has_violation: bool
    detections: List[SeatbeltDetectionItem]
    measured_metrics: Optional[Dict[str, Any]] = None
    detail: Optional[str] = None


@router.get("/metrics")
def get_seatbelt_metrics():
    """Returns measured precision, recall, mAP and limitations from evaluation report."""
    metrics_path = PROJECT_ROOT / "data" / "evaluation_results_seatbelt.json"
    if not metrics_path.exists():
        return {
            "status": "uncalibrated",
            "experimental": True,
            "message": "Evaluation JSON data/evaluation_results_seatbelt.json not found. Run scripts/evaluate_seatbelt.py after training.",
        }
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        data["experimental"] = True
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse evaluation metrics: {e}")


@router.post("/detect", response_model=SeatbeltInferenceResponse)
async def detect_seatbelt(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.35, ge=0.05, le=1.0),
):
    """
    Runs seatbelt detection on an uploaded windshield or front-cabin vehicle crop.
    Gracefully returns standby if the seatbelt model is not configured.
    """
    model = get_seatbelt_model()
    metrics_summary = None

    metrics_path = PROJECT_ROOT / "data" / "evaluation_results_seatbelt.json"
    if metrics_path.exists():
        try:
            m_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_summary = {
                "val_precision": m_data.get("overall", {}).get("precision"),
                "val_recall": m_data.get("overall", {}).get("recall"),
                "no_seatbelt_precision": m_data.get("per_class", {}).get("no_seatbelt", {}).get("precision"),
                "no_seatbelt_recall": m_data.get("per_class", {}).get("no_seatbelt", {}).get("recall"),
            }
        except Exception:
            pass

    if model is None:
        return SeatbeltInferenceResponse(
            status="standby",
            experimental=True,
            model_configured=False,
            has_violation=False,
            detections=[],
            measured_metrics=metrics_summary,
            detail="Seatbelt detection model not configured. Operational standby.",
        )

    # Read uploaded image
    try:
        content = await file.read()
        im = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    results = model(im, conf=confidence_threshold, verbose=False)
    detections: List[SeatbeltDetectionItem] = []
    has_violation = False

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_idx = int(box.cls[0].item())
            cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
            conf = float(box.conf[0].item())
            xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]

            if cls_idx == 1 or "no" in cls_name.lower():
                has_violation = True

            detections.append(
                SeatbeltDetectionItem(
                    class_id=cls_idx,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=xyxy,
                )
            )

    return SeatbeltInferenceResponse(
        status="processed",
        experimental=True,
        model_configured=True,
        has_violation=has_violation,
        detections=detections,
        measured_metrics=metrics_summary,
        detail="Detection completed successfully.",
    )

