"""Database helpers shared by the dashboard views."""

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

from backend.database.database import SessionLocal
from backend.db_models.models import CameraSource, Evidence, Violation


@contextmanager
def session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def camera_names(db) -> Dict[int, str]:
    return {c.id: c.name for c in db.query(CameraSource).all()}


def where(viol: Violation, cams: Dict[int, str]) -> str:
    if viol.source == "CITIZEN":
        return f"Citizen upload · {viol.location}"
    if viol.camera_id and viol.camera_id in cams:
        return cams[viol.camera_id]
    return viol.location or "—"


def evidence_for(db, viol: Violation) -> Dict[str, Any]:
    ev = db.query(Evidence).filter(Evidence.violation_id == viol.id).first()
    image = (ev.image_path if ev else None) or viol.evidence_path
    video = ev.video_path if ev else None
    meta: Dict[str, Any] = {}
    if ev and ev.metadata_json:
        try:
            if os.path.exists(ev.metadata_json):
                with open(ev.metadata_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = json.loads(ev.metadata_json)
        except Exception:
            meta = {}
    return {
        "image": image if image and os.path.exists(image) else None,
        "video": video if video and os.path.exists(video) else None,
        "meta": meta,
    }


def plate_crop(image_path: Optional[str]):
    """Return the saved plate crop next to the evidence image, if the pipeline stored one."""
    if not image_path:
        return None
    p = os.path.join(os.path.dirname(image_path), "plate_crop.jpg")
    return p if os.path.exists(p) else None
