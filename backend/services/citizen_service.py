"""
Citizen proof uploads (dashboard page "Upload proof").

A citizen uploads a video or photo of a traffic violation. Videos are run through the same detection
pipeline as the cameras (only the rider rules — helmet and triple riding — because red-light and
wrong-way need a calibrated camera). Every resulting record is saved with source = CITIZEN and status
PENDING_REVIEW, so nothing happens until an officer checks it in the Review queue.

If the system does not detect the violation the citizen described (or the upload is a photo), the
citizen's own report is saved as a record so an officer still sees it.
"""

import datetime
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2

from backend.config.settings import settings
from backend.database.database import SessionLocal, ensure_schema
from backend.db_models.models import Evidence, Violation, Vehicle

UPLOAD_DIR = Path("data/citizen_uploads")
VIDEO_TYPES = {".mp4", ".avi", ".mkv", ".mov"}
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
MAX_FRAMES = 900  # about 30 s of video at 30 fps; longer clips are cut to keep processing quick on a CPU
PLATE_RE = re.compile(r"[^A-Z0-9]")


def clean_plate(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    p = PLATE_RE.sub("", text.upper())
    return p or None


def mask_phone(phone: Optional[str]) -> str:
    if not phone:
        return "—"
    digits = re.sub(r"\D", "", phone)
    return f"••••••{digits[-4:]}" if len(digits) >= 4 else "••••"


def _ref() -> str:
    return f"CIT-{datetime.datetime.utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4].upper()}"


def save_upload(name: str, data: bytes) -> Path:
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_TYPES | PHOTO_TYPES:
        raise ValueError("Please upload a video (MP4, AVI, MKV, MOV) or a photo (JPG, PNG).")
    folder = UPLOAD_DIR / _ref()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"upload{ext}"
    path.write_bytes(data)
    return path


def _middle_frame(video: Path, out: Path) -> Optional[Path]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    cv2.imwrite(str(out), frame)
    return out


def _manual_record(
    db,
    upload: Path,
    info: Dict[str, Any],
    auto_plate: Optional[str] = None,
    auto_crop: Optional[Any] = None,
) -> Violation:
    """The citizen's own report, saved when the system did not detect what they described."""
    event_id = _ref().replace("CIT-", "EVT-CIT-")
    folder = Path(settings.EVIDENCE_DIR) / "citizen" / event_id
    folder.mkdir(parents=True, exist_ok=True)
    is_video = upload.suffix.lower() in VIDEO_TYPES
    if is_video:
        image = _middle_frame(upload, folder / "violation_frame.jpg")
        clip = folder / f"evidence{upload.suffix.lower()}"
        shutil.copy(upload, clip)
    else:
        image = folder / f"violation_frame{upload.suffix.lower()}"
        shutil.copy(upload, image)
        clip = None

    if auto_crop is not None and getattr(auto_crop, "size", 0) > 0:
        try:
            cv2.imwrite(str(folder / "plate_crop.jpg"), auto_crop)
        except Exception:
            pass

    resolved_plate = clean_plate(info.get("plate")) or auto_plate
    meta = {
        "event_id": event_id,
        "source": "CITIZEN",
        "extra": {
            "citizen_manual": True,
            "source": "CITIZEN",
            "detected_plate": auto_plate,
        },
        "upload": str(upload),
        "claimed_type": info.get("violation_type"),
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vehicle_rec = None
    if resolved_plate:
        vehicle_rec = db.query(Vehicle).filter(Vehicle.plate_number == resolved_plate).first()
        if not vehicle_rec:
            vehicle_rec = Vehicle(plate_number=resolved_plate, total_violations=0)
            db.add(vehicle_rec)
            db.flush()

    v = Violation(
        event_id=event_id,
        violation_type=info.get("violation_type") or "CITIZEN_REPORT",
        vehicle_id=vehicle_rec.id if vehicle_rec else None,
        plate_number=resolved_plate,
        confidence=0.0,
        timestamp=info.get("when_utc") or datetime.datetime.utcnow(),
        location=info.get("location") or "—",
        camera_id=info.get("camera_id"),
        evidence_path=str(image) if image else None,
        status="PENDING_REVIEW",
        source="CITIZEN",
        merged_count=0,
        reporter_name=info.get("name") or None,
        reporter_phone=info.get("phone") or None,
        report_note=info.get("note") or None,
    )
    db.add(v)
    db.flush()
    db.add(Evidence(event_id=event_id, violation_id=v.id, image_path=str(image) if image else None,
                    video_path=str(clip) if clip else None, metadata_json=str(folder / "metadata.json")))
    return v


def process_upload(upload: Path, info: Dict[str, Any],
                   progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Run detection on an uploaded video (or photo). Returns a summary."""
    ensure_schema()
    detected_ids: List[int] = []
    frames = 0
    auto_plate: Optional[str] = None
    auto_crop = None

    if upload.suffix.lower() in VIDEO_TYPES:
        from algorithms.calibration.camera_calibration import load_config_fresh
        from pipeline.traffic_pipeline import TrafficVisionPipeline

        cfg = load_config_fresh()
        viol = dict(cfg.get("violations") or {})
        viol.update({"red_light": False, "wrong_way": False})  # need a calibrated camera
        cfg["violations"] = viol
        pipe = TrafficVisionPipeline(config=cfg, camera_id=info.get("camera_id"),
                                     location=info.get("location") or "Citizen upload", source="CITIZEN")
        cap = cv2.VideoCapture(str(upload))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = min(total, MAX_FRAMES) if total else MAX_FRAMES
        try:
            while frames < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                pipe.process_frame(frame, frames, fps=fps, signal_state="GREEN")
                if progress and frames % 5 == 0:
                    progress(frames / max(1, limit), f"Checking frame {frames} of {limit}…")
        finally:
            cap.release()
            pipe.evidence_recorder.flush()
        detected_ids = sorted(pipe._created_ids)
        auto_plate, auto_crop = pipe.get_best_plate_and_crop()

    elif upload.suffix.lower() in PHOTO_TYPES:
        try:
            from algorithms.plate_ocr.plate_recognizer import PlateRecognizer
            img = cv2.imread(str(upload))
            if img is not None:
                ocr = PlateRecognizer()
                res = ocr.read_plate_from_full_image(img)
                if res:
                    auto_plate, _, auto_crop = res
        except Exception:
            pass

    db = SessionLocal()
    try:
        records = db.query(Violation).filter(Violation.id.in_(detected_ids)).all() if detected_ids else []
        plate = clean_plate(info.get("plate")) or auto_plate
        for r in records:  # attach the citizen's details to what the system found
            r.reporter_name = info.get("name") or None
            r.reporter_phone = info.get("phone") or None
            r.report_note = info.get("note") or None
            if info.get("when_utc"):
                r.timestamp = info["when_utc"]
            if plate and not r.plate_number:
                r.plate_number = plate
                vehicle_rec = db.query(Vehicle).filter(Vehicle.plate_number == plate).first()
                if not vehicle_rec:
                    vehicle_rec = Vehicle(plate_number=plate, total_violations=0)
                    db.add(vehicle_rec)
                    db.flush()
                r.vehicle_id = vehicle_rec.id

            if auto_crop is not None and getattr(auto_crop, "size", 0) > 0 and r.evidence_path:
                folder = Path(r.evidence_path).parent
                p_path = folder / "plate_crop.jpg"
                if not p_path.exists():
                    try:
                        cv2.imwrite(str(p_path), auto_crop)
                    except Exception:
                        pass

        claimed = info.get("violation_type")
        found_types = {r.violation_type for r in records}
        manual = None
        if not records or (claimed and claimed != "CITIZEN_REPORT" and claimed not in found_types):
            manual = _manual_record(db, upload, info, auto_plate=auto_plate, auto_crop=auto_crop)
        db.commit()
        out = [{"id": r.id, "event_id": r.event_id, "type": r.violation_type, "plate": r.plate_number,
                "auto": True, "confidence": r.confidence} for r in records]
        if manual is not None:
            out.append({"id": manual.id, "event_id": manual.event_id, "type": manual.violation_type,
                        "plate": manual.plate_number, "auto": False, "confidence": 0.0})
        return {"frames": frames, "records": out, "detected": len(records)}
    finally:
        db.close()
