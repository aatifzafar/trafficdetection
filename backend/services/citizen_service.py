"""
Citizen proof uploads (dashboard page "Upload proof").

A citizen uploads a video or photo of a traffic violation. Videos and photos are run
through the detection pipeline with our trained violation detector (helmet compliance,
triple riding, and mobile phone usage). Every resulting record is saved with source = CITIZEN
and status PENDING_REVIEW, so nothing happens until an officer checks it in the Review queue.

If the system does not detect the violation the citizen described, the citizen's own
report is saved as a record (with confidence 0.0) so an officer still sees it.
"""

import datetime
import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2

from backend.config.settings import settings
from backend.database.database import SessionLocal, ensure_schema
from backend.db_models.models import Evidence, Violation, Vehicle

logger = logging.getLogger(__name__)

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
    vtype: Optional[str] = None,
    ai_confidence: float = 0.0,
    auto_plate: Optional[str] = None,
    auto_crop: Optional[Any] = None,
    annotated_img: Optional[Any] = None,
    shared_folder: Optional[Path] = None,
    shared_image: Optional[Path] = None,
) -> Violation:
    """The citizen's report, saved with AI detection details when available, or for officer review."""
    event_id = _ref().replace("CIT-", "EVT-CIT-")
    if shared_folder is not None:
        folder = shared_folder
    else:
        folder = Path(settings.EVIDENCE_DIR) / "citizen" / event_id
        folder.mkdir(parents=True, exist_ok=True)

    is_video = upload.suffix.lower() in VIDEO_TYPES
    if shared_image is not None:
        image = shared_image
        clip = folder / f"evidence{upload.suffix.lower()}" if is_video else None
    elif is_video:
        image = _middle_frame(upload, folder / "violation_frame.jpg")
        clip = folder / f"evidence{upload.suffix.lower()}"
        shutil.copy(upload, clip)
    else:
        image = folder / f"violation_frame{upload.suffix.lower()}"
        if annotated_img is not None and getattr(annotated_img, "size", 0) > 0:
            cv2.imwrite(str(image), annotated_img)
        else:
            shutil.copy(upload, image)
        clip = None

    if auto_crop is not None and getattr(auto_crop, "size", 0) > 0:
        p_path = folder / "plate_crop.jpg"
        if not p_path.exists():
            try:
                cv2.imwrite(str(p_path), auto_crop)
            except Exception:
                pass

    resolved_plate = clean_plate(info.get("plate")) or auto_plate
    chosen_type = vtype or info.get("violation_type") or "CITIZEN_REPORT"

    meta = {
        "event_id": event_id,
        "source": "CITIZEN",
        "extra": {
            "citizen_manual": ai_confidence == 0.0,
            "source": "CITIZEN",
            "detected_plate": auto_plate,
            "ai_confidence": round(ai_confidence, 3),
            "claimed_violation": info.get("claimed_violation"),
            "auto_classified": info.get("auto_classified", False),
            "detection_note": info.get("detection_note"),
            "concurrent_violations": info.get("concurrent_violations", []),
            "detected_violations": info.get("detected_violations", []),
        },
        "upload": str(upload),
        "claimed_type": info.get("claimed_violation") or chosen_type,
        "actual_type": chosen_type,
    }
    meta_path = folder / f"metadata_{event_id}.json" if shared_folder else folder / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vehicle_rec = None
    if resolved_plate:
        vehicle_rec = db.query(Vehicle).filter(Vehicle.plate_number == resolved_plate).first()
        if not vehicle_rec:
            vehicle_rec = Vehicle(plate_number=resolved_plate, total_violations=0)
            db.add(vehicle_rec)
            db.flush()

    v = Violation(
        event_id=event_id,
        violation_type=chosen_type,
        vehicle_id=vehicle_rec.id if vehicle_rec else None,
        plate_number=resolved_plate,
        confidence=round(ai_confidence, 3),
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
                    video_path=str(clip) if clip else None, metadata_json=str(meta_path)))
    return v


def process_upload(upload: Path, info: Dict[str, Any],
                   progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Run detection on an uploaded video or photo. Supports multiple simultaneous violations."""
    ensure_schema()
    detected_ids: List[int] = []
    frames = 0
    auto_plate: Optional[str] = None
    auto_crop = None
    photo_ai_conf = 0.0
    photo_annotated = None
    unique_violations: Dict[str, float] = {}

    claimed_viol = info.get("violation_type")
    info["claimed_violation"] = claimed_viol

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
            from algorithms.detection.detector import TrafficDetector
            img = cv2.imread(str(upload))
            if img is not None:
                ocr = PlateRecognizer()
                res = ocr.read_plate_from_full_image(img)
                if res:
                    auto_plate, _, auto_crop = res

                # Run AI detector on photo!
                detector = TrafficDetector(confidence=0.18)
                dets = detector.detect(img)
                
                # Identify all violations and compliance
                violation_candidates = []
                box_draw_list = []
                for d in dets:
                    cname = d.class_name.upper().replace(" ", "_").replace("-", "_")
                    viol_name = None
                    badge_lbl = None
                    is_bad = True
                    
                    if cname in {"MORE_THAN_TWO_PERSONS", "TRIPLE_RIDING", "TRIPLE_RIDER"}:
                        viol_name = "TRIPLE_RIDING"
                        badge_lbl = "TRIPLE RIDING"
                    elif cname in {"WITHOUT_HELMET", "NO_HELMET"}:
                        viol_name = "NO_HELMET"
                        badge_lbl = "NO HELMET"
                    elif cname in {"WITH_HELMET", "HELMET"}:
                        badge_lbl = "HELMET"
                        is_bad = False
                    elif cname in {"USING_MOBILE", "MOBILE_PHONE"}:
                        viol_name = "MOBILE_PHONE_WHILE_DRIVING"
                        badge_lbl = "PHONE USAGE"

                    if badge_lbl:
                        box_draw_list.append((d.bbox, badge_lbl, d.confidence, is_bad))
                    if viol_name:
                        violation_candidates.append({
                            "type": viol_name,
                            "confidence": d.confidence,
                            "bbox": d.bbox,
                            "class": cname,
                        })

                # Draw bounding boxes on photo
                if box_draw_list:
                    ann = img.copy()
                    for box, lbl, c, is_bad in box_draw_list:
                        x1, y1, x2, y2 = [int(p) for p in box]
                        color = (0, 0, 255) if is_bad else (0, 200, 0)
                        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
                        text = f"{lbl} {int(c * 100)}%"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(ann, (x1, max(0, y1 - 22)), (x1 + tw + 6, y1), color, -1)
                        cv2.putText(ann, text, (x1 + 3, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    photo_annotated = ann

                # Group detected violations by type to capture ALL simultaneous offenses
                for cand in violation_candidates:
                    t = cand["type"]
                    unique_violations[t] = max(unique_violations.get(t, 0.0), cand["confidence"])

                if unique_violations:
                    photo_ai_conf = max(unique_violations.values())
        except Exception as e:
            logger.warning(f"Error analyzing photo upload: {e}")

    db = SessionLocal()
    try:
        records = db.query(Violation).filter(Violation.id.in_(detected_ids)).all() if detected_ids else []
        plate = clean_plate(info.get("plate")) or auto_plate
        for r in records:
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

        # If records were detected by the video pipeline, use them!
        # If photo was uploaded: Create a record for EACH distinct violation found by AI!
        manual_records = []
        if not records:
            if unique_violations:
                base_event = _ref().replace("CIT-", "EVT-CIT-")
                shared_folder = Path(settings.EVIDENCE_DIR) / "citizen" / base_event
                shared_folder.mkdir(parents=True, exist_ok=True)
                shared_img = shared_folder / f"violation_frame{upload.suffix.lower()}"
                if photo_annotated is not None and getattr(photo_annotated, "size", 0) > 0:
                    cv2.imwrite(str(shared_img), photo_annotated)
                else:
                    shutil.copy(upload, shared_img)

                all_types = list(unique_violations.keys())
                for v_type, conf in unique_violations.items():
                    info_copy = dict(info)
                    info_copy["violation_type"] = v_type
                    info_copy["auto_classified"] = True
                    info_copy["concurrent_violations"] = all_types
                    other_v = [x for x in all_types if x != v_type]
                    if other_v:
                        info_copy["detection_note"] = f"Concurrent violation: Also detected {', '.join(other_v)} on this vehicle."
                    else:
                        info_copy["detection_note"] = f"AI verified '{v_type}' with {round(conf * 100)}% accuracy."

                    rec = _manual_record(
                        db,
                        upload,
                        info_copy,
                        vtype=v_type,
                        ai_confidence=conf,
                        auto_plate=auto_plate,
                        auto_crop=auto_crop,
                        shared_folder=shared_folder,
                        shared_image=shared_img,
                    )
                    manual_records.append(rec)
            else:
                info_copy = dict(info)
                fallback_type = claimed_viol if claimed_viol not in {None, "AUTO_DETECT"} else "CITIZEN_REPORT"
                info_copy["violation_type"] = fallback_type
                info_copy["auto_classified"] = False
                rec = _manual_record(
                    db,
                    upload,
                    info_copy,
                    vtype=fallback_type,
                    ai_confidence=0.0,
                    auto_plate=auto_plate,
                    auto_crop=auto_crop,
                    annotated_img=photo_annotated,
                )
                manual_records.append(rec)

        db.commit()

        out = []
        for r in records:
            out.append({
                "id": r.id,
                "event_id": r.event_id,
                "type": r.violation_type,
                "plate": r.plate_number,
                "auto": True,
                "confidence": r.confidence,
                "claimed": claimed_viol,
                "auto_classified": True,
            })
        for m in manual_records:
            out.append({
                "id": m.id,
                "event_id": m.event_id,
                "type": m.violation_type,
                "plate": m.plate_number,
                "auto": m.confidence > 0.0,
                "confidence": m.confidence,
                "claimed": claimed_viol,
                "auto_classified": m.confidence > 0.0,
            })

        total_detected = len(records) + (len(unique_violations) if unique_violations else (1 if photo_ai_conf > 0 else 0))
        return {
            "frames": frames,
            "records": out,
            "detected": total_detected,
            "ai_detected": photo_ai_conf > 0.0 or len(records) > 0,
            "all_violations": list(unique_violations.keys()),
        }
    finally:
        db.close()
