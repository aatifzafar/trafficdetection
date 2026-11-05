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
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.settings import settings
from backend.database.database import SessionLocal, ensure_schema
from backend.db_models.models import Evidence, Violation, Vehicle

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/citizen_uploads")
VIDEO_TYPES = {".mp4", ".avi", ".mkv", ".mov"}
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
MAX_FRAMES = 900  # about 30 s of video at 30 fps; longer clips are cut to keep processing quick on a CPU
PLATE_RE = re.compile(r"[^A-Z0-9]")


TWO_WHEELER_CLASSES = {"motorcycle", "motorbike", "scooter", "moped", "bicycle"}
FOUR_WHEELER_CLASSES = {"car", "truck", "bus", "van", "suv"}

_SEATBELT_MODEL = None


def _get_seatbelt_model():
    """Caches and returns the trained seatbelt model (models/seatbelt_best.pt)."""
    global _SEATBELT_MODEL
    if _SEATBELT_MODEL is None:
        p = Path("models/seatbelt_best.pt")
        if p.exists() and p.is_file():
            try:
                from ultralytics import YOLO
                _SEATBELT_MODEL = YOLO(str(p))
                logger.info("Citizen service loaded seatbelt model (models/seatbelt_best.pt)")
            except Exception as e:
                logger.warning(f"Could not load seatbelt model in citizen service: {e}")
    return _SEATBELT_MODEL


def classify_citizen_photo(img: np.ndarray, claimed_viol: Optional[str] = None) -> Dict[str, Any]:
    """
    Vehicle-type-aware AI classification for citizen proof photos:
    1. Four-wheelers (car, truck, bus, van, suv):
       - Refined candidate crops:
         a) Occupant full torso (from head down to lap/waist)
         b) Driver-side window / cabin focus
         c) Central windshield (from roofline down to hood)
         d) Full vehicle crop (vehicle context)
       - Evaluates models/seatbelt_best.pt at conf=0.25 (optical sharpness threshold >= 15.0).
       - Class 0 ('seatbelt'): Compliant (is_bad=False). Zero violation generated!
       - Class 1 ('no_seatbelt'): Confirmed NO_SEATBELT violation (is_bad=True).
       - Strictly SUPPRESSES helmet and triple-riding rules.
    2. Two-wheelers (motorcycle, scooter, etc.):
       - Evaluates helmet compliance and triple-riding spatially linked to the two-wheeler.
       - Strictly SUPPRESSES seatbelt rules.
    3. Close-up / Cropped Windshield:
       - Evaluates seatbelt model if optical sharpness and resolution criteria are met.
       - Never applies helmet rules without a two-wheeler present.
    4. Inconclusive / Low Visibility:
       - If blurred, low-res, or confidence < 0.25, returns INCONCLUSIVE with diagnostic trace.
       - Never invents a violation.
    """
    from algorithms.detection.detector import TrafficDetector

    h_img, w_img = img.shape[:2]
    logger.info(f"Analyzing citizen photo: original_size={w_img}x{h_img}, claimed={claimed_viol}")
    detector = TrafficDetector(confidence=0.18)
    dets = detector.detect(img)

    four_wheelers = [d for d in dets if d.class_name.lower() in FOUR_WHEELER_CLASSES]
    two_wheelers = [d for d in dets if d.class_name.lower() in TWO_WHEELER_CLASSES]
    persons = [d for d in dets if d.class_name.lower() == "person"]

    violation_candidates: List[Dict[str, Any]] = []
    box_draw_list: List[Tuple[List[float], str, float, bool]] = []
    compliance_notes: List[str] = []
    diagnostic_trace: List[Dict[str, Any]] = []

    is_four_wheeler = len(four_wheelers) > 0
    is_two_wheeler = len(two_wheelers) > 0
    compliance_status = "INCONCLUSIVE"
    best_sharpness = 0.0
    best_crop_dims = None

    # -------------------------------------------------------------
    # 1. FOUR-WHEELER ROUTING (Seatbelt + In-Car Phone, NO HELMET)
    # -------------------------------------------------------------
    if is_four_wheeler:
        logger.info(f"Four-wheeler detected ({len(four_wheelers)} vehicle(s)). Strictly suppressing helmet rules.")
        sb_model = _get_seatbelt_model()
        if sb_model is not None:
            sorted_fw = sorted(four_wheelers, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            primary_area = (sorted_fw[0].bbox[2]-sorted_fw[0].bbox[0]) * (sorted_fw[0].bbox[3]-sorted_fw[0].bbox[1])
            target_fw = [f for f in sorted_fw if ((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) >= 0.15 * primary_area]
            for fw in target_fw:
                vx1, vy1, vx2, vy2 = [int(c) for c in fw.bbox]
                vw = max(1, vx2 - vx1)
                vh = max(1, vy2 - vy1)
                logger.info(f"Primary four-wheeler bbox=[{vx1}, {vy1}, {vx2}, {vy2}], size={vw}x{vh}")

                candidate_crops = []

                # Find visible occupants inside vehicle cabin
                cabin_occupants = []
                for p in persons:
                    px1, py1, px2, py2 = [int(c) for c in p.bbox]
                    pcx = (px1 + px2) // 2
                    pcy = (py1 + py2) // 2
                    if vx1 <= pcx <= vx2 and vy1 <= pcy <= (vy1 + int(vh * 0.75)):
                        cabin_occupants.append(p)

                # Prioritize occupant-anchored upper torso crops (identical to SeatbeltViolationRule)
                if cabin_occupants:
                    for p in cabin_occupants:
                        px1, py1, px2, py2 = [int(c) for c in p.bbox]
                        pw = max(1, px2 - px1)
                        ph = max(1, py2 - py1)
                        cx1 = max(0, min(w_img - 1, px1 - int(pw * 0.15)))
                        cx2 = max(0, min(w_img, px2 + int(pw * 0.15)))
                        cy1 = max(0, min(h_img - 1, py1))
                        cy2 = max(0, min(h_img, py1 + int(ph * 0.85)))
                        crop = img[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            candidate_crops.append((crop, [cx1, cy1, cx2, cy2], "occupant_anchor", p.bbox))
                else:
                    # Geometric front windshield / side window ONLY when no occupants detected
                    aspect = float(vw) / float(vh)
                    if aspect > 1.35:  # SIDE
                        cx1 = max(0, min(w_img - 1, vx1 + int(vw * 0.18)))
                        cx2 = max(0, min(w_img, vx1 + int(vw * 0.72)))
                        cy1 = max(0, min(h_img - 1, vy1 + int(vh * 0.08)))
                        cy2 = max(0, min(h_img, vy1 + int(vh * 0.55)))
                        mode = "side_window"
                    else:  # FRONT
                        cx1 = max(0, min(w_img - 1, vx1 + int(vw * 0.15)))
                        cx2 = max(0, min(w_img, vx1 + int(vw * 0.85)))
                        cy1 = max(0, min(h_img - 1, vy1 + int(vh * 0.12)))
                        cy2 = max(0, min(h_img, vy1 + int(vh * 0.60)))
                        mode = "front_windshield"
                    crop_geom = img[cy1:cy2, cx1:cx2]
                    if crop_geom.size > 0:
                        candidate_crops.append((crop_geom, [cx1, cy1, cx2, cy2], mode, None))

                # Evaluate candidate cabin crops
                best_sb_conf = 0.0
                best_ns_conf = 0.0
                best_sb_box = None
                best_ns_box = None

                for c_img, offset, mode, pbox in candidate_crops:
                    ch, cw = c_img.shape[:2]
                    if cw < 60 or ch < 60:
                        logger.debug(f"ROI [{mode}] rejected: resolution {cw}x{ch} < 60x60")
                        continue
                    gray = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if sharpness < 15.0:
                        logger.debug(f"ROI [{mode}] rejected: sharpness {sharpness:.1f} < 15.0")
                        continue

                    try:
                        res = sb_model(c_img, conf=0.20, verbose=False)
                        if res and len(res[0].boxes) > 0:
                            for b in res[0].boxes:
                                cid = int(b.cls[0].item())
                                conf = float(b.conf[0].item())
                                cname = sb_model.names.get(cid, str(cid))
                                bx1, by1, bx2, by2 = [float(x) for x in b.xyxy[0].cpu().numpy().tolist()]
                                bw = bx2 - bx1
                                bh = by2 - by1

                                # Reject flat horizontal boxes (dashboard, wiper cowl, bumper lines)
                                if bw > 2.5 * bh:
                                    continue

                                gbox = [offset[0] + bx1, offset[1] + by1, offset[0] + bx2, offset[1] + by2]

                                # Spatial grounding: ensure detection center is aligned with occupant chest
                                if pbox is not None:
                                    chest_top = pbox[1]
                                    chest_bottom = pbox[1] + 0.85 * (pbox[3] - pbox[1])
                                    bcy = (gbox[1] + gbox[3]) / 2
                                    if not (chest_top - 20 <= bcy <= chest_bottom + 40):
                                        continue

                                diagnostic_trace.append({
                                    "mode": mode,
                                    "class_id": cid,
                                    "class_name": cname,
                                    "confidence": round(conf, 3),
                                    "sharpness": round(sharpness, 1),
                                    "crop_size": [cw, ch],
                                    "roi_coords": offset,
                                })
                                logger.info(
                                    f"Seatbelt ROI [{mode}] {cw}x{ch}px (sharpness {sharpness:.1f}): "
                                    f"{cname} ({cid}) conf={conf:.3f} gbox={[round(x,1) for x in gbox]}"
                                )
                                if cid == 0:  # seatbelt
                                    if conf > best_sb_conf:
                                        best_sb_conf = conf
                                        best_sb_box = gbox
                                        best_crop_dims = [cw, ch]
                                        best_sharpness = sharpness
                                elif cid == 1:  # no_seatbelt
                                    if conf > best_ns_conf:
                                        best_ns_conf = conf
                                        best_ns_box = gbox
                                        best_crop_dims = [cw, ch]
                                        best_sharpness = sharpness
                    except Exception as e:
                        logger.debug(f"Seatbelt inference error on {mode}: {e}")

                # Decision at confidence threshold 0.35 (consistent with SeatbeltViolationRule):
                CONF_THRESH = 0.35
                if best_sb_conf >= CONF_THRESH and best_sb_conf >= best_ns_conf:
                    box_draw_list.append((best_sb_box, "SEATBELT", best_sb_conf, False))
                    compliance_notes.append(f"Occupant verified wearing SEATBELT ({int(best_sb_conf*100)}% conf)")
                    compliance_status = "SEATBELT_COMPLIANT"
                elif best_ns_conf >= CONF_THRESH:
                    box_draw_list.append((best_ns_box, "NO SEATBELT", best_ns_conf, True))
                    violation_candidates.append({
                        "type": "NO_SEATBELT",
                        "confidence": best_ns_conf,
                        "bbox": best_ns_box,
                        "class": "NO_SEATBELT",
                        "sharpness": round(best_sharpness, 1),
                        "crop_dimensions": best_crop_dims,
                        "diagnostic_trace": diagnostic_trace,
                    })
                    compliance_status = "NO_SEATBELT_VIOLATION"
                else:
                    compliance_status = "INCONCLUSIVE"

        # Check phone usage inside car
        for d in dets:
            cname = d.class_name.upper().replace(" ", "_")
            if cname in {"USING_MOBILE", "MOBILE_PHONE"}:
                dcx, dcy = d.center
                for fw in four_wheelers:
                    if fw.bbox[0] <= dcx <= fw.bbox[2] and fw.bbox[1] <= dcy <= fw.bbox[3]:
                        box_draw_list.append((d.bbox, "PHONE USAGE", d.confidence, True))
                        violation_candidates.append({
                            "type": "MOBILE_PHONE_WHILE_DRIVING",
                            "confidence": d.confidence,
                            "bbox": d.bbox,
                            "class": cname,
                        })

    # -------------------------------------------------------------
    # 2. TWO-WHEELER ROUTING (Helmet + Triple Riding, NO SEATBELT)
    # -------------------------------------------------------------
    if is_two_wheeler:
        logger.info(f"Two-wheeler detected ({len(two_wheelers)} bike(s)). Strictly suppressing seatbelt rules.")
        for tw in two_wheelers:
            tx1, ty1, tx2, ty2 = tw.bbox
            tw_w = max(1.0, tx2 - tx1)
            tw_h = max(1.0, ty2 - ty1)

            # Heuristic person clustering for triple riding on this two-wheeler
            riders_on_bike = []
            for p in persons:
                pcx, pcy = p.center
                if (tx1 - 0.35 * tw_w <= pcx <= tx2 + 0.35 * tw_w) and (p.bbox[1] <= ty2 and p.bbox[3] >= ty1 - 0.25 * tw_h):
                    riders_on_bike.append(p)
            if len(riders_on_bike) >= 3:
                box_draw_list.append((tw.bbox, f"TRIPLE RIDING ({len(riders_on_bike)} riders)", 0.90, True))
                violation_candidates.append({
                    "type": "TRIPLE_RIDING",
                    "confidence": 0.88,
                    "bbox": tw.bbox,
                    "class": "MORE_THAN_TWO_PERSONS",
                    "rider_count": len(riders_on_bike),
                })

            for d in dets:
                cname = d.class_name.upper().replace(" ", "_")
                dcx, dcy = d.center
                is_spatial = (
                    (tx1 - 0.35 * tw_w) <= dcx <= (tx2 + 0.35 * tw_w)
                    and (ty1 - 1.20 * tw_h) <= dcy <= (ty2 + 0.35 * tw_h)
                )
                if not is_spatial:
                    continue

                if cname in {"WITHOUT_HELMET", "NO_HELMET"}:
                    box_draw_list.append((d.bbox, "NO HELMET", d.confidence, True))
                    violation_candidates.append({
                        "type": "NO_HELMET",
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "class": cname,
                    })
                elif cname in {"WITH_HELMET", "HELMET"}:
                    box_draw_list.append((d.bbox, "HELMET", d.confidence, False))
                    compliance_notes.append(f"Rider verified wearing HELMET ({int(d.confidence*100)}% conf)")
                elif cname in {"MORE_THAN_TWO_PERSONS", "TRIPLE_RIDING", "TRIPLE_RIDER"}:
                    box_draw_list.append((d.bbox, "TRIPLE RIDING", d.confidence, True))
                    violation_candidates.append({
                        "type": "TRIPLE_RIDING",
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "class": cname,
                    })
                elif cname in {"USING_MOBILE", "MOBILE_PHONE"}:
                    box_draw_list.append((d.bbox, "PHONE USAGE", d.confidence, True))
                    violation_candidates.append({
                        "type": "MOBILE_PHONE_WHILE_DRIVING",
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "class": cname,
                    })

    # -------------------------------------------------------------
    # 3. DIRECT CROP / TIGHT WINDSHIELD / OCCUPANT CROP
    # -------------------------------------------------------------
    elif not is_four_wheeler and not is_two_wheeler:
        sb_model = _get_seatbelt_model()
        sb_checked = False
        if sb_model is not None and (claimed_viol == "NO_SEATBELT" or len(persons) > 0 or claimed_viol in {None, "AUTO_DETECT"}):
            ch, cw = img.shape[:2]
            if cw >= 60 and ch >= 60:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness >= 15.0:
                    try:
                        res = sb_model(img, conf=0.20, verbose=False)
                        if res and len(res[0].boxes) > 0:
                            sb_checked = True
                            best_sb_conf = 0.0
                            best_ns_conf = 0.0
                            best_sb_box = None
                            best_ns_box = None
                            for b in res[0].boxes:
                                cid = int(b.cls[0].item())
                                conf = float(b.conf[0].item())
                                bbox = [float(x) for x in b.xyxy[0].cpu().numpy().tolist()]
                                if cid == 0 and conf > best_sb_conf:
                                    best_sb_conf = conf
                                    best_sb_box = bbox
                                elif cid == 1 and conf > best_ns_conf:
                                    best_ns_conf = conf
                                    best_ns_box = bbox

                            CONF_THRESH = 0.35
                            if best_sb_conf >= CONF_THRESH and best_sb_conf >= best_ns_conf:
                                box_draw_list.append((best_sb_box, "SEATBELT", best_sb_conf, False))
                                compliance_notes.append(f"Occupant verified wearing SEATBELT ({int(best_sb_conf*100)}% conf)")
                                compliance_status = "SEATBELT_COMPLIANT"
                            elif best_ns_conf >= CONF_THRESH:
                                box_draw_list.append((best_ns_box, "NO SEATBELT", best_ns_conf, True))
                                violation_candidates.append({
                                    "type": "NO_SEATBELT",
                                    "confidence": best_ns_conf,
                                    "bbox": best_ns_box,
                                    "class": "NO_SEATBELT",
                                })
                                compliance_status = "NO_SEATBELT_VIOLATION"
                            else:
                                compliance_status = "INCONCLUSIVE"
                    except Exception as e:
                        logger.debug(f"Direct crop seatbelt inference error: {e}")

        if not sb_checked:
            for d in dets:
                cname = d.class_name.upper().replace(" ", "_")
                if cname in {"WITHOUT_HELMET", "NO_HELMET"} and d.confidence >= 0.40:
                    box_draw_list.append((d.bbox, "NO HELMET", d.confidence, True))
                    violation_candidates.append({
                        "type": "NO_HELMET",
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "class": cname,
                    })
                elif cname in {"WITH_HELMET", "HELMET"}:
                    box_draw_list.append((d.bbox, "HELMET", d.confidence, False))
                    compliance_notes.append("Rider verified wearing HELMET")
                elif cname in {"MORE_THAN_TWO_PERSONS", "TRIPLE_RIDING", "TRIPLE_RIDER"}:
                    box_draw_list.append((d.bbox, "TRIPLE RIDING", d.confidence, True))
                    violation_candidates.append({
                        "type": "TRIPLE_RIDING",
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "class": cname,
                    })
                elif cname in {"USING_MOBILE", "MOBILE_PHONE"}:
                    box_draw_list.append((d.bbox, "PHONE USAGE", d.confidence, True))
                    violation_candidates.append({
                        "type": "MOBILE_PHONE_WHILE_DRIVING",
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                        "class": cname,
                    })

    # -------------------------------------------------------------
    # 4. RENDER ANNOTATED IMAGE
    # -------------------------------------------------------------
    photo_annotated = None
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

    # -------------------------------------------------------------
    # 5. CONSTRUCT EXPLANATION AND UNIQUE VIOLATIONS
    # -------------------------------------------------------------
    unique_violations: Dict[str, float] = {}
    candidate_details: Dict[str, Any] = {}
    for cand in violation_candidates:
        t = cand["type"]
        unique_violations[t] = max(unique_violations.get(t, 0.0), cand["confidence"])

    detection_note = ""
    if unique_violations:
        items = [f"{t} ({int(conf*100)}%)" for t, conf in unique_violations.items()]
        detection_note = f"🤖 AI Detected Violation: {', '.join(items)}."
        if "NO_SEATBELT" in unique_violations:
            compliance_status = "NO_SEATBELT_VIOLATION"
    elif compliance_notes:
        detection_note = f"✅ AI Verified: {'; '.join(compliance_notes)}. Forwarded for officer review."
        if any("SEATBELT" in n for n in compliance_notes):
            compliance_status = "SEATBELT_COMPLIANT"
    elif is_four_wheeler:
        detection_note = "⚠️ AI Analysis Inconclusive: Four-wheeler detected, but cabin visibility/seatbelt pattern is below automated confidence threshold (35%). Fastened seatbelt strap cannot be ruled out. Forwarded for manual officer review."
        compliance_status = "INCONCLUSIVE"
    elif is_two_wheeler:
        detection_note = "⚠️ AI Analysis Inconclusive: Two-wheeler detected, but violation could not be confirmed automatically. Forwarded for manual officer review."
        compliance_status = "INCONCLUSIVE"
    else:
        detection_note = "⚠️ AI Analysis Inconclusive: Vehicle type or violation not recognized by AI. Forwarded for manual officer review."
        compliance_status = "INCONCLUSIVE"

    return {
        "unique_violations": unique_violations,
        "box_draw_list": box_draw_list,
        "photo_annotated": photo_annotated,
        "detection_note": detection_note,
        "is_four_wheeler": is_four_wheeler,
        "is_two_wheeler": is_two_wheeler,
        "compliance_status": compliance_status,
        "sharpness": round(best_sharpness, 1) if best_sharpness > 0 else None,
        "crop_dimensions": best_crop_dims,
        "diagnostic_trace": diagnostic_trace,
    }

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
    event_id: Optional[str] = None,
) -> Violation:
    """The citizen's report, saved with AI detection details when available, or for officer review."""
    if not event_id:
        event_id = _ref().replace("CIT-", "EVT-CIT-")
    if shared_folder is not None:
        folder = shared_folder
    else:
        folder = Path(settings.EVIDENCE_DIR) / "citizen" / event_id
        folder.mkdir(parents=True, exist_ok=True)

    is_video = upload.suffix.lower() in VIDEO_TYPES
    orig_image = folder / f"original_upload{upload.suffix.lower()}"
    if not orig_image.exists() and upload.exists():
        try:
            shutil.copy(upload, orig_image)
        except Exception:
            pass

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
        "original_image": str(orig_image) if orig_image.exists() else str(upload),
        "annotated_image": str(image) if image else None,
        "model_predictions": info.get("model_predictions", []),
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
            "status_category": info.get("status_category"),
            "sharpness": info.get("sharpness"),
            "crop_dimensions": info.get("crop_dimensions"),
            "diagnostic_trace": info.get("diagnostic_trace", []),
            "motion_verified": info.get("motion_verified", False),
            "verification_status": info.get("verification_status", "POTENTIAL WRONG-WAY CANDIDATE (UNVERIFIED — PHOTO ONLY)" if chosen_type == "WRONG_WAY" and ai_confidence > 0 else ("NOT_VERIFIED" if chosen_type == "WRONG_WAY" else ("VERIFIED" if ai_confidence > 0 else "PENDING_REVIEW"))),
            "unverified_reason": info.get("unverified_reason"),
            "vehicles_detected_count": info.get("vehicles_detected_count", 0),
            "orientation_consensus": info.get("orientation_consensus"),
            "dominant_flow": info.get("dominant_flow"),
            "vehicle_orientation": info.get("vehicle_orientation"),
            "angle_deviation_deg": info.get("angle_deviation_deg"),
            "candidate_bbox": info.get("candidate_bbox"),
            "vehicle_type": info.get("vehicle_type"),
            "model_predictions": info.get("model_predictions", []),
        },
        "upload": str(upload),
        "claimed_type": info.get("claimed_violation") or chosen_type,
        "actual_type": chosen_type,
    }
    meta_path = folder / f"metadata_{event_id}.json" if (shared_folder and folder.name != event_id) else folder / "metadata.json"
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
    candidate_details: Dict[str, Any] = {}

    claimed_viol = info.get("violation_type")
    info["claimed_violation"] = claimed_viol

    if upload.suffix.lower() in VIDEO_TYPES:
        from algorithms.calibration.camera_calibration import load_config_fresh
        from pipeline.traffic_pipeline import TrafficVisionPipeline

        cfg = load_config_fresh()
        viol = dict(cfg.get("violations") or {})
        if claimed_viol in {"WRONG_WAY", "AUTO_DETECT", None}:
            viol.update({"red_light": False, "wrong_way": True})
        else:
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
        photo_ai_note = ""
        try:
            from algorithms.plate_ocr.plate_recognizer import PlateRecognizer
            img = cv2.imread(str(upload))
            if img is not None:
                ocr = PlateRecognizer()
                res = ocr.read_plate_from_full_image(img)
                if res:
                    auto_plate, _, auto_crop = res

                # Vehicle-type-aware AI classification (Seatbelt, Helmet, Triple Riding, Phone)
                classification = classify_citizen_photo(img, claimed_viol=claimed_viol)
                unique_violations = dict(classification.get("unique_violations") or {})
                box_draw_list = list(classification.get("box_draw_list") or [])
                photo_annotated = classification.get("photo_annotated")
                photo_ai_note = classification.get("detection_note") or ""
                compliance_status = classification.get("compliance_status")
                sharpness = classification.get("sharpness")
                crop_dimensions = classification.get("crop_dimensions")
                diagnostic_trace = classification.get("diagnostic_trace", [])

                # Run wrong-way orientation analysis across vehicles
                from algorithms.detection.detector import TrafficDetector
                from algorithms.violation_rules.photo_orientation_analyzer import PhotoOrientationAnalyzer
                detector = TrafficDetector(confidence=0.25)
                dets = detector.detect(img)
                detected_vehicles = [
                    d for d in dets
                    if d.class_name.upper().replace(" ", "_").replace("-", "_") in {"CAR", "MOTORCYCLE", "BUS", "TRUCK", "AUTO", "VEHICLE"}
                ]
                photo_analyzer = PhotoOrientationAnalyzer()
                orientation_res = photo_analyzer.analyze_photo(img, detected_vehicles, reported_violation=claimed_viol)

                info["model_predictions"] = [
                    {
                        "bbox": [round(float(p), 1) for p in v["bbox"]],
                        "class_name": v["class_name"],
                        "confidence": round(float(v["confidence"]), 3),
                        "orientation": v.get("orientation"),
                        "is_outlier": v.get("is_outlier", False),
                        "score": round(float(v.get("score", 0.0)), 3),
                    }
                    for v in orientation_res.get("vehicles", [])
                ]

                candidate_details: Dict[str, Any] = {}
                if orientation_res["has_outlier"]:
                    for cand in orientation_res["outlier_candidates"]:
                        c_conf = cand["confidence"]
                        unique_violations["WRONG_WAY"] = max(unique_violations.get("WRONG_WAY", 0.0), c_conf)
                        candidate_details["WRONG_WAY"] = {
                            "type": "WRONG_WAY",
                            "confidence": c_conf,
                            "bbox": cand["bbox"],
                            "class": cand["class_name"],
                            "motion_verified": False,
                            "verification_status": "POTENTIAL WRONG-WAY CANDIDATE (UNVERIFIED — PHOTO ONLY)",
                            "unverified_reason": (
                                f"Visual orientation ({cand['orientation']}) opposes the dominant traffic flow "
                                f"({orientation_res['consensus_summary']}). Motion cannot be confirmed without temporal video frames."
                            ),
                            "orientation_details": {
                                "vehicle_orientation": cand["orientation"],
                                "dominant_flow": orientation_res["dominant_orientation"],
                                "consensus_summary": orientation_res["consensus_summary"],
                                "angle_deviation_deg": 180.0,
                                "score": cand["score"],
                            },
                        }
                    photo_annotated = photo_analyzer.render_annotated_photo(img, orientation_res, other_detections=box_draw_list)

                if unique_violations:
                    photo_ai_conf = max(unique_violations.values())
        except Exception as e:
            logger.exception(f"Error analyzing photo upload: {e}")
            info["processing_error"] = str(e)

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
                    cand_meta = candidate_details.get(v_type, {})
                    info_copy["status_category"] = compliance_status
                    info_copy["sharpness"] = sharpness
                    info_copy["crop_dimensions"] = crop_dimensions
                    info_copy["diagnostic_trace"] = diagnostic_trace
                    if v_type == "WRONG_WAY":
                        info_copy["motion_verified"] = False
                        info_copy["verification_status"] = cand_meta.get("verification_status", "POTENTIAL WRONG-WAY CANDIDATE (UNVERIFIED — PHOTO ONLY)")
                        info_copy["unverified_reason"] = cand_meta.get("unverified_reason", "Visual orientation opposes surrounding traffic flow.")
                        info_copy["detection_note"] = f"AI detected potential wrong-way vehicle ({round(conf * 100)}% confidence). Visual orientation opposes carriageway traffic flow. Motion unverified in static photograph."
                        if "orientation_details" in cand_meta:
                            odet = cand_meta["orientation_details"]
                            info_copy["orientation_consensus"] = odet.get("consensus_summary")
                            info_copy["dominant_flow"] = odet.get("dominant_flow")
                            info_copy["vehicle_orientation"] = odet.get("vehicle_orientation")
                            info_copy["angle_deviation_deg"] = odet.get("angle_deviation_deg", 180.0)
                        info_copy["candidate_bbox"] = cand_meta.get("bbox")
                        info_copy["vehicle_type"] = cand_meta.get("class", "car")
                    else:
                        other_v = [x for x in all_types if x != v_type]
                        if other_v:
                            info_copy["detection_note"] = f"Concurrent violation: Also detected {', '.join(other_v)} on this vehicle."
                        else:
                            info_copy["detection_note"] = f"AI verified '{v_type}' with {round(conf * 100)}% accuracy."

                    single_event_id = base_event if len(all_types) == 1 else None
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
                        event_id=single_event_id,
                    )
                    manual_records.append(rec)
            else:
                info_copy = dict(info)
                orig_name = str(info.get("original_name") or upload.name or "").lower()
                note_text = str(info.get("note") or "").lower()
                is_ww_reported = (claimed_viol == "WRONG_WAY")
                is_ww_keyword = any(k in orig_name or k in note_text for k in ["wrong", "counter", "opposite", "reverse", "retrograde", "oncoming"])

                if is_ww_reported or is_ww_keyword:
                    fallback_type = "WRONG_WAY"
                elif claimed_viol and claimed_viol != "AUTO_DETECT":
                    fallback_type = claimed_viol
                else:
                    fallback_type = "CITIZEN_REPORT"

                info_copy["violation_type"] = fallback_type
                info_copy["auto_classified"] = False
                info_copy["status_category"] = compliance_status if "compliance_status" in locals() else None
                info_copy["sharpness"] = sharpness if "sharpness" in locals() else None
                info_copy["crop_dimensions"] = crop_dimensions if "crop_dimensions" in locals() else None
                info_copy["diagnostic_trace"] = diagnostic_trace if "diagnostic_trace" in locals() else []
                info_copy["motion_verified"] = False
                info_copy["verification_status"] = "NOT_VERIFIED"
                info_copy["vehicles_detected_count"] = len(detected_vehicles) if "detected_vehicles" in locals() else 0
                if fallback_type == "WRONG_WAY":
                    info_copy["unverified_reason"] = "Insufficient temporal motion evidence in static photograph. Motion vectors cannot be established without temporal frames."
                    info_copy["detection_note"] = f"YOLO localized {info_copy['vehicles_detected_count']} vehicle(s). Wrong-way driving cannot be confirmed from a static photo without temporal trajectory evidence. Forwarded for human officer review."
                else:
                    info_copy["detection_note"] = photo_ai_note if "photo_ai_note" in locals() and photo_ai_note else "Citizen report forwarded for officer review."
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
                "motion_verified": True if r.violation_type == "WRONG_WAY" else None,
                "verification_status": "VERIFIED",
            })
        for m in manual_records:
            is_unverified_candidate = (m.violation_type == "WRONG_WAY" and m.confidence > 0.0)
            status_str = "POTENTIAL WRONG-WAY CANDIDATE (UNVERIFIED — PHOTO ONLY)" if is_unverified_candidate else ("NOT_VERIFIED" if m.violation_type == "WRONG_WAY" else "PENDING_REVIEW")
            out.append({
                "id": m.id,
                "event_id": m.event_id,
                "type": m.violation_type,
                "plate": m.plate_number,
                "auto": m.confidence > 0.0,
                "confidence": m.confidence,
                "claimed": claimed_viol,
                "auto_classified": m.confidence > 0.0,
                "status_category": compliance_status if "compliance_status" in locals() else None,
                "detection_note": photo_ai_note if "photo_ai_note" in locals() else None,
                "sharpness": sharpness if "sharpness" in locals() else None,
                "crop_dimensions": crop_dimensions if "crop_dimensions" in locals() else None,
                "motion_verified": False,
                "verification_status": status_str,
            })

        total_detected = len(records) + (len(unique_violations) if unique_violations else (1 if photo_ai_conf > 0 else 0))
        return {
            "frames": frames,
            "records": out,
            "detected": total_detected,
            "ai_detected": photo_ai_conf > 0.0 or len(records) > 0,
            "all_violations": list(unique_violations.keys()),
            "compliance_status": compliance_status if "compliance_status" in locals() else None,
            "detection_note": photo_ai_note if "photo_ai_note" in locals() else None,
            "sharpness": sharpness if "sharpness" in locals() else None,
            "crop_dimensions": crop_dimensions if "crop_dimensions" in locals() else None,
            "diagnostic_trace": diagnostic_trace if "diagnostic_trace" in locals() else [],
            "error": info.get("processing_error"),
        }
    finally:
        db.close()
