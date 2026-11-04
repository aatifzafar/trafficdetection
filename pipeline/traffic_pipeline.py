import os
import time
import json
import uuid
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator, Tuple
import cv2
import numpy as np

from backend.config.settings import settings
from backend.database.database import SessionLocal, ensure_schema
from backend.db_models.models import Violation, Vehicle, Evidence, Alert, CameraSource
from backend.services.alert_service import alert_service
from algorithms.detection.detector import TrafficDetector, DetectionResult
from algorithms.tracking import create_tracker, TrackedObject
from algorithms.association.rider_association import RiderMotorcycleAssociator
from algorithms.plate_ocr.plate_recognizer import PlateRecognizer
from algorithms.violation_rules.rule_engine import TrafficRuleEngine
from pipeline.evidence_recorder import EvidenceRecorder
from algorithms.calibration.camera_calibration import (
    load_config_fresh, config_for_camera, get_camera_calibration, signal_state_from_roi, light_condition,
)

# Classes produced by the custom model; they are evidence, not objects to track
CUSTOM_CLASSES = {"helmet", "no_helmet", "number_plate", "plate", "with_helmet", "without_helmet", "more_than_two_persons", "using_mobile", "triple_riding", "triple_rider", "triple-riding-bikes", "normal"}
OCR_EVERY_N_FRAMES = 8       # periodic plate reads for vehicles without a plate yet
OCR_MIN_VEHICLE_PX = 80      # skip vehicles too small for a readable plate

logger = logging.getLogger(__name__)


class TrafficVisionPipeline:
    """
    End-to-end Master Computer Vision Pipeline:
    Frame -> Detect -> Track -> Associate -> OCR -> Rules -> Evidence -> DB -> Alert
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        camera_id: Optional[int] = None,
        location: str = "MG Road Junction, Bengaluru",
        persist: bool = True,
        source: str = "CCTV",
        model_traffic: Optional[str] = None,
    ):
        self.camera_id = camera_id
        self.location = location
        self.persist = persist  # False = dry run (camera test): count events, save nothing
        self.source = source    # CCTV for cameras, CITIZEN for videos uploaded on the Upload proof page
        self.event_counts: Dict[str, int] = {}
        ensure_schema()  # adds new review columns to older databases
        # config.yaml read fresh, with dashboard Settings and this camera's calibration applied
        base_cfg = config or load_config_fresh() or settings.CONFIG_RAW

        # 1. Detector
        self.detector = TrafficDetector(
            model_traffic_path=model_traffic or (base_cfg.get("detection") or {}).get("model_traffic", settings.MODEL_TRAFFIC),
            model_custom_path=getattr(settings, "MODEL_CUSTOM", "models/traffic_custom.pt"),
            model_helmet_path=settings.MODEL_HELMET,
            model_plate_path=settings.MODEL_PLATE,
            confidence=float((base_cfg.get("detection") or {}).get("confidence", settings.CONFIDENCE)),
            iou=settings.IOU,
            device=settings.DEVICE,
        )

        self.model_name = Path(str(self.detector.model_traffic.ckpt_path or "")).stem or "model"

        # 2. Tracker
        self.tracker = create_tracker(
            algorithm=(base_cfg.get("tracking") or {}).get("algorithm", settings.TRACKER_ALGO),
        )

        # 3. Rider Associator
        self.associator = RiderMotorcycleAssociator(
            max_distance=settings.MAX_RIDER_DISTANCE,
            min_overlap_ratio=settings.SPATIAL_IOU,
        )

        # 4. ALPR & OCR
        self.plate_recognizer = PlateRecognizer(
            engine_type=settings.OCR_ENGINE,
            plate_regex=settings.PLATE_REGEX,
        )

        # 5. Rule Engine with this camera's calibration applied
        self.calibration = get_camera_calibration(base_cfg, camera_id)
        self.rule_engine = TrafficRuleEngine(config=config_for_camera(base_cfg, camera_id))
        self.dedup_window_s = float(base_cfg.get("dedup_window_seconds", 300))
        # (violation_type, key) -> (video_time_s, violation_db_id) for duplicate merging
        self._recent_events: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self.merged_duplicates = 0
        self._created_ids: set = set()
        self._plate_crops: Dict[int, np.ndarray] = {}  # last plate crop per track, saved with evidence  # violations written by this run (plate back-fill is limited to these)
        self.last_light = "day"

        # 6. Evidence Recorder
        self.evidence_recorder = EvidenceRecorder(
            base_dir=settings.EVIDENCE_DIR,
            pre_seconds=settings.PRE_SECONDS,
            post_seconds=settings.POST_SECONDS,
        )

        # State tracking
        self.total_frames_processed = 0
        self.total_vehicles_seen = set()
        self.violation_counts = 0
        self.current_fps = 0.0
        self.avg_latency_ms = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = 25.0,
        signal_state: str = "RED",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Processes a single frame and returns annotated visualization and violation list."""
        start_time = time.time()
        self.total_frames_processed += 1

        # Keep rolling frame buffer for evidence clips
        self.evidence_recorder.push_frame(frame, frame_idx)

        # A. Detection
        detections = self.detector.detect(frame)
        if frame_idx % 30 == 0 or frame_idx <= 1:
            self.last_light = light_condition(frame)

        # Signal state: "AUTO" reads the lamp inside the calibrated signal area
        if str(signal_state).upper() == "AUTO":
            signal_state = signal_state_from_roi(frame, self.calibration.get("signal_roi"))
            if signal_state == "UNKNOWN":
                signal_state = "GREEN"  # never flag red-light jumps when the lamp cannot be read

        # B. Tracking (only vehicles and people; helmet/plate boxes are evidence, not tracks)
        track_dets = [d for d in detections if d.class_name.lower() not in CUSTOM_CLASSES]
        tracked_objects = self.tracker.update(frame, track_dets, frame_idx)

        # Filter vehicle classes vs other classes
        vehicle_classes = {"car", "motorcycle", "bus", "truck", "bicycle"}
        tracked_vehicles = [t for t in tracked_objects if t.class_name.lower() in vehicle_classes]
        for v in tracked_vehicles:
            self.total_vehicles_seen.add(v.track_id)

        # C. Rider-Motorcycle Association
        motorcycle_groups = self.associator.associate(tracked_objects)

        # D. Spatial association of number plate detections to vehicle boxes
        vehicle_plate_boxes: Dict[int, List[float]] = {}
        plate_dets = [d for d in detections if "plate" in d.class_name.lower()]
        for p_det in plate_dets:
            pcx, pcy = p_det.center
            for v in tracked_vehicles:
                vx1, vy1, vx2, vy2 = v.bbox
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    vehicle_plate_boxes[v.track_id] = p_det.bbox
                    break

        # E. Violation Rules Evaluation
        context = {
            "frame": frame,
            "frame_idx": frame_idx,
            "fps": fps,
            "detections": detections,
            "tracked_objects": tracked_objects,
            "tracked_vehicles": tracked_vehicles,
            "motorcycle_groups": motorcycle_groups,
            "detector": self.detector,
            "traffic_signal_state": signal_state,
        }
        confirmed_violations = self.rule_engine.evaluate_frame(context)

        # F. Plate reading (OCR is the slowest step, so it runs sparingly):
        #    every confirmed violator in this frame, plus one periodic read per unread vehicle
        #    every OCR_EVERY_N_FRAMES frames, and only for vehicles big enough to have a readable plate.
        viol_vids = {viol.get("track_id") for viol in confirmed_violations if viol.get("track_id")}

        target_vehicles = []
        for v in tracked_vehicles:
            vid = v.track_id
            vw = v.bbox[2] - v.bbox[0]
            vh = v.bbox[3] - v.bbox[1]
            if vw < OCR_MIN_VEHICLE_PX or vh < OCR_MIN_VEHICLE_PX:
                continue
            current_plate = self.plate_recognizer.get_track_plate(vid)
            periodic = current_plate is None and (frame_idx + vid) % OCR_EVERY_N_FRAMES == 0
            if vid in viol_vids or periodic:
                target_vehicles.append(v)

        h_f, w_f = frame.shape[:2]
        for v in target_vehicles:
            vid = v.track_id
            crop = None
            if getattr(self.detector, "has_custom_model", False) and vid in vehicle_plate_boxes:
                px1, py1, px2, py2 = vehicle_plate_boxes[vid]
                pw = px2 - px1
                ph = py2 - py1
                crop_x1 = max(0, int(round(px1 - 0.10 * pw)))
                crop_y1 = max(0, int(round(py1 - 0.10 * ph)))
                crop_x2 = min(w_f, int(round(px2 + 0.10 * pw)))
                crop_y2 = min(h_f, int(round(py2 + 0.10 * ph)))
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            else:
                vx1, vy1, vx2, vy2 = [int(p) for p in v.bbox]
                vh = vy2 - vy1
                # Motorcycles carry front plates under the headlight (y1 + 0.45*vh to vy2)
                y_offset = 0.45 if v.class_name.lower() == "motorcycle" else 0.60
                crop = frame[max(0, int(vy1 + y_offset * vh)):min(h_f, vy2), max(0, vx1):min(w_f, vx2)]

            if crop is not None and crop.size > 0:
                ocr_res = self.plate_recognizer.read_plate(crop)
                if ocr_res:
                    text, conf = ocr_res
                    self._plate_crops[vid] = crop.copy()
                    best_plate = self.plate_recognizer.update_track_plate(vid, text, conf)
                    if best_plate:
                        self._sync_plate_to_violations(vid, best_plate, conf)
                elif vid not in self._plate_crops:
                    self._plate_crops[vid] = crop.copy()

        # Persist newly confirmed violations (duplicates are merged into the earlier record)
        new_violations = []
        for viol in confirmed_violations:
            if self._handle_confirmed_violation(viol, frame, frame_idx, fps):
                new_violations.append(viol)


        # G. Render Annotations
        annotated_frame = self._render_annotations(
            frame.copy(),
            tracked_objects,
            motorcycle_groups,
            confirmed_violations,
            signal_state,
            start_time,
            detections=detections,
        )

        latency = (time.time() - start_time) * 1000.0
        self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency
        self.current_fps = 1000.0 / max(1.0, latency)

        return annotated_frame, new_violations

    def get_best_plate_and_crop(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Returns the best consensus plate number and corresponding crop image found so far."""
        best = self.plate_recognizer.get_best_overall_plate()
        if not best:
            crop = next(iter(self._plate_crops.values())) if self._plate_crops else None
            return None, crop
        plate_text, conf, track_id = best
        crop = self._plate_crops.get(track_id)
        if crop is None and self._plate_crops:
            crop = next(iter(self._plate_crops.values()))
        return plate_text, crop

    def _dedup_key(self, viol_type: str, track_id, plate_number) -> Tuple[str, str]:
        return (viol_type, f"plate:{plate_number}" if plate_number else f"track:{track_id}")

    def _merge_into(self, violation_id: int) -> None:
        if violation_id < 0:
            return
        db = SessionLocal()
        try:
            row = db.query(Violation).filter(Violation.id == violation_id).first()
            if row is not None:
                row.merged_count = (row.merged_count or 0) + 1
                db.commit()
        finally:
            db.close()

    def _handle_confirmed_violation(
        self,
        viol: Dict[str, Any],
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
    ):
        track_id = viol.get("track_id")
        plate_number = self.plate_recognizer.get_track_plate(track_id) if track_id else None
        viol_type = viol.get("violation_type", "UNKNOWN")
        confidence = float(viol.get("confidence", 0.85))

        # Duplicate merging: same vehicle (plate, or track when the plate is unknown) and same
        # violation within the dedup window of video time -> count it on the earlier record.
        video_t = frame_idx / (fps or 25.0)
        for key in {self._dedup_key(viol_type, track_id, plate_number), self._dedup_key(viol_type, track_id, None)}:
            prev = self._recent_events.get(key)
            if prev and video_t - prev[0] <= self.dedup_window_s:
                self._merge_into(prev[1])
                self.merged_duplicates += 1
                return False

        self.violation_counts += 1
        self.event_counts[viol_type] = self.event_counts.get(viol_type, 0) + 1
        if not self.persist:
            self._recent_events[self._dedup_key(viol_type, track_id, None)] = (video_t, -1)
            return True
        event_id = f"EVT-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
        viol.setdefault("extra", {})
        viol["extra"].update({"light": self.last_light, "frame_idx": frame_idx, "source": self.source})

        # 1. Record Physical Evidence (Image + Video + Metadata)
        evidence_info = self.evidence_recorder.record_violation(
            event_id=event_id,
            violation_type=viol_type,
            track_id=track_id,
            plate_number=plate_number,
            confidence=confidence,
            location=self.location,
            camera_id=self.camera_id,
            current_frame=frame,
            frame_idx=frame_idx,
            fps=fps,
            extra_metadata=viol.get("extra", {}),
        )

        # Save the plate crop next to the snapshot so reviewers can check the OCR
        crop = self._plate_crops.get(track_id) if track_id is not None else None
        if crop is not None and evidence_info.get("folder"):
            try:
                cv2.imwrite(os.path.join(evidence_info["folder"], "plate_crop.jpg"), crop)
            except Exception as e:
                logger.debug(f"Could not save plate crop: {e}")

        # 2. Database Persistence
        try:
            db = SessionLocal()
            try:
                # Find or create vehicle record if plate identified
                vehicle_rec = None
                if plate_number:
                    vehicle_rec = db.query(Vehicle).filter(Vehicle.plate_number == plate_number).first()
                    if not vehicle_rec:
                        vehicle_rec = Vehicle(
                            plate_number=plate_number,
                            total_violations=0,  # counted when an officer approves
                        )
                        db.add(vehicle_rec)
                        db.flush()
                    else:
                        vehicle_rec.last_seen = datetime.datetime.utcnow()

                # Create violation row
                v_row = Violation(
                    event_id=event_id,
                    violation_type=viol_type,
                    track_id=track_id,
                    vehicle_id=vehicle_rec.id if vehicle_rec else None,
                    plate_number=plate_number,
                    confidence=confidence,
                    location=self.location,
                    camera_id=self.camera_id,
                    evidence_path=evidence_info.get("frame_path"),
                    status="PENDING_REVIEW",  # nothing reaches a citizen until an officer approves
                    source=self.source,
                    merged_count=0,
                    light_condition=self.last_light,
                )
                db.add(v_row)
                db.flush()
                new_id = v_row.id
                self._created_ids.add(new_id)

                # Create evidence row
                ev_row = Evidence(
                    event_id=event_id,
                    violation_id=v_row.id,
                    image_path=evidence_info.get("frame_path"),
                    video_path=evidence_info.get("video_path"),
                    metadata_json=evidence_info.get("metadata_path"),
                )
                db.add(ev_row)

                # Create alert row
                al_row = Alert(
                    event_id=event_id,
                    violation_id=v_row.id,
                    violation_type=viol_type,
                    channel="DASHBOARD",
                    message=f"Violation {viol_type} detected at {self.location} for vehicle {plate_number or 'UNKNOWN'}",
                    status="SENT",
                )
                db.add(al_row)
                db.commit()
                for key in {self._dedup_key(viol_type, track_id, plate_number), self._dedup_key(viol_type, track_id, None)}:
                    self._recent_events[key] = (video_t, new_id)
            except Exception as dbe:
                db.rollback()
                logger.error(f"Database insertion failed for event {event_id}: {dbe}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Database connection error: {e}")

        # 3. Realtime Alert Notification
        alert_service.send_violation_alert(
            event_id=event_id,
            violation_type=viol_type,
            plate_number=plate_number,
            confidence=confidence,
            camera_name=f"Camera {self.camera_id or 1}",
            location=self.location,
            extra=viol.get("extra"),
        )
        return True

    def _sync_plate_to_violations(
        self,
        track_id: int,
        plate_number: str,
        confidence: float,
    ):
        """
        Retroactively updates previously logged violation and evidence records
        once OCR resolves a vehicle's license plate.
        """
        if not plate_number or not track_id or not self._created_ids:
            return

        try:
            db = SessionLocal()
            try:
                # 1. Ensure Vehicle registry entry exists
                vehicle_rec = db.query(Vehicle).filter(Vehicle.plate_number == plate_number).first()
                if not vehicle_rec:
                    vehicle_rec = Vehicle(
                        plate_number=plate_number,
                        total_violations=0,
                    )
                    db.add(vehicle_rec)
                    db.flush()

                # 2. Find any violations for this track_id missing a plate or marked UNIDENTIFIED
                unresolved_viols = (
                    db.query(Violation)
                    .filter(Violation.track_id == track_id)
                    .filter(Violation.id.in_(list(self._created_ids)))
                    .filter((Violation.plate_number == None) | (Violation.plate_number == "UNIDENTIFIED"))
                    .all()
                )

                for viol in unresolved_viols:
                    viol.plate_number = plate_number
                    viol.vehicle_id = vehicle_rec.id
                    # register under the plate key so later repeats of this plate merge into it
                    if viol.track_id is not None:
                        prev = self._recent_events.get(self._dedup_key(viol.violation_type, viol.track_id, None))
                        if prev:
                            pkey = self._dedup_key(viol.violation_type, None, plate_number)
                            other = self._recent_events.get(pkey)
                            if other and other[1] != viol.id and abs(other[0] - prev[0]) <= self.dedup_window_s:
                                # the same vehicle was recorded twice under two track ids: fold this one in
                                keep = db.query(Violation).filter(Violation.id == other[1]).first()
                                if keep is not None and viol.status == "PENDING_REVIEW":
                                    keep.merged_count = (keep.merged_count or 0) + 1 + (viol.merged_count or 0)
                                    db.query(Alert).filter(Alert.violation_id == viol.id).delete()
                                    db.query(Evidence).filter(Evidence.violation_id == viol.id).delete()
                                    db.delete(viol)
                                    self.merged_duplicates += 1
                                    continue
                            self._recent_events[pkey] = (prev[0], viol.id)

                    # Update associated alert message
                    alert = db.query(Alert).filter(Alert.violation_id == viol.id).first()  # noqa
                    if alert:
                        alert.message = f"Violation {viol.violation_type} detected at {viol.location} for vehicle {plate_number}"

                    # Save plate crop into evidence folder if missing
                    crop = self._plate_crops.get(track_id)
                    if crop is not None and viol.evidence_path:
                        folder = os.path.dirname(viol.evidence_path)
                        p_path = os.path.join(folder, "plate_crop.jpg")
                        if not os.path.exists(p_path):
                            try:
                                cv2.imwrite(p_path, crop)
                            except Exception as ce:
                                logger.debug(f"Could not save retroactive plate crop: {ce}")

                    # Update metadata JSON on disk if evidence exists
                    ev = db.query(Evidence).filter(Evidence.violation_id == viol.id).first()
                    if ev and ev.metadata_json and os.path.exists(ev.metadata_json):
                        try:
                            with open(ev.metadata_json, "r", encoding="utf-8") as f_meta:
                                meta_data = json.load(f_meta)
                            meta_data["plate_number"] = plate_number
                            with open(ev.metadata_json, "w", encoding="utf-8") as f_meta:
                                json.dump(meta_data, f_meta, indent=2)
                        except Exception as meta_err:
                            logger.debug(f"Could not update evidence metadata JSON: {meta_err}")

                db.commit()
                if unresolved_viols:
                    logger.info(
                        f"Retroactively linked plate {plate_number} to {len(unresolved_viols)} violation(s) for track ID {track_id}"
                    )
            except Exception as e:
                db.rollback()
                logger.error(f"Error retroactively syncing plate {plate_number} to track {track_id}: {e}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Database connection error in _sync_plate_to_violations: {e}")

    def _render_annotations(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        groups: List[Any],
        violations: List[Dict[str, Any]],
        signal_state: str,
        start_time: float,
        detections: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """Draws rich bounding boxes, labels, stop-lines, and telemetry."""
        h, w = frame.shape[:2]

        # Draw Stop Line
        stop_line_rule = self.rule_engine.rules.get("red_light")
        if stop_line_rule:
            sl = stop_line_rule.stop_line
            line_color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0)
            p1 = (int(sl["x1"]), int(sl["y1"]))
            p2 = (int(sl["x2"]), int(sl["y2"]))
            cv2.line(frame, p1, p2, line_color, 3)
            cv2.putText(
                frame,
                f"STOP LINE [{signal_state}]",
                (p1[0], max(20, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_color,
                2,
            )

        # Map track_id to violation types in this frame
        track_violation_map = {}
        for v in violations:
            tid = v.get("track_id")
            if tid is not None:
                track_violation_map.setdefault(tid, []).append(v.get("violation_type", "VIOLATION"))

        # Draw Tracked Objects
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(p) for p in obj.bbox]
            is_violating = obj.track_id in track_violation_map

            # Color scheme: Red for violator, Yellow for person, Amber for vehicles
            if is_violating:
                box_color = (0, 0, 255)
            elif obj.class_name == "person":
                box_color = (0, 215, 255)
            else:
                box_color = (255, 180, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label overlay
            plate = self.plate_recognizer.get_track_plate(obj.track_id)
            label = f"ID:{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
            if plate:
                label += f" [{plate}]"
            if is_violating:
                v_names = "/".join(track_violation_map[obj.track_id])
                label += f" [{v_names}]"

            # Badge background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0) if not is_violating else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw Custom Model Detections (Triple Riding, Helmet, Phone)
        if detections:
            for d in detections:
                cname = d.class_name.upper().replace(" ", "_").replace("-", "_")
                dx1, dy1, dx2, dy2 = [int(p) for p in d.bbox]
                badge_text = None
                badge_color = None
                if cname in {"MORE_THAN_TWO_PERSONS", "TRIPLE_RIDING", "TRIPLE_RIDER"}:
                    badge_text = f"TRIPLE RIDING {int(d.confidence * 100)}%"
                    badge_color = (0, 0, 255)
                elif cname in {"WITHOUT_HELMET", "NO_HELMET"}:
                    badge_text = f"NO HELMET {int(d.confidence * 100)}%"
                    badge_color = (0, 69, 255)
                elif cname in {"WITH_HELMET", "HELMET"}:
                    badge_text = f"HELMET {int(d.confidence * 100)}%"
                    badge_color = (0, 200, 0)
                elif cname in {"USING_MOBILE", "MOBILE_PHONE"}:
                    badge_text = f"PHONE USAGE {int(d.confidence * 100)}%"
                    badge_color = (0, 165, 255)

                if badge_text and badge_color:
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), badge_color, 2)
                    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (dx1, max(0, dy1 - 20)), (dx1 + bw + 6, dy1), badge_color, -1)
                    cv2.putText(
                        frame,
                        badge_text,
                        (dx1 + 3, max(12, dy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Draw Telemetry HUD
        hud_bg = frame[0:70, 0:420]
        if hud_bg.shape[0] == 70 and hud_bg.shape[1] == 420:
            sub_rect = frame[0:70, 0:420]
            black_rect = np.zeros(sub_rect.shape, dtype=np.uint8)
            cv2.addWeighted(sub_rect, 0.2, black_rect, 0.8, 1.0, sub_rect)

        fps_text = f"FPS: {self.current_fps:.1f} | Latency: {self.avg_latency_ms:.1f}ms"
        stat_text = f"Vehicles: {len(self.total_vehicles_seen)} | Violations: {self.violation_counts}"
        mode_text = "DEMO / RESEARCH PROTOTYPE" if settings.DEMO_MODE else "PRODUCTION"

        cv2.putText(frame, "TRAFFIC AI SURVEILLANCE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"{stat_text} | {mode_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        return frame

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Processes an entire video file from start to finish."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        all_violations = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                annotated, viols = self.process_frame(frame, frame_idx, fps=fps)
                if viols:
                    all_violations.extend(viols)

                if writer:
                    writer.write(annotated)

        finally:
            cap.release()
            if writer:
                writer.release()
            self.evidence_recorder.flush()

        return {
            "total_frames": frame_idx,
            "total_violations": len(all_violations),
            "unique_vehicles": len(self.total_vehicles_seen),
            "output_path": output_path,
        }
