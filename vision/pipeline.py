import os
import time
import uuid
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator, Tuple
import cv2
import numpy as np

from backend.config.settings import settings
from backend.database.database import SessionLocal
from backend.models.models import Violation, Vehicle, Evidence, Alert, CameraSource
from backend.services.alert_service import alert_service
from vision.detector.detector import TrafficDetector, DetectionResult
from vision.tracker import create_tracker, TrackedObject
from vision.association.rider_association import RiderMotorcycleAssociator
from vision.ocr.plate_recognizer import PlateRecognizer
from vision.rules.rule_engine import TrafficRuleEngine
from vision.evidence.evidence_recorder import EvidenceRecorder

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
    ):
        self.camera_id = camera_id
        self.location = location

        # 1. Detector
        self.detector = TrafficDetector(
            model_traffic_path=settings.MODEL_TRAFFIC,
            model_custom_path=getattr(settings, "MODEL_CUSTOM", "models/traffic_custom.pt"),
            model_helmet_path=settings.MODEL_HELMET,
            model_plate_path=settings.MODEL_PLATE,
            confidence=settings.CONFIDENCE,
            iou=settings.IOU,
            device=settings.DEVICE,
        )

        # 2. Tracker
        self.tracker = create_tracker(
            algorithm=settings.TRACKER_ALGO,
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

        # 5. Rule Engine (pass real configuration)
        self.rule_engine = TrafficRuleEngine(config=config or settings.CONFIG_RAW)

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

        # B. Tracking
        tracked_objects = self.tracker.update(frame, detections, frame_idx)

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

        # F. OCR ONLY for vehicles with a confirmed violation (with 10% padding crop if custom model)
        for viol in confirmed_violations:
            vid = viol.get("track_id")
            if vid is not None:
                if getattr(self.detector, "has_custom_model", False) and vid in vehicle_plate_boxes:
                    px1, py1, px2, py2 = vehicle_plate_boxes[vid]
                    pw = px2 - px1
                    ph = py2 - py1
                    h_f, w_f = frame.shape[:2]
                    # Crop with 10% padding
                    crop_x1 = max(0, int(round(px1 - 0.10 * pw)))
                    crop_y1 = max(0, int(round(py1 - 0.10 * ph)))
                    crop_x2 = min(w_f, int(round(px2 + 0.10 * pw)))
                    crop_y2 = min(h_f, int(round(py2 + 0.10 * ph)))
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size > 0:
                        ocr_res = self.plate_recognizer.read_plate(crop)
                        if ocr_res:
                            text, conf = ocr_res
                            self.plate_recognizer.update_track_plate(vid, text, conf)
                else:
                    # Baseline bumper crop fallback if no custom model
                    v_obj = next((v for v in tracked_vehicles if v.track_id == vid), None)
                    if v_obj:
                        vx1, vy1, vx2, vy2 = [int(p) for p in v_obj.bbox]
                        h_f, w_f = frame.shape[:2]
                        vh = vy2 - vy1
                        crop = frame[max(0, int(vy1 + 0.6 * vh)):min(h_f, vy2), max(0, vx1):min(w_f, vx2)]
                        if crop.size > 0:
                            ocr_res = self.plate_recognizer.read_plate(crop)
                            if ocr_res:
                                text, conf = ocr_res
                                self.plate_recognizer.update_track_plate(vid, text, conf)

            # Persist violation, generate physical evidence, and dispatch alert
            self._handle_confirmed_violation(viol, frame, frame_idx, fps)


        # G. Render Annotations
        annotated_frame = self._render_annotations(
            frame.copy(),
            tracked_objects,
            motorcycle_groups,
            confirmed_violations,
            signal_state,
            start_time,
        )

        latency = (time.time() - start_time) * 1000.0
        self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency
        self.current_fps = 1000.0 / max(1.0, latency)

        return annotated_frame, confirmed_violations

    def _handle_confirmed_violation(
        self,
        viol: Dict[str, Any],
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
    ):
        self.violation_counts += 1
        event_id = f"EVT-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
        track_id = viol.get("track_id")
        plate_number = self.plate_recognizer.get_track_plate(track_id) if track_id else None
        viol_type = viol.get("violation_type", "UNKNOWN")
        confidence = float(viol.get("confidence", 0.85))

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
                            total_violations=1,
                        )
                        db.add(vehicle_rec)
                        db.flush()
                    else:
                        vehicle_rec.total_violations += 1
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
                    status="CONFIRMED",
                )
                db.add(v_row)
                db.flush()

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

    def _render_annotations(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        groups: List[Any],
        violations: List[Dict[str, Any]],
        signal_state: str,
        start_time: float,
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

        violating_track_ids = {v["track_id"] for v in violations if "track_id" in v}

        # Draw Tracked Objects
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(p) for p in obj.bbox]
            is_violating = obj.track_id in violating_track_ids

            # Color scheme: Red for violator, Yellow for person, Cyan for vehicles
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
                label += " [VIOLATION]"

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
