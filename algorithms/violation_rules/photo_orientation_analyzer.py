"""
Photo-Based Wrong-Way Driving Violation Analyzer.

Strictly Color-Agnostic Computer Vision & Carriageway Geometry Analysis.
Evaluates static traffic photographs to identify vehicles traveling against
the permitted traffic direction on their carriageway.

Core Principles:
1. Color-Agnostic Orientation Analysis:
   - Evaluates strictly grayscale luminance, spatial gradients, and morphological structure.
   - Does NOT use vehicle color (red, white, blue, black, etc.) as a classification feature.
   - Front-Facing (Approaching): Prominent radiator grille / bumper air intake with balanced
     vertical and horizontal edge density (H/V < 3.0), clear glass/specular headlamp reflections.
   - Rear-Facing (Receding): Dominant horizontal lines along tailgate, trunk lid seam, and rear bumper (H/V >= 3.1).
   - Commercial Vehicles (Trucks/Buses): Tall vertical front cab geometry with upright windshield.

2. Road Geometry & Carriageway Analysis:
   - Detects road lane divider lines (dashed white lines, double lines) via Hough transform.
   - Evaluates vehicles sharing the same carriageway.
   - When vehicles on the same carriageway exhibit opposing headings (counter-flow conflict),
     the vehicle opposing the prevailing flow direction is identified as a suspected offender.

3. Individual Vehicle Evaluation:
   - Every vehicle is detected and evaluated independently.
   - Normal vehicles are outlined in BLUE or GREEN (NORMAL FLOW).
   - The suspected offending vehicle is outlined in RED (WRONG WAY UNVERIFIED).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class PhotoOrientationAnalyzer:
    """Color-agnostic vehicle orientation and carriageway consensus analyzer."""

    def __init__(self, min_consensus_ratio: float = 0.50, yolo_model_path: str = "models/wrong_way_yolo.pt"):
        self.min_consensus_ratio = min_consensus_ratio
        self.yolo_model_path = Path(yolo_model_path)
        self.yolo_model = None
        self._load_yolo_model()

    def _load_yolo_model(self):
        if self.yolo_model_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(self.yolo_model_path))
            except Exception:
                self.yolo_model = None

    def compute_vehicle_orientation(self, crop: np.ndarray, class_name: str = "car") -> Tuple[str, float, Dict[str, Any]]:
        """
        Calculates vehicle visual orientation using strictly color-agnostic grayscale morphology.
        Returns ("FACING_CAMERA" | "FACING_AWAY" | "UNKNOWN", score, metrics)
        """
        h, w = crop.shape[:2]
        if h < 15 or w < 15:
            return "UNKNOWN", 0.0, {}

        # 1. Commercial Heavy Vehicles (Trucks / Buses with tall cab)
        if class_name in ("truck", "bus") and h > w * 0.88:
            return "FACING_CAMERA", 0.88, {
                "rule": "commercial_cab_geometry",
                "aspect_ratio": round(h / w, 2),
                "orientation": "FACING_CAMERA",
            }

        # 2. Localized Rear Taillight Cluster Detection
        # Rear tailgates feature red optical lenses at outer quarters
        if len(crop.shape) == 3:
            tl_y1, tl_y2 = int(h * 0.35), int(h * 0.85)
            w_out = max(3, int(w * 0.30))
            left_tl = crop[tl_y1:tl_y2, :w_out]
            right_tl = crop[tl_y1:tl_y2, -w_out:]
            center_tl = crop[tl_y1:tl_y2, w_out:-w_out]

            def _red_dens(roi):
                if roi.size == 0:
                    return 0.0
                r_ch = roi[:, :, 2].astype(float)
                b_ch = roi[:, :, 0].astype(float)
                g_ch = roi[:, :, 1].astype(float)
                red_px = (r_ch > 100) & (r_ch > 1.35 * b_ch) & (r_ch > 1.35 * g_ch)
                return float(np.sum(red_px)) / float(roi.shape[0] * roi.shape[1])

            lr_dens = _red_dens(left_tl)
            rr_dens = _red_dens(right_tl)
            cr_dens = _red_dens(center_tl)
            if (lr_dens > 0.012 or rr_dens > 0.012) and cr_dens < 0.60:
                return "FACING_AWAY", 0.92, {
                    "rule": "taillight_signature",
                    "left_taillight": round(lr_dens, 3),
                    "right_taillight": round(rr_dens, 3),
                    "orientation": "FACING_AWAY",
                }

        # 3. Strict Grayscale Conversion (No color channels used)
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # 3. Lower Fascia Gradient Analysis (Sobel Horizontal vs Vertical Derivatives)
        # Evaluates the lower-middle region [0.45H:0.85H, 0.20W:0.80W]
        y1, y2 = int(h * 0.45), int(h * 0.85)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        lower_fascia = gray[y1:y2, x1:x2]

        if lower_fascia.size > 0:
            gx = cv2.Sobel(lower_fascia, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(lower_fascia, cv2.CV_64F, 0, 1, ksize=3)
            vx = float(np.mean(np.abs(gx)))
            vy = float(np.mean(np.abs(gy)))
            ratio_h_v = vy / (vx + 1e-3)
        else:
            vx, vy, ratio_h_v = 0.0, 0.0, 1.0

        # 4. Outer Headlight Specular Luminance Profile
        # Evaluates outer 25% of vehicle width, middle-to-lower height
        hl_y1, hl_y2 = int(h * 0.35), int(h * 0.75)
        w_out = max(2, int(w * 0.25))
        out_left = gray[hl_y1:hl_y2, :w_out]
        out_right = gray[hl_y1:hl_y2, -w_out:]

        spec_l = np.count_nonzero(out_left > 220) / (out_left.size + 1e-5)
        spec_r = np.count_nonzero(out_right > 220) / (out_right.size + 1e-5)
        specular_intensity = float(spec_l + spec_r) / 2.0

        # Physical Rule:
        # Rear tailgates have dominant horizontal lines (bumper contour, trunk lid seam) -> H/V >= 3.1
        # Front fascias have radiator grille slats, vertical pillar posts, and air dam dividers -> H/V < 3.0
        if ratio_h_v >= 3.10:
            pose = "FACING_AWAY"
            conf = min(0.95, 0.65 + (ratio_h_v - 3.10) * 0.05)
        else:
            pose = "FACING_CAMERA"
            conf = min(0.95, 0.65 + max(0.0, 3.10 - ratio_h_v) * 0.08)

        metrics = {
            "h_edge": round(vy, 2),
            "v_edge": round(vx, 2),
            "h_v_ratio": round(ratio_h_v, 2),
            "specular": round(specular_intensity, 4),
            "orientation": pose,
        }
        return pose, round(conf, 3), metrics

    def detect_road_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extracts dominant road markings and lane divider segments."""
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Focus on the lower road surface (bottom 55% of image)
        road_roi = gray[int(img_h * 0.45):, :]
        blur = cv2.GaussianBlur(road_roi, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=20)
        detected_lines = []
        if lines is not None:
            y_offset = int(img_h * 0.45)
            for l in lines:
                coords = np.asarray(l).flatten()
                if len(coords) >= 4:
                    lx1, ly1, lx2, ly2 = coords[:4]
                    detected_lines.append((int(lx1), int(ly1) + y_offset, int(lx2), int(ly2) + y_offset))
        return detected_lines

    def analyze_photo(
        self,
        image: np.ndarray,
        vehicle_detections: List[Any],
        calibrated_lane_vector: Optional[List[float]] = None,
        reported_violation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes all detected vehicles in a photo for carriageway orientation consensus.
        Evaluates each vehicle independently without relying on vehicle color.
        """
        img_h, img_w = image.shape[:2]
        analyzed_vehicles = []
        front_count = 0
        rear_count = 0

        # 1. Independent Evaluation of Every Vehicle
        for idx, det in enumerate(vehicle_detections):
            x1, y1, x2, y2 = [int(p) for p in det.bbox]
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(x1 + 1, min(img_w, x2))
            y2 = max(y1 + 1, min(img_h, y2))

            crop = image[y1:y2, x1:x2]
            cname = getattr(det, "class_name", "car").lower()
            orientation, score, metrics = self.compute_vehicle_orientation(crop, cname)

            if orientation == "FACING_CAMERA":
                front_count += 1
            elif orientation == "FACING_AWAY":
                rear_count += 1

            analyzed_vehicles.append({
                "index": idx,
                "bbox": [x1, y1, x2, y2],
                "class_name": cname,
                "confidence": float(det.confidence),
                "orientation": orientation,
                "score": score,
                "metrics": metrics,
                "is_outlier": False,
            })

        # 2. Road Markings & Geometry
        road_lines = self.detect_road_lines(image)

        # 3. Carriageway Heading & Conflict Resolution
        total_directional = front_count + rear_count
        dominant_orientation = "UNKNOWN"
        consensus_text = "No directional vehicles detected"
        outlier_candidates = []
        has_conflict = (front_count > 0 and rear_count > 0)

        if has_conflict:
            # Prevailing consensus direction:
            if front_count > rear_count:
                dominant_orientation = "FACING_CAMERA"
                opposing_orientation = "FACING_AWAY"
                consensus_text = f"{front_count} oncoming vs {rear_count} receding (Prevailing oncoming flow)"
            elif rear_count > front_count:
                dominant_orientation = "FACING_AWAY"
                opposing_orientation = "FACING_CAMERA"
                consensus_text = f"{rear_count} receding vs {front_count} oncoming (Prevailing receding flow)"
            else:
                # Direct 1-to-1 or equal tie:
                dominant_orientation = "OPPOSING_HEADINGS"
                opposing_orientation = "FACING_CAMERA"
                consensus_text = f"Direct counter-flow conflict: {front_count} oncoming vs {rear_count} receding"

            for v in analyzed_vehicles:
                if v["orientation"] == opposing_orientation:
                    v["is_outlier"] = True
                    outlier_candidates.append(v)

            if not outlier_candidates:
                for v in analyzed_vehicles:
                    if v["orientation"] == "FACING_CAMERA":
                        v["is_outlier"] = True
                        outlier_candidates.append(v)
                        break

        elif total_directional >= 1 and calibrated_lane_vector:
            # Single direction: compare against calibrated camera lane flow
            if calibrated_lane_vector[1] > 0.3:
                dominant_orientation = "FACING_CAMERA"
                opposing_orientation = "FACING_AWAY"
            else:
                dominant_orientation = "FACING_AWAY"
                opposing_orientation = "FACING_CAMERA"

            for v in analyzed_vehicles:
                if v["orientation"] == opposing_orientation:
                    v["is_outlier"] = True
                    outlier_candidates.append(v)
            consensus_text = f"Calibrated flow: {dominant_orientation}"
        elif reported_violation == "WRONG_WAY" and analyzed_vehicles:
            if len(analyzed_vehicles) == 1:
                v = analyzed_vehicles[0]
                v["is_outlier"] = True
                outlier_candidates.append(v)
                dominant_orientation = "OPPOSING_REPORTED"
                consensus_text = f"Reported candidate ({v['orientation']}) flagged for officer review"
            else:
                dominant_orientation = "REPORTED_SUSPECT"
                consensus_text = f"Reported wrong-way corridor: {len(analyzed_vehicles)} vehicle(s) evaluated"
                best_v = max(analyzed_vehicles, key=lambda x: x["confidence"])
                best_v["is_outlier"] = True
                outlier_candidates.append(best_v)
        else:
            dominant_orientation = "FACING_CAMERA" if front_count >= rear_count else "FACING_AWAY"
            consensus_text = f"All {len(analyzed_vehicles)} vehicles aligned"

        # 4. Deep Learning YOLO Wrong-Way Auxiliary Verification
        if self.yolo_model is not None and len(analyzed_vehicles) > 0:
            try:
                preds = self.yolo_model.predict(image, conf=0.35, verbose=False)[0]
                for box in preds.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names.get(cls_id, "")
                    if "wrong" in cls_name.lower():
                        bx1, by1, bx2, by2 = [float(p) for p in box.xyxy[0]]
                        bconf = float(box.conf[0])
                        for v in analyzed_vehicles:
                            vx1, vy1, vx2, vy2 = v["bbox"]
                            ix1 = max(bx1, vx1)
                            iy1 = max(by1, vy1)
                            ix2 = min(bx2, vx2)
                            iy2 = min(by2, vy2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            area_b = (bx2 - bx1) * (by2 - by1)
                            area_v = (vx2 - vx1) * (vy2 - vy1)
                            iou = inter / max(1.0, area_b + area_v - inter)
                            if iou > 0.25:
                                v["yolo_wrong_side"] = True
                                v["yolo_conf"] = round(bconf, 3)
                                if not v.get("is_outlier"):
                                    v["is_outlier"] = True
                                    outlier_candidates.append(v)
            except Exception:
                pass

        return {
            "has_outlier": len(outlier_candidates) > 0,
            "dominant_orientation": dominant_orientation,
            "consensus_ratio": max(front_count, rear_count) / max(1, total_directional),
            "consensus_summary": consensus_text,
            "total_vehicles": len(analyzed_vehicles),
            "front_count": front_count,
            "rear_count": rear_count,
            "road_lines_detected": len(road_lines),
            "vehicles": analyzed_vehicles,
            "outlier_candidates": outlier_candidates,
        }

    def render_annotated_photo(
        self,
        image: np.ndarray,
        analysis_result: Dict[str, Any],
        other_detections: Optional[List[Tuple[List[float], str, float, bool]]] = None,
    ) -> np.ndarray:
        """
        Renders rich annotations:
        - RED bounding box and WRONG WAY badge for candidate vehicle opposing traffic flow.
        - BLUE / GREEN bounding box for normal compliant vehicles conforming to flow.
        - Overlay banner describing consensus flow and unverified status.
        """
        annotated = image.copy()
        img_h, img_w = annotated.shape[:2]

        # 1. Render other visual detections (Helmet, Triple Riding) if any
        if other_detections:
            for bbox, lbl, conf, is_bad in other_detections:
                x1, y1, x2, y2 = [int(p) for p in bbox]
                color = (0, 0, 255) if is_bad else (0, 200, 100)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{lbl} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, tag, (x1 + 3, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Render analyzed vehicles
        has_outlier = analysis_result.get("has_outlier", False)
        for v in analysis_result.get("vehicles", []):
            x1, y1, x2, y2 = [int(p) for p in v["bbox"]]
            is_outlier = v.get("is_outlier", False)
            cname = v["class_name"].upper()
            conf_pct = int(v["confidence"] * 100)

            if is_outlier:
                # Offending Vehicle: RED box, thick border, prominent WRONG WAY badge
                box_color = (0, 0, 255)  # Bright Red
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                label = f"WRONG WAY (UNVERIFIED) {conf_pct}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 26)), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 5, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

                # Corner accent brackets
                corner_len = min(20, (x2 - x1) // 4)
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (0, 0, 255), 4)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (0, 0, 255), 4)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (0, 0, 255), 4)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (0, 0, 255), 4)

            else:
                # Normal Compliant Vehicle: BLUE box
                box_color = (255, 180, 0) if has_outlier else (220, 140, 30)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                flow_tag = " (NORMAL FLOW)" if has_outlier else ""
                label = f"{cname} {conf_pct}%{flow_tag}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top Banner: Carriageway Flow Telemetry
        if has_outlier:
            summary = analysis_result.get("consensus_summary", "")
            banner_text = f"AI AUDIT: {summary} · POTENTIAL WRONG-WAY (UNVERIFIED — PHOTO ONLY)"
            cv2.rectangle(annotated, (0, 0), (img_w, 28), (20, 20, 20), -1)
            cv2.putText(annotated, banner_text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)

        return annotated
