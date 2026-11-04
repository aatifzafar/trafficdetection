import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from algorithms.violation_rules.base_rule import BaseViolationRule
from algorithms.tracking.base_tracker import TrackedObject
from algorithms.violation_rules.wrong_way_net import WrongWayLSTM, extract_trajectory_features

logger = logging.getLogger(__name__)


class WrongWayViolationRule(BaseViolationRule):
    """
    Intelligent Transportation Systems (ITS) Wrong-Way Driving Violation Detector.
    
    Architecture & Logic:
      1. Centroid History Tracking: Tracks vehicle centroid coordinates across consecutive frames.
      2. Displacement & Velocity Analysis: Computes trajectory displacement vector (dx, dy) over window.
      3. Stationary/Jitter Filtering: Enforces minimum displacement (>=15px) to reject stationary vehicles.
      4. Motion Vector Comparison: Calculates angular deviation against camera-calibrated lane direction.
      5. Deep Learning Trajectory Classification: Optional Bi-LSTM temporal sequence verification.
      6. Temporal Confirmation Window: Requires at least 6 of 10 consecutive frames to confirm violation.
      7. Cooldown Period: Prevents duplicate infractions within cooldown window (15.0s).
    """

    def __init__(
        self,
        lane_vector: Optional[List[float]] = None,
        max_angle_deviation_deg: float = 120.0,
        min_displacement_px: float = 15.0,
        min_trajectory_obs: int = 8,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 15.0,
        model_path: str = "models/wrong_way_lstm.pt",
    ):
        super().__init__(
            rule_name="WRONG_WAY",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )
        self.lane_vector = lane_vector or [0.0, 1.0]  # Default downward traffic flow
        self.max_angle_deviation_deg = max_angle_deviation_deg
        self.min_displacement_px = min_displacement_px
        self.min_trajectory_obs = min_trajectory_obs
        self.model_path = Path(model_path)
        self.model: Optional[WrongWayLSTM] = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
                model = WrongWayLSTM(
                    input_dim=ckpt.get("input_dim", 7),
                    hidden_dim=ckpt.get("hidden_dim", 64),
                    num_layers=ckpt.get("num_layers", 2),
                    num_classes=ckpt.get("num_classes", 2),
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self.model = model
                logger.info(f"Loaded trained Wrong-Way Bi-LSTM model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Wrong-Way DL checkpoint {self.model_path}: {e}")

    def set_lane_vector(self, dx: float, dy: float):
        norm = math.sqrt(dx * dx + dy * dy)
        if norm > 1e-4:
            self.lane_vector = [dx / norm, dy / norm]

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        vehicles: List[TrackedObject] = context.get("tracked_vehicles", [])
        violations: List[Dict[str, Any]] = []

        fps = float(context.get("fps") or 25.0)
        frame_idx = int(context.get("frame_idx") or 0)
        current_time = frame_idx / fps if fps > 0 else time.time()

        # Reference lane vector normalized
        ref_dx, ref_dy = self.lane_vector
        ref_norm = math.sqrt(ref_dx * ref_dx + ref_dy * ref_dy)
        if ref_norm < 1e-4:
            return violations
        ref_dx /= ref_norm
        ref_dy /= ref_norm

        for v in vehicles:
            # 1. Require minimum trajectory history observations
            if len(v.trajectory) < self.min_trajectory_obs:
                continue

            # 2. Velocity vector across observation window
            vx, vy = v.get_velocity_vector(window=self.min_trajectory_obs)
            displacement = math.sqrt(vx * vx + vy * vy)

            # 3. Filter stationary vehicles and micro-motion jitter
            if displacement < self.min_displacement_px:
                self.record_frame_evidence(v.track_id, False, current_time=current_time)
                continue

            # 4. Normalized motion direction
            norm_vx = vx / displacement
            norm_vy = vy / displacement

            # 5. Dot product to compute angular deviation from permitted lane direction
            dot = norm_vx * ref_dx + norm_vy * ref_dy
            dot = max(-1.0, min(1.0, dot))
            angle_rad = math.acos(dot)
            angle_deg = math.degrees(angle_rad)

            # 6. Physical Vector Angular Deviation Check
            # Vehicles travelling with traffic or making normal lane changes (<90 deg) are strictly non-violating
            is_wrong_way_vector = (angle_deg >= self.max_angle_deviation_deg)

            # 7. Deep Learning Bi-LSTM Inference
            dl_confidence = None
            is_wrong_way_dl = False
            if self.model is not None and len(v.trajectory) >= self.min_trajectory_obs:
                try:
                    feat = extract_trajectory_features(v.trajectory, self.lane_vector, fps=fps)
                    if feat is not None:
                        t_in = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                        probs = self.model.predict_proba(t_in).numpy()[0]
                        dl_confidence = float(probs[1])
                        # DL can confirm if kinematics also indicate opposing motion (>= 90 deg)
                        if angle_deg >= 90.0 and dl_confidence >= 0.70:
                            is_wrong_way_dl = True
                except Exception as e:
                    logger.debug(f"DL inference error on track {v.track_id}: {e}")

            # Combined candidate decision (physics-grounded: must be at least opposing flow)
            is_wrong_way_candidate = is_wrong_way_vector or (is_wrong_way_dl and angle_deg >= 105.0)

            # 8. Temporal Confirmation Window (requires >= min_positive_frames in sliding window)
            if self.record_frame_evidence(v.track_id, is_wrong_way_candidate, current_time=current_time):
                # Calculate grounded AI confidence based on angular deviation and temporal stability
                base_conf = 0.75 + (min(180.0, angle_deg) - self.max_angle_deviation_deg) / (180.0 - self.max_angle_deviation_deg) * 0.20
                if dl_confidence is not None:
                    final_conf = max(base_conf, dl_confidence)
                else:
                    final_conf = base_conf
                final_conf = min(0.98, max(0.75, final_conf))

                violations.append({
                    "violation_type": "WRONG_WAY",
                    "track_id": v.track_id,
                    "confidence": round(final_conf, 3),
                    "timestamp": time.time(),
                    "evidence_required": True,
                    "extra": {
                        "motion_verified": True,
                        "verification_status": "VERIFIED",
                        "lane_vector": [round(ref_dx, 3), round(ref_dy, 3)],
                        "velocity_vector": [round(vx, 2), round(vy, 2)],
                        "motion_direction": [round(norm_vx, 3), round(norm_vy, 3)],
                        "angle_deviation_deg": round(angle_deg, 1),
                        "displacement_px": round(displacement, 1),
                        "confirming_observations": f"{self.min_positive_frames}/{self.temporal_window} observations confirmed",
                        "temporal_confirmed": True,
                        "dl_model_active": bool(self.model is not None),
                        "dl_confidence": round(dl_confidence, 3) if dl_confidence is not None else None,
                        "trajectory_len": len(v.trajectory),
                    },
                })

        return violations
