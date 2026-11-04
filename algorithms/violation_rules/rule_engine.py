import logging
from typing import Dict, Any, List, Optional
from algorithms.violation_rules.base_rule import BaseViolationRule
from algorithms.violation_rules.helmet_rule import HelmetViolationRule
from algorithms.violation_rules.triple_riding_rule import TripleRidingRule
from algorithms.violation_rules.red_light_rule import RedLightViolationRule
from algorithms.violation_rules.wrong_way_rule import WrongWayViolationRule
from algorithms.violation_rules.seatbelt_rule import SeatbeltViolationRule
from algorithms.violation_rules.mobile_phone_rule import MobilePhoneViolationRule

logger = logging.getLogger(__name__)


class TrafficRuleEngine:
    """
    Extensible Traffic Violation Rule Engine.
    Executes modular rules, applies temporal confirmation windows,
    and aggregates confirmed violation events for evidence and database storage.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        v_cfg = self.config.get("violations", {})
        temp_cfg = self.config.get("temporal_confirmation", {})
        cams_cfg = self.config.get("cameras", {})
        det_cfg = self.config.get("detection", {})

        w = int(temp_cfg.get("window", 10))
        min_p = int(temp_cfg.get("minimum_positive_frames", 6))
        cd = float(temp_cfg.get("violation_cooldown_seconds", 10.0))

        self.rules: Dict[str, BaseViolationRule] = {}

        if v_cfg.get("helmet", True):
            self.rules["helmet"] = HelmetViolationRule(
                temporal_window=w, min_positive_frames=min_p, cooldown_seconds=cd,
                allow_baseline=bool(v_cfg.get("helmet_baseline", True)),
            )

        if v_cfg.get("triple_riding", True):
            self.rules["triple_riding"] = TripleRidingRule(
                temporal_window=w, min_positive_frames=min(4, min_p), cooldown_seconds=cd
            )

        if v_cfg.get("red_light", True):
            stop_line = cams_cfg.get("default_stop_line", None)
            self.rules["red_light"] = RedLightViolationRule(
                stop_line=stop_line,
                temporal_window=max(3, w // 2),
                min_positive_frames=2,
                cooldown_seconds=cd,
            )

        if v_cfg.get("wrong_way", True):
            lane_vec = cams_cfg.get("default_lane_direction", [0, 1])
            max_angle = float(cams_cfg.get("max_angle_deviation_deg", 120))
            self.rules["wrong_way"] = WrongWayViolationRule(
                lane_vector=lane_vec,
                max_angle_deviation_deg=max_angle,
                temporal_window=w,
                min_positive_frames=min_p,
                cooldown_seconds=cd,
            )

        if v_cfg.get("seatbelt", True):
            sb_model = det_cfg.get("model_seatbelt", None)
            sb_perspective = cams_cfg.get("seatbelt_perspective") or cams_cfg.get("default_perspective") or cams_cfg.get("perspective", "AUTO")
            sb_conf = float(v_cfg.get("seatbelt_confidence", 0.35))
            sb_min_crop = int(cams_cfg.get("seatbelt_min_crop_px", 90))
            sb_min_sharp = float(cams_cfg.get("seatbelt_min_sharpness", 30.0))

            self.rules["seatbelt"] = SeatbeltViolationRule(
                model_path=sb_model,
                perspective=sb_perspective,
                confidence_threshold=sb_conf,
                temporal_window=w,
                min_positive_frames=min_p,
                cooldown_seconds=cd,
                min_crop_w=sb_min_crop,
                min_crop_h=sb_min_crop,
                min_sharpness=sb_min_sharp,
            )

        if v_cfg.get("mobile_phone", True):
            self.rules["mobile_phone"] = MobilePhoneViolationRule(
                temporal_window=w, min_positive_frames=min_p, cooldown_seconds=cd
            )

        logger.info(f"Initialized Rule Engine with active rules: {list(self.rules.keys())}")

    def evaluate_frame(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes all active rules for the current frame context.
        Returns a list of confirmed violation event dicts.
        """
        all_violations: List[Dict[str, Any]] = []

        for rule_name, rule in self.rules.items():
            try:
                rule_violations = rule.evaluate(context)
                if rule_violations:
                    all_violations.extend(rule_violations)
            except Exception as e:
                logger.error(f"Error evaluating rule '{rule_name}': {e}", exc_info=True)

        return all_violations
