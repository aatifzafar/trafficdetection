import time
import logging
from typing import Dict, Any, List
from vision.rules.base_rule import BaseViolationRule
from vision.association.rider_association import MotorcycleRiderGroup

logger = logging.getLogger(__name__)


class TripleRidingRule(BaseViolationRule):
    """
    Detects triple (or greater) riding on two-wheelers.
    Triggers when > 2 distinct riders are spatially associated with a single motorcycle.
    """

    def __init__(
        self,
        temporal_window: int = 10,
        min_positive_frames: int = 6,
        cooldown_seconds: float = 12.0,
    ):
        super().__init__(
            rule_name="TRIPLE_RIDING",
            temporal_window=temporal_window,
            min_positive_frames=min_positive_frames,
            cooldown_seconds=cooldown_seconds,
        )

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        groups: List[MotorcycleRiderGroup] = context.get("motorcycle_groups", [])
        violations: List[Dict[str, Any]] = []

        for group in groups:
            bike_id = group.motorcycle.track_id
            is_triple = group.rider_count > 2

            if self.record_frame_evidence(bike_id, is_triple):
                conf = min(0.98, 0.75 + 0.08 * group.rider_count)
                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "track_id": bike_id,
                    "confidence": round(conf, 3),
                    "timestamp": time.time(),
                    "evidence_required": True,
                    "extra": {
                        "rider_count": group.rider_count,
                        "rider_ids": [r.track_id for r in group.riders],
                    },
                })

        return violations
