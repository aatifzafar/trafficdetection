"""
E-Challan and Legal Notice Service
Generates official traffic violation notices compliant with the Indian Motor Vehicles Act.
"""

import uuid
import datetime
from typing import Dict, Any, Optional

# Legal penal sections and fine schedules under the Motor Vehicles (Amendment) Act
PENAL_CODE_SCHEDULE = {
    "NO_HELMET": {
        "section": "Section 194D, Motor Vehicles Act",
        "description": "Riding two-wheeler without protective safety headgear (helmet)",
        "fine_inr": 1000,
        "points": 2,
    },
    "TRIPLE_RIDING": {
        "section": "Section 194C, Motor Vehicles Act",
        "description": "Riding a motorcycle with more than one pillion passenger (triple riding)",
        "fine_inr": 1000,
        "points": 3,
    },
    "RED_LIGHT_JUMP": {
        "section": "Section 184, Motor Vehicles Act",
        "description": "Driving in violation of mandatory traffic control signal (jumping red light)",
        "fine_inr": 1000,
        "points": 4,
    },
    "WRONG_WAY": {
        "section": "Section 184 & 177, Motor Vehicles Act",
        "description": "Dangerous driving against designated one-way traffic flow vector",
        "fine_inr": 5000,
        "points": 6,
    },
    "NO_SEATBELT": {
        "section": "Section 194B, Motor Vehicles Act",
        "description": "Driving four-wheeled vehicle without fastening safety seatbelt",
        "fine_inr": 1000,
        "points": 1,
    },
    "MOBILE_PHONE_WHILE_DRIVING": {
        "section": "Section 184, Motor Vehicles Act",
        "description": "Use of handheld mobile communication device while operating motor vehicle",
        "fine_inr": 5000,
        "points": 4,
    },
}

DEFAULT_SCHEDULE = {
    "section": "Section 177, Motor Vehicles Act",
    "description": "General traffic order and safety violation",
    "fine_inr": 500,
    "points": 1,
}


class ChallanService:
    @staticmethod
    def get_penalty_info(violation_type: str) -> Dict[str, Any]:
        return PENAL_CODE_SCHEDULE.get(violation_type.upper(), DEFAULT_SCHEDULE)

    @classmethod
    def generate_challan(
        cls,
        violation_id: int,
        event_id: str,
        violation_type: str,
        plate_number: Optional[str],
        location: str,
        timestamp: datetime.datetime,
        confidence: float,
        evidence_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        penalty = cls.get_penalty_info(violation_type)
        challan_no = f"PROTO-{timestamp.strftime('%Y%m')}-{violation_id:05d}"
        due_date = timestamp + datetime.timedelta(days=30)

        challan_data = {
            "notice_type": "Prototype violation notice",
            "notice_number": challan_no,
            "challan_number": challan_no,
            "violation_id": violation_id,
            "event_id": event_id,
            "issuing_authority": "Prototype AI Traffic Violation Detection System (Research Project)",
            "jurisdiction": "Academic Research Prototype",
            "vehicle_plate": plate_number or "UNIDENTIFIED",
            "violation_type": violation_type,
            "mv_act_section": penalty["section"],
            "offense_description": penalty["description"],
            "fine_amount_inr": penalty["fine_inr"],
            "demerit_points": penalty["points"],
            "offense_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "location": location,
            "confidence_score": f"{confidence:.1%}",
            "evidence_image_url": evidence_image_path,
            "status": "PROTOTYPE_NOTICE",
            "verification_hash": uuid.uuid4().hex[:16].upper(),
        }
        return challan_data



challan_service = ChallanService()
