"""
Officer review workflow.

Every detection (CCTV or citizen report) starts as PENDING_REVIEW. Only an officer can
move it to APPROVED (a prototype notice is created for the linked app user) or REJECTED
(kept, with the reason, as training data for the next model version).
"""

import datetime
from typing import Optional

from backend.db_models.models import Violation, Vehicle

PENDING = "PENDING_REVIEW"
APPROVED = "APPROVED"
REJECTED = "REJECTED"
DISPUTED = "DISPUTED"

STATUS_LABELS = {
    PENDING: "Waiting for review",
    APPROVED: "Approved",
    REJECTED: "Rejected",
    DISPUTED: "Disputed by owner",
    # legacy values from older databases
    "CONFIRMED": "Approved",
    "CHALLAN_ISSUED": "Approved",
    "DISMISSED": "Rejected",
}

VIOLATION_LABELS = {
    "NO_HELMET": "No helmet",
    "TRIPLE_RIDING": "Triple riding",
    "RED_LIGHT_JUMP": "Red light jump",
    "WRONG_WAY": "Wrong way",
    "NO_SEATBELT": "No seatbelt",
    "MOBILE_PHONE_WHILE_DRIVING": "Phone while driving",
    "CITIZEN_REPORT": "Citizen report",
}

REJECT_REASONS = [
    "Not a violation",
    "Wrong vehicle or rider",
    "Duplicate of another record",
    "Image too unclear to decide",
    "Other",
]


def violation_label(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    return VIOLATION_LABELS.get(code, code.replace("_", " ").capitalize())


def status_label(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    return STATUS_LABELS.get(code, code.replace("_", " ").capitalize())


def _now():
    return datetime.datetime.utcnow()


def _get(db, viol_id: int) -> Violation:
    viol = db.query(Violation).filter(Violation.id == viol_id).first()
    if viol is None:
        raise ValueError(f"Violation {viol_id} not found")
    return viol


def _link_vehicle(db, viol: Violation):
    if not viol.plate_number:
        return
    veh = db.query(Vehicle).filter(Vehicle.plate_number == viol.plate_number).first()
    if veh is None:
        veh = Vehicle(plate_number=viol.plate_number, total_violations=0)
        db.add(veh)
        db.flush()
    viol.vehicle_id = veh.id


def approve(db, viol_id: int, reviewer: str) -> Violation:
    viol = _get(db, viol_id)
    viol.status = APPROVED
    viol.reviewed_by = reviewer
    viol.reviewed_at = _now()
    viol.reject_reason = None
    _link_vehicle(db, viol)
    if viol.vehicle_id:
        veh = db.query(Vehicle).filter(Vehicle.id == viol.vehicle_id).first()
        if veh:
            veh.total_violations = (veh.total_violations or 0) + 1
            veh.last_seen = viol.timestamp or _now()
    db.commit()
    return viol


def reject(db, viol_id: int, reviewer: str, reason: str) -> Violation:
    if not reason:
        raise ValueError("A reason is required to reject a detection")
    viol = _get(db, viol_id)
    viol.status = REJECTED
    viol.reviewed_by = reviewer
    viol.reviewed_at = _now()
    viol.reject_reason = reason
    db.commit()
    return viol


def correct_plate(db, viol_id: int, plate: str, reviewer: str) -> Violation:
    plate = (plate or "").strip().upper().replace(" ", "")
    if not plate:
        raise ValueError("Plate cannot be empty")
    viol = _get(db, viol_id)
    viol.plate_number = plate
    viol.plate_edited = True
    viol.reviewed_by = reviewer
    _link_vehicle(db, viol)
    db.commit()
    return viol


def dispute(db, viol_id: int) -> Violation:
    viol = _get(db, viol_id)
    if viol.status != APPROVED:
        raise ValueError("Only approved notices can be disputed")
    viol.status = DISPUTED
    db.commit()
    return viol
