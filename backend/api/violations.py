import os
import json
import logging
import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.database import get_db
from backend.db_models.models import Violation, Evidence
from pydantic import BaseModel
from backend.schemas.schemas import ViolationResponse, ViolationDetailResponse, EvidenceResponse
from backend.services import review_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/violations", tags=["Violations"])


@router.get("", response_model=List[ViolationResponse])
def get_violations(
    violation_type: Optional[str] = Query(None, description="Filter by violation type"),
    plate_number: Optional[str] = Query(None, description="Filter by vehicle plate"),
    camera_id: Optional[int] = Query(None, description="Filter by camera ID"),
    min_confidence: Optional[float] = Query(0.0, description="Minimum detection confidence"),
    status: Optional[str] = Query(None, description="Filter by review status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Retrieve violations with multi-criteria filtering and pagination."""
    query = db.query(Violation)

    if violation_type:
        query = query.filter(Violation.violation_type == violation_type.upper())
    if plate_number:
        query = query.filter(Violation.plate_number.ilike(f"%{plate_number}%"))
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    if min_confidence > 0.0:
        query = query.filter(Violation.confidence >= min_confidence)
    if status:
        query = query.filter(Violation.status == status.upper())

    violations = query.order_by(desc(Violation.timestamp)).offset(skip).limit(limit).all()
    return violations


from backend.services.challan_service import challan_service


@router.get("/{viol_id}", response_model=ViolationDetailResponse)
def get_violation_by_id(viol_id: int, db: Session = Depends(get_db)):
    """Retrieve detailed violation record with attached evidence references."""
    viol = db.query(Violation).filter(Violation.id == viol_id).first()
    if not viol:
        raise HTTPException(status_code=404, detail="Violation record not found")
    return viol


@router.get("/{viol_id}/evidence")
def get_violation_evidence(viol_id: int, db: Session = Depends(get_db)):
    """Retrieve evidence artifacts (snapshot frame, video clip, and metadata JSON)."""
    viol = db.query(Violation).filter(Violation.id == viol_id).first()
    if not viol:
        raise HTTPException(status_code=404, detail="Violation not found")

    evidence = db.query(Evidence).filter(Evidence.violation_id == viol_id).first()
    meta_content = {}
    if evidence and evidence.metadata_json and os.path.exists(evidence.metadata_json):
        try:
            with open(evidence.metadata_json, "r", encoding="utf-8") as f:
                meta_content = json.load(f)
        except Exception:
            pass

    return {
        "event_id": viol.event_id,
        "violation_type": viol.violation_type,
        "plate_number": viol.plate_number,
        "confidence": viol.confidence,
        "image_path": evidence.image_path if evidence else viol.evidence_path,
        "video_path": evidence.video_path if evidence else None,
        "metadata": meta_content,
    }


@router.get("/{viol_id}/challan")
def get_violation_challan(viol_id: int, db: Session = Depends(get_db)):
    """Generate or retrieve official Motor Vehicles Act e-challan notice."""
    viol = db.query(Violation).filter(Violation.id == viol_id).first()
    if not viol:
        raise HTTPException(status_code=404, detail="Violation not found")

    evidence = db.query(Evidence).filter(Evidence.violation_id == viol_id).first()
    img_path = evidence.image_path if evidence else viol.evidence_path

    challan = challan_service.generate_challan(
        violation_id=viol.id,
        event_id=viol.event_id,
        violation_type=viol.violation_type,
        plate_number=viol.plate_number,
        location=viol.location,
        timestamp=viol.timestamp or datetime.datetime.utcnow(),
        confidence=viol.confidence,
        evidence_image_path=img_path,
    )
    return challan


@router.post("/{viol_id}/issue-challan")
def issue_violation_challan(viol_id: int, reviewer: str = Query("API user"), db: Session = Depends(get_db)):
    """Approve the detection and create a prototype notice (not an official e-challan)."""
    try:
        viol = review_service.approve(db, viol_id, reviewer)
    except ValueError:
        raise HTTPException(status_code=404, detail="Violation not found")

    evidence = db.query(Evidence).filter(Evidence.violation_id == viol_id).first()
    img_path = evidence.image_path if evidence else viol.evidence_path

    challan = challan_service.generate_challan(
        violation_id=viol.id,
        event_id=viol.event_id,
        violation_type=viol.violation_type,
        plate_number=viol.plate_number,
        location=viol.location,
        timestamp=viol.timestamp or datetime.datetime.utcnow(),
        confidence=viol.confidence,
        evidence_image_path=img_path,
    )
    return {"message": "Approved. Prototype notice created.", "challan": challan}


class ReviewRequest(BaseModel):
    action: str  # approve | reject | correct_plate | dispute
    reviewer: str = "API user"
    reason: Optional[str] = None
    plate: Optional[str] = None


@router.post("/{viol_id}/review")
def review_violation(viol_id: int, payload: ReviewRequest, db: Session = Depends(get_db)):
    """Officer review: approve, reject (reason required), correct the plate, or record an owner dispute."""
    try:
        if payload.action == "approve":
            viol = review_service.approve(db, viol_id, payload.reviewer)
        elif payload.action == "reject":
            viol = review_service.reject(db, viol_id, payload.reviewer, payload.reason or "")
        elif payload.action == "correct_plate":
            viol = review_service.correct_plate(db, viol_id, payload.plate or "", payload.reviewer)
        elif payload.action == "dispute":
            viol = review_service.dispute(db, viol_id)
        else:
            raise HTTPException(status_code=400, detail="action must be approve, reject, correct_plate or dispute")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": viol.id, "status": viol.status, "plate_number": viol.plate_number,
            "reviewed_by": viol.reviewed_by, "reject_reason": viol.reject_reason}


@router.post("/{viol_id}/status")
def update_violation_status(viol_id: int, new_status: str = Query(..., description="PENDING_REVIEW, APPROVED, REJECTED, DISPUTED"), db: Session = Depends(get_db)):
    """Update violation audit status."""
    viol = db.query(Violation).filter(Violation.id == viol_id).first()
    if not viol:
        raise HTTPException(status_code=404, detail="Violation not found")

    viol.status = new_status.upper()
    db.commit()
    return {"message": f"Violation status updated to {viol.status}", "status": viol.status}

