import os
import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.database import get_db
from backend.models.models import Violation, Evidence
from backend.schemas.schemas import ViolationResponse, ViolationDetailResponse, EvidenceResponse

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
