from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.database import get_db
from backend.models.models import Alert
from backend.schemas.schemas import AlertResponse
from backend.services.alert_service import alert_service

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=List[AlertResponse])
def get_recent_alerts(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieve recent violation alerts."""
    alerts = db.query(Alert).order_by(desc(Alert.sent_at)).limit(limit).all()
    return alerts


@router.get("/live")
def get_live_alerts(limit: int = Query(10, ge=1, le=50)):
    """Fetch live real-time violation alerts from in-memory queue."""
    return alert_service.dashboard_handler.get_latest_alerts(limit=limit)
