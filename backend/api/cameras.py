import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.models.models import CameraSource
from backend.schemas.schemas import CameraCreate, CameraResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cameras", tags=["Cameras"])

# Active camera processing threads / flags
active_camera_streams = {}


@router.get("", response_model=List[CameraResponse])
def get_cameras(db: Session = Depends(get_db)):
    """List all registered traffic camera sources."""
    cams = db.query(CameraSource).all()
    # Seed default sample cameras if table is empty
    if not cams:
        default_cams = [
            CameraSource(
                name="MG Road Junction North",
                location="MG Road - Brigade Rd Crossing, Bengaluru",
                source_type="VIDEO_FILE",
                source_url="data/videos/sample_traffic.mp4",
                latitude=12.9754,
                longitude=77.6067,
                direction="Northbound",
                status="ONLINE",
            ),
            CameraSource(
                name="Silk Board Flyover Entrance",
                location="Silk Board Junction, Hosur Road, Bengaluru",
                source_type="RTSP",
                source_url="rtsp://demo:demo@stream.traffic.mock/silkboard",
                latitude=12.9176,
                longitude=77.6238,
                direction="Southbound",
                status="ONLINE",
            ),
            CameraSource(
                name="Hebbal Outer Ring Rd",
                location="Hebbal Flyover, Bellary Road, Bengaluru",
                source_type="VIDEO_FILE",
                source_url="data/videos/sample_hebbal.mp4",
                latitude=13.0358,
                longitude=77.5970,
                direction="Inbound",
                status="OFFLINE",
            ),
        ]
        db.add_all(default_cams)
        db.commit()
        cams = db.query(CameraSource).all()
    return cams


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    """Register a new CCTV or virtual camera source."""
    new_cam = CameraSource(
        name=payload.name,
        location=payload.location,
        source_type=payload.source_type,
        source_url=payload.source_url,
        latitude=payload.latitude,
        longitude=payload.longitude,
        direction=payload.direction,
        status="ONLINE",
    )
    db.add(new_cam)
    db.commit()
    db.refresh(new_cam)
    return new_cam


@router.post("/{cam_id}/start")
def start_camera(cam_id: int, db: Session = Depends(get_db)):
    """Start stream ingestion and detection pipeline for camera."""
    cam = db.query(CameraSource).filter(CameraSource.id == cam_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam.status = "ONLINE"
    db.commit()
    active_camera_streams[cam_id] = True
    return {"message": f"Camera {cam.name} stream started successfully", "status": "ONLINE"}


@router.post("/{cam_id}/stop")
def stop_camera(cam_id: int, db: Session = Depends(get_db)):
    """Stop stream ingestion and detection pipeline for camera."""
    cam = db.query(CameraSource).filter(CameraSource.id == cam_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam.status = "OFFLINE"
    db.commit()
    active_camera_streams[cam_id] = False
    return {"message": f"Camera {cam.name} stream stopped", "status": "OFFLINE"}
