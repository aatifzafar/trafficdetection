import os
import shutil
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.models.models import CameraSource
from vision.pipeline import TrafficVisionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Video Upload & Processing"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
MAX_FILE_SIZE = 250 * 1024 * 1024  # 250 MB


def sanitize_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ")
    return cleaned.replace(" ", "_")


def run_pipeline_task(video_path: str, camera_id: int, location: str):
    try:
        pipeline = TrafficVisionPipeline(camera_id=camera_id, location=location)
        output_path = f"data/videos/processed_{os.path.basename(video_path)}"
        results = pipeline.process_video(video_path, output_path=output_path)
        logger.info(f"Video processing complete for {video_path}: {results}")
    except Exception as e:
        logger.error(f"Error during video processing of {video_path}: {e}", exc_info=True)


@router.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    location: str = Form("MG Road Crossing, Bengaluru"),
    process_immediately: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Accepts video upload, validates format and size, stores file,
    registers a virtual camera, and optionally initiates pipeline processing.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    upload_dir = Path("data/videos")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex[:8]}_{sanitize_filename(file.filename)}"
    destination_path = upload_dir / safe_name

    # Save uploaded file safely
    try:
        with destination_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    # Check file size limit
    file_size = destination_path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File exceeds maximum allowed size of 250MB")

    # Register virtual camera entry for this video
    cam = CameraSource(
        name=f"Virtual Cam ({file.filename})",
        location=location,
        source_type="VIDEO_FILE",
        source_url=str(destination_path),
        latitude=12.9716,
        longitude=77.5946,
        direction="Inbound",
        status="PROCESSING" if process_immediately else "OFFLINE",
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    if process_immediately:
        background_tasks.add_task(run_pipeline_task, str(destination_path), cam.id, location)

    return {
        "message": "Video successfully uploaded and registered.",
        "filename": safe_name,
        "video_path": str(destination_path),
        "camera_id": cam.id,
        "processing_started": process_immediately,
    }
