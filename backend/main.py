import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config.settings import settings
from backend.database.database import Base, engine
from backend.api import cameras, violations, analytics, uploads, vehicles, alerts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("traffic_system")

# Ensure required storage directories exist
os.makedirs("data/videos", exist_ok=True)
os.makedirs("data/evidence", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Initialize database tables
Base.metadata.create_all(bind=engine)
logger.info("Database schemas verified and initialized.")

app = FastAPI(
    title="AI-Driven Smart Traffic Violation Detection and Alert System API",
    description="Backend services for real-time computer vision traffic surveillance, ALPR, and automated violation evidence generation.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evidence directory for serving images and video clips securely
evidence_path = Path(settings.EVIDENCE_DIR)
evidence_path.mkdir(parents=True, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(evidence_path)), name="evidence")

# Include Routers
app.include_router(cameras.router)
app.include_router(violations.router)
app.include_router(analytics.router)
app.include_router(uploads.router)
app.include_router(vehicles.router)
app.include_router(alerts.router)


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "demo_mode": settings.DEMO_MODE,
        "device": settings.DEVICE,
        "ocr_engine": settings.OCR_ENGINE,
        "tracker": settings.TRACKER_ALGO,
        "database_connected": True,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
