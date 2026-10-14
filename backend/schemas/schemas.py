import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Camera Schemas
class CameraBase(BaseModel):
    name: str
    location: str
    source_type: str = "VIDEO_FILE"  # RTSP, WEBCAM, VIDEO_FILE
    source_url: str
    latitude: Optional[float] = 12.9716  # Default Bengaluru coords
    longitude: Optional[float] = 77.5946
    direction: Optional[str] = "Both"


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: int
    status: str
    created_at: datetime.datetime
    model_config = ConfigDict(from_attributes=True)


# Evidence Schemas
class EvidenceResponse(BaseModel):
    id: int
    event_id: str
    violation_id: Optional[int] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: datetime.datetime
    model_config = ConfigDict(from_attributes=True)


# Violation Schemas
class ViolationBase(BaseModel):
    event_id: str
    violation_type: str
    track_id: Optional[int] = None
    vehicle_id: Optional[int] = None
    plate_number: Optional[str] = None
    confidence: float
    timestamp: datetime.datetime
    location: str
    camera_id: Optional[int] = None
    evidence_path: Optional[str] = None
    status: str = "CONFIRMED"


class ViolationResponse(ViolationBase):
    id: int
    created_at: datetime.datetime
    model_config = ConfigDict(from_attributes=True)


class ViolationDetailResponse(ViolationResponse):
    camera: Optional[CameraResponse] = None
    evidence_items: List[EvidenceResponse] = []


# Vehicle Schemas
class VehicleResponse(BaseModel):
    id: int
    plate_number: str
    vehicle_type: Optional[str] = None
    first_seen: datetime.datetime
    last_seen: datetime.datetime
    total_violations: int
    violations: List[ViolationResponse] = []
    model_config = ConfigDict(from_attributes=True)


# Alert Schemas
class AlertResponse(BaseModel):
    id: int
    event_id: str
    violation_id: Optional[int] = None
    violation_type: str
    channel: str
    message: str
    status: str
    sent_at: datetime.datetime
    model_config = ConfigDict(from_attributes=True)


# Analytics Schemas
class AnalyticsSummaryResponse(BaseModel):
    total_violations: int
    today_violations: int
    active_cameras: int
    total_vehicles_seen: int
    detection_fps: float
    avg_latency_ms: float
    demo_mode: bool


class AnalyticsByTypeResponse(BaseModel):
    violation_type: str
    count: int


class AnalyticsByHourResponse(BaseModel):
    hour: int
    count: int


class AnalyticsHotspotResponse(BaseModel):
    location: str
    latitude: float
    longitude: float
    violation_count: int
    primary_violation: str
