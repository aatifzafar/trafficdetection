import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import relationship
from backend.database.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    role = Column(String(20), default="operator")  # "admin", "officer", "operator"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class CameraSource(Base):
    __tablename__ = "camera_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)
    source_type = Column(String(20), default="VIDEO_FILE")  # RTSP, WEBCAM, VIDEO_FILE
    source_url = Column(String(500), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    direction = Column(String(50), nullable=True)  # e.g., "Northbound", "MG Road Inbound"
    status = Column(String(20), default="OFFLINE")  # ONLINE, OFFLINE, PROCESSING
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    violations = relationship("Violation", back_populates="camera")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(30), unique=True, index=True, nullable=False)
    vehicle_type = Column(String(50), nullable=True)  # motorcycle, car, bus, truck, auto
    first_seen = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    total_violations = Column(Integer, default=0)

    violations = relationship("Violation", back_populates="vehicle")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True, nullable=False)
    class_name = Column(String(50), nullable=False)
    camera_id = Column(Integer, ForeignKey("camera_sources.id"), nullable=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_frames = Column(Integer, default=1)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("camera_sources.id"), nullable=True)
    frame_number = Column(Integer, nullable=False)
    track_id = Column(Integer, index=True, nullable=True)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox = Column(String(100), nullable=False)  # JSON or "x1,y1,x2,y2"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)
    violation_type = Column(String(50), index=True, nullable=False)  # NO_HELMET, TRIPLE_RIDING, RED_LIGHT_JUMP, WRONG_WAY, NO_SEATBELT, MOBILE_PHONE
    track_id = Column(Integer, nullable=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    plate_number = Column(String(30), index=True, nullable=True)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    location = Column(String(200), nullable=False)
    camera_id = Column(Integer, ForeignKey("camera_sources.id"), nullable=True)
    evidence_path = Column(String(500), nullable=True)
    status = Column(String(30), default="CONFIRMED")  # PENDING_REVIEW, CONFIRMED, DISMISSED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    camera = relationship("CameraSource", back_populates="violations")
    vehicle = relationship("Vehicle", back_populates="violations")
    evidence_items = relationship("Evidence", back_populates="violation")
    alerts = relationship("Alert", back_populates="violation")


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), index=True, nullable=False)
    violation_id = Column(Integer, ForeignKey("violations.id"), nullable=True)
    image_path = Column(String(500), nullable=True)
    video_path = Column(String(500), nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    violation = relationship("Violation", back_populates="evidence_items")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), index=True, nullable=False)
    violation_id = Column(Integer, ForeignKey("violations.id"), nullable=True)
    violation_type = Column(String(50), nullable=False)
    channel = Column(String(20), default="CONSOLE")  # DASHBOARD, CONSOLE, EMAIL, SMS
    message = Column(Text, nullable=False)
    status = Column(String(20), default="SENT")  # SENT, FAILED, QUEUED
    sent_at = Column(DateTime, default=datetime.datetime.utcnow)

    violation = relationship("Violation", back_populates="alerts")
