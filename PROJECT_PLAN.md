# B.Tech Design Project-I: Project Plan & Architectural Blueprint

**Project Title:** AI-Driven Smart Traffic Violation Detection and Alert System  
**Lead AI Engineer & Full-Stack Developer:** Aatif Zafar  
**Scope:** Real-Time Traffic Violation Detection, Multi-Object Tracking, Evidence Generation, Multi-Frame License Plate Recognition (ALPR), and Command Center Dashboard.

---

## 1. System Overview & Objectives
The system automates the detection and reporting of traffic safety and traffic order violations using video streams from surveillance cameras, dashcams, and mobile video feeds. 

### The 6 Official Project Violations (Alliance University):
1. **Riding Without Helmet (`NO_HELMET`):** Detection of motorcycle riders and pillion passengers without safety helmets, with spatial rider-motorcycle association.
2. **Triple Riding (`TRIPLE_RIDING`):** Direct deep learning detection via `MORE_THAN_TWO_PERSONS` (92.3% mAP50) + spatial clustering fallback (>2 riders).
3. **Red-Light Jumping (`RED_LIGHT_JUMP`):** Dynamic virtual stop-line crossing detection correlated with traffic signal phase (RED/GREEN).
4. **Wrong-Way Driving (`WRONG_WAY`):** Lane vector and trajectory centroid angle tracking across sliding frame windows.
5. **Illegal Parking (`ILLEGAL_PARKING`):** Dwell-time tracking for stationary vehicles inside calibrated prohibited polygon zones.
6. **Unsafe Movement / Speeding (`UNSAFE_MOVEMENT`):** Homography distance-calibrated velocity estimation and erratic zigzag trajectory tracking.

---

## 2. Technical Stack
- **Language:** Python 3.11+ (Active Environment: Python 3.13)
- **Computer Vision & Deep Learning:** Ultralytics YOLO (`yolo26n.pt`, `traffic_custom.pt`), PyTorch, OpenCV
- **Multi-Object Tracking (MOT):** ByteTrack (Ultralytics native engine with IoU fallback)
- **OCR Engine:** EasyOCR / PyTesseract with regex-based Indian High-Security Registration Plate (HSRP) format validation and multi-frame voting
- **Backend API:** FastAPI with async ASGI handlers and CORS
- **Database & ORM:** SQLAlchemy ORM with SQLite (Local prototype/Zero-setup) and PostgreSQL support
- **Monitoring Dashboard:** Streamlit with Plotly visual analytics, live annotated video feed, filterable violation triage, and vehicle search
- **Alert System:** Modular AlertService dispatching to Dashboard (live queue), Console logger, and Prototype Violation Notice Generator

---

## 3. Model Weight & Dataset Reality Assessment

### Active Models & Deployed Weights:
- **Base Pretrained Traffic Model (`models/yolo26n.pt`):**
  - Real-time NMS-free vehicle and person tracking.
- **Custom Trained Multi-Violation Model (`models/traffic_custom.pt`):**
  - Trained on Google Colab (Tesla T4 GPU) on 6,057 images (`05_triple_riding`).
  - Class 0: `MORE_THAN_TWO_PERSONS` (Triple Riding — **92.3% mAP50**)
  - Class 1: `USING_MOBILE` (Mobile Phone — **71.5% mAP50**)
  - Class 2: `WITHOUT_HELMET` (No Helmet — **61.8% mAP50**)
  - Class 3: `WITH_HELMET` (Helmet Compliant — **83.5% mAP50**)
  - Class 4: `normal` (Standard Traffic — **67.5% mAP50**)

---

## 4. Phase-by-Phase Roadmap & Milestone Status

### Phase 0: Base Architecture Upgrade (COMPLETED)
- [x] Upgraded to `ultralytics>=8.4.0` supporting modern YOLO architectures.
- [x] Integrated `yolo26n.pt` and `yolov8n.pt` into `models/`.
- [x] Verified box parsing in `detector.py` and validated pipeline execution.

### Phase 1: Merged Dataset Construction (COMPLETED)
- [x] Consolidated raw dataset archives into a unified dataset.
- [x] Image byte deduplication and train/validation splits.

### Phase 2: Google Colab Training Pipeline (COMPLETED)
- [x] Created `training/train_colab.ipynb` for T4 GPU training with automated export.

### Phase 3: Core Pipeline Code Fixes & Robustness (COMPLETED)
- [x] Custom model full-frame single-pass inference (`models/traffic_custom.pt`).
- [x] Spatial top-35% rider bounding box rule for helmet detection.
- [x] Targeted 10% padded plate crop OCR triggered for confirmed violation tracks.
- [x] Synchronized video-time cooldown calculation (`frame_idx / fps`).
- [x] ByteTrack integration with native Ultralytics tracker.
- [x] Eliminated all hardcoded metrics from Dashboard and Analytics API.
- [x] Rebranded notices to "Prototype violation notice" (`PROTO-`).

### Phase 4: Video Processing CLI & Hardware Benchmarking (COMPLETED)
- [x] End-to-end video inference CLI (`scripts/run_video.py`) with stop-line and traffic signal state controls.
- [x] Processed test video `data/videos/22.mp4` and generated forensic detection snapshots.
- [x] Real hardware inference benchmark (`evaluation/benchmark.py`) achieving ~23.9 FPS on CPU with YOLO.
- [x] Comprehensive documentation in `docs/MODEL_RESULTS.md` detailing dataset distributions.

### Phase 5: Direct Triple Riding & Multi-Violation Model Integration (COMPLETED)
- [x] Acquired 6,057-image annotated traffic violation dataset (`05_triple_riding`).
- [x] Built Google Colab T4 GPU automated training pipeline (`training/train_triple_riding_colab.ipynb`).
- [x] Trained 5-class custom model achieving **92.3% mAP50** on `MORE_THAN_TWO_PERSONS`, **83.5%** on `WITH_HELMET`, and **71.5%** on `USING_MOBILE`.
- [x] Integrated `models/traffic_custom.pt` into detection pipeline with dual-mode evaluation (Direct model + Spatial heuristic fallback).
- [x] Enhanced citizen proof upload module to run real-time AI detection and bounding box rendering on photos as well as videos.

### Phase 6: Remaining Violations & Final Polish (UPCOMING TODO)
- [ ] **TODO 1: Implement Illegal Parking Rule (`ILLEGAL_PARKING`)**:
  - Virtual "No-Parking" polygon calibration in `camera_calibration.py`.
  - Vehicle stationary dwell-time tracking exceeding threshold ($T > 30\text{ s}$).
- [ ] **TODO 2: Implement Unsafe Movement / Speed Estimation Rule (`UNSAFE_MOVEMENT`)**:
  - Homography transformation (pixels $\rightarrow$ meters).
  - Velocity estimation ($v = \Delta d / \Delta t$) and zigzag trajectory anomaly detection.
- [ ] **TODO 3: Automated Traffic Signal Light Classifier**:
  - Train lightweight classifier on `datasets/raw/03_traffic_light_state/` (1,488 images) for dynamic signal state.
- [ ] **TODO 4: Official Prototype Notice PDF Export**:
  - Downloadable e-challan PDF with QR verification code and violation breakdown.
