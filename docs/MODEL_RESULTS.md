# Model Evaluation & Performance Results

## 1. Merged Training & Validation Dataset

The training dataset was constructed by merging and harmonizing multiple raw datasets (Pascal VOC XML and Roboflow YOLO formats) from `datasets/raw/`. Exact duplicates across collections (such as between HelmetViolations v1 and v2) were removed using SHA-256 byte hashing. All video frames from sequential sources (archive 9) were strictly grouped into the training partition to prevent cross-split data leakage.

### Dataset Overview
- **Total Unique Images**: 4,016
- **Duplicate Images Filtered**: 1,440
- **Train Split**: 3,213 images (80%)
- **Validation Split**: 803 images (20%)
- **Target Classes**: 3 (`0: helmet`, `1: no_helmet`, `2: number_plate`)

### Box Distribution by Class & Split

| Class ID | Class Name | Train Annotations | Validation Annotations | Total Annotations |
| :--- | :--- | :--- | :--- | :--- |
| **0** | `helmet` | 932 | 201 | 1,133 |
| **1** | `no_helmet` | 584 | 152 | 736 |
| **2** | `number_plate` | 2,844 | 732 | 3,576 |
| **Total** | **All Classes** | **4,360** | **1,085** | **5,445** |

---

## 2. Triple Riding & Multi-Class Violation Dataset (`05_triple_riding`)

To directly detect triple riding and rider mobile phone usage without relying exclusively on geometric heuristics, a dedicated dataset of **6,057 images** was integrated (`Prototype/datasets/raw/05_triple_riding`):

- **Training Split**: 5,295 images
- **Validation Split**: 514 images
- **Test Split**: 248 images
- **Classes**:
  - `0: MORE_THAN_TWO_PERSONS` (Triple Riding Violation)
  - `1: USING_MOBILE` (Mobile Phone while Riding)
  - `2: WITHOUT_HELMET` (Helmet Non-Compliance)
  - `3: WITH_HELMET` (Compliant Helmet Rider)
  - `4: normal` (Normal Two-Wheeler Traffic)

---

## 3. Real Training Run Metrics (Google Colab Tesla T4 GPU)

Trained for 30 epochs on Tesla T4 GPU (0.60 hours / 36 minutes total training time):

| Class | Precision ($P$) | Recall ($R$) | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ | Primary Violation Mapped |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`MORE_THAN_TWO_PERSONS`** | **85.6%** | **86.4%** | **92.3%** | 53.8% | **Triple Riding** |
| **`WITH_HELMET`** | **84.0%** | **75.7%** | **83.5%** | 46.6% | Helmet Compliance |
| **`USING_MOBILE`** | **77.0%** | **63.1%** | **71.5%** | 37.7% | Mobile Phone Use |
| **`WITHOUT_HELMET`** | **64.9%** | **55.7%** | **61.8%** | 32.0% | **No Helmet** |
| **`normal`** | **82.9%** | **63.6%** | **67.5%** | 31.6% | Standard Traffic |
| **All Classes (Mean)** | **78.9%** | **68.9%** | **75.3%** | **40.4%** | — |

- **Inference Speed:** **3.9 ms** per frame on GPU (~256 FPS), **~52 ms** on local CPU (~19 FPS).
- **Model Parameters:** 3,133,679 parameters across 73 fused layers (8.6 GFLOPs).
- **Active Weights File:** `Prototype/models/traffic_custom.pt`

---

## 4. Hardware Inference Benchmark (Local System)

Measured with `evaluation/benchmark.py` running 50 iterations at 640x640 resolution with ByteTrack:

- **Primary Traffic Model**: `yolo26n.pt` (NMS-free)
- **Inference Hardware**: Local CPU (`win32`, Python 3.13)
- **Average Inference Speed**: **23.87 FPS**
- **Mean Pipeline Latency**: **39.11 ms**
- **Median Latency**: **38.46 ms**
- **95th Percentile Latency**: **45.54 ms**
- **Tracker**: Ultralytics Native ByteTrack

---

## 5. Honest Known Limitations & Future Work

1. **Night & Low-Light Conditions**:
   - The current dataset contains predominantly daytime and well-lit outdoor footage. Performance under rainy, foggy, or night-time headlight glare may experience increased false negatives.
2. **Triple-Riding Detection Status**:
   - **Upgraded:** Now features direct deep learning object detection via the `MORE_THAN_TWO_PERSONS` model class (**92.3% mAP50**). The system retains the geometric spatial rider-motorcycle clustering heuristic as an automated fallback if the custom model is not loaded.
3. **Plate OCR on Low-Resolution CCTV Feeds**:
   - While license plate bounding boxes are localized effectively, reading high-accuracy alphanumeric characters via Tesseract / EasyOCR requires at least 100x30 pixels. In wide-angle panoramic CCTV cameras where vehicles are distant, plate crops are often 20x10 pixels and unreadable without super-resolution preprocessing.
4. **Extreme Angles & Oblique Perspectives**:
   - Riders captured from steep overhead angles may partially occlude helmet silhouettes or face masks; resolved via multi-frame consensus tracking.
