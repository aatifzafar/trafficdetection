"""
Seatbelt Model Testing and Validation Dashboard View.
Allows uploading single/multiple images or a ZIP file (up to 3,000 images),
or loading the held-out test samples in data/seatbelt/test_samples/.
Features streaming batch inference, memory management, color-coded gallery with pagination,
summary KPIs, filterable data table, and CSV export.
"""

import io
import os
import gc
import csv
import json
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "seatbelt_best.pt",
    PROJECT_ROOT / "models" / "seatbelt_model.pt",
    PROJECT_ROOT / "models" / "traffic_seatbelt.pt",
]

_CACHED_MODEL = None


def load_seatbelt_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    for p in MODEL_CANDIDATES:
        if p.exists() and p.is_file():
            try:
                from ultralytics import YOLO
                _CACHED_MODEL = YOLO(str(p))
                return _CACHED_MODEL
            except Exception as e:
                st.warning(f"Failed to load model {p.name}: {e}")
    return None


def get_evaluation_metrics() -> Optional[Dict[str, Any]]:
    eval_file = PROJECT_ROOT / "data" / "evaluation_results_seatbelt.json"
    if eval_file.exists():
        try:
            return json.loads(eval_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def annotate_image(image: Image.Image, boxes) -> Image.Image:
    """Draw bounding boxes: Green for seatbelt (0), Red for no_seatbelt (1)."""
    np_img = np.array(image.convert("RGB"))
    h, w = np_img.shape[:2]

    for b in boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = [int(v) for v in b.xyxy[0].tolist()]
        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])

        if cls_id == 1 or "no" in str(b.cls).lower():
            color = (239, 68, 68)  # Red
            label = f"NO SEATBELT {conf:.2f}"
        else:
            color = (34, 197, 94)  # Green
            label = f"SEATBELT {conf:.2f}"

        cv2.rectangle(np_img, (x1, y1), (x2, y2), color, 2)
        t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(np_img, (x1, max(0, y1 - 20)), (x1 + t_size[0] + 6, max(20, y1)), color, -1)
        cv2.putText(np_img, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(np_img)


def render():
    st.markdown('<div class="brand" style="margin-bottom:4px">🚗 Seatbelt Violation AI Lab</div>', unsafe_allow_html=True)
    st.caption("Experimental Computer Vision Enforcement Module · Two-stage Windshield Region Detection & ALPR Integration")

    # Metrics banner from evaluation report
    eval_metrics = get_evaluation_metrics()
    if eval_metrics:
        val_overall = eval_metrics.get("overall", {})
        val_ns = eval_metrics.get("per_class", {}).get("no_seatbelt", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation mAP@50", f"{val_overall.get('mAP50', 0.0)*100:.1f}%")
        c2.metric("No-Seatbelt Precision", f"{val_ns.get('precision', 0.0)*100:.1f}%")
        c3.metric("No-Seatbelt Recall", f"{val_ns.get('recall', 0.0)*100:.1f}%")
        c4.metric("Status", "EXPERIMENTAL", delta="Human-in-the-loop review", delta_color="normal")
        with st.expander("ℹ️ Architectural & CCTV Real-World Limitations", expanded=False):
            st.info(eval_metrics.get("limitations", "Model trained on front windshield crops."))

    model = load_seatbelt_model()
    if model is None:
        st.warning("⚠️ **Seatbelt detection model not configured.** Running in operational standby mode.\n\n"
                   "To activate real-time detection, place the trained weights file at `models/seatbelt_best.pt` "
                   "or run the training notebook `notebooks/train_seatbelt_colab.ipynb`.")
        return

    # Image source options
    st.markdown("### 1. Select Input Source")
    tab_upload, tab_test_samples = st.tabs(["📤 Upload Images or ZIP Archive", "🧪 Curated Held-Out Test Set"])

    images_to_process = []  # list of (filename, Image.Image or bytes)

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload image(s) or a ZIP archive containing up to 3,000 images:",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(uf) as z:
                        namelist = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and not n.startswith("__MACOSX")][:3000]
                        st.info(f"Extracting {len(namelist)} images from `{uf.name}`...")
                        for name in namelist:
                            try:
                                img_data = z.read(name)
                                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                                images_to_process.append((Path(name).name, img))
                            except Exception:
                                pass
                else:
                    try:
                        img = Image.open(uf).convert("RGB")
                        images_to_process.append((uf.name, img))
                    except Exception:
                        pass

    with tab_test_samples:
        test_samples_dir = PROJECT_ROOT / "data" / "seatbelt" / "test_samples"
        if test_samples_dir.exists():
            manifest_file = test_samples_dir / "manifest.json"
            sample_count = len(list(test_samples_dir.glob("*.jpg")))
            st.write(f"📁 Local curated held-out test split has **{sample_count}** images ready.")
            if st.button("Load All Held-Out Test Samples", type="primary"):
                for img_path in sorted(test_samples_dir.glob("*.jpg")):
                    try:
                        im = Image.open(img_path).convert("RGB")
                        images_to_process.append((img_path.name, im))
                    except Exception:
                        pass
                st.session_state["loaded_test_samples"] = images_to_process

            if "loaded_test_samples" in st.session_state and not images_to_process:
                images_to_process = st.session_state["loaded_test_samples"]
        else:
            st.info("Run `python scripts/make_seatbelt_test_set.py` to populate test samples.")

    if not images_to_process:
        st.info("Please upload images or load the held-out test samples above to test the model.")
        return

    st.write(f"**Loaded {len(images_to_process)} images.**")
    conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)

    if st.button(f"🚀 Run Seatbelt AI Inference on {len(images_to_process)} Images", type="primary"):
        with st.spinner("Processing through YOLO seatbelt detector with streaming batching..."):
            results_records = []
            annotated_images = []
            start_t = time.time()

            total_imgs = len(images_to_process)
            batch_size = 32
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for b_idx in range(0, total_imgs, batch_size):
                batch = images_to_process[b_idx: b_idx + batch_size]
                batch_imgs = [item[1] for item in batch]

                preds = model(batch_imgs, conf=conf_thresh, verbose=False)

                for sub_idx, (fname, img) in enumerate(batch):
                    p = preds[sub_idx] if preds and len(preds) > sub_idx else None
                    boxes = p.boxes if p is not None else []

                    has_no_seatbelt = False
                    has_seatbelt = False
                    max_conf = 0.0

                    for b in boxes:
                        cls_id = int(b.cls[0].item())
                        c = float(b.conf[0].item())
                        if c > max_conf:
                            max_conf = c
                        if cls_id == 1:
                            has_no_seatbelt = True
                        elif cls_id == 0:
                            has_seatbelt = True

                    status = "NO SEATBELT" if has_no_seatbelt else ("SEATBELT" if has_seatbelt else "NO DETECTION")

                    # Keep annotated preview for the first 120 images to prevent browser memory exhaustion
                    ann_img = None
                    if len(annotated_images) < 120:
                        ann_img = annotate_image(img, boxes)

                    results_records.append({
                        "filename": fname,
                        "status": status,
                        "confidence": round(max_conf, 3),
                        "has_violation": has_no_seatbelt,
                        "box_count": len(boxes),
                    })
                    if ann_img is not None:
                        annotated_images.append((fname, ann_img, status, round(max_conf, 3)))

                current_done = min(b_idx + len(batch), total_imgs)
                progress_bar.progress(current_done / total_imgs)
                status_text.text(f"Processed {current_done} of {total_imgs} images...")

                if b_idx % 128 == 0:
                    gc.collect()

            elapsed = time.time() - start_t
            status_text.empty()
            progress_bar.empty()

            st.session_state["seatbelt_results"] = results_records
            st.session_state["seatbelt_ann_images"] = annotated_images
            st.session_state["seatbelt_elapsed"] = elapsed

    # Display results if available
    if "seatbelt_results" in st.session_state:
        records = st.session_state["seatbelt_results"]
        ann_imgs = st.session_state.get("seatbelt_ann_images", [])
        elapsed = st.session_state.get("seatbelt_elapsed", 0.0)

        st.markdown("---")
        st.markdown("### 2. Inference Results & Summary")

        total = len(records)
        ns_count = sum(1 for r in records if r["status"] == "NO SEATBELT")
        s_count = sum(1 for r in records if r["status"] == "SEATBELT")
        no_det = sum(1 for r in records if r["status"] == "NO DETECTION")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Analyzed", total)
        m2.metric("Violations (No Seatbelt)", ns_count, delta=f"{ns_count/total*100:.1f}% rate" if total else "0%")
        m3.metric("Compliant (Wearing Belt)", s_count)
        m4.metric("Inference Time", f"{elapsed:.2f}s", delta=f"{total/elapsed:.1f} FPS" if elapsed > 0 else "")

        # Filters
        st.markdown("#### Filter Gallery & Table")
        filter_opt = st.radio(
            "Show:",
            ["All Results", "Only Violations (NO SEATBELT)", "Only Compliant (SEATBELT)"],
            horizontal=True,
        )

        filtered_records = []
        filtered_imgs = []
        for r in records:
            if filter_opt == "Only Violations (NO SEATBELT)" and r["status"] != "NO SEATBELT":
                continue
            if filter_opt == "Only Compliant (SEATBELT)" and r["status"] != "SEATBELT":
                continue
            filtered_records.append(r)

        for img_tuple in ann_imgs:
            fn, a_img, stt, cf = img_tuple
            if filter_opt == "Only Violations (NO SEATBELT)" and stt != "NO SEATBELT":
                continue
            if filter_opt == "Only Compliant (SEATBELT)" and stt != "SEATBELT":
                continue
            filtered_imgs.append(img_tuple)

        # Download CSV
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=["filename", "status", "confidence", "has_violation", "box_count"])
        writer.writeheader()
        writer.writerows(filtered_records)

        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="seatbelt_detection_results.csv",
            mime="text/csv",
        )

        # Image Gallery with Pagination
        st.markdown(f"**Visual Gallery ({len(filtered_imgs)} previewed images):**")
        if filtered_imgs:
            page_size = 24
            total_pages = max(1, (len(filtered_imgs) + page_size - 1) // page_size)
            page = 1
            if total_pages > 1:
                page = st.number_input("Gallery Page", min_value=1, max_value=total_pages, value=1)
            
            start_p = (page - 1) * page_size
            end_p = min(start_p + page_size, len(filtered_imgs))
            page_slice = filtered_imgs[start_p:end_p]

            cols = st.columns(4)
            for i, (fn, a_img, stt, cf) in enumerate(page_slice):
                col = cols[i % 4]
                tag = "🔴 VIOLATION" if stt == "NO SEATBELT" else ("🟢 COMPLIANT" if stt == "SEATBELT" else "⚪ NONE")
                col.image(a_img, caption=f"{fn}\n{tag} ({cf:.2f})", use_container_width=True)
        else:
            st.info("No preview images match the selected filter.")

        # Summary Table
        st.markdown("#### Detailed Data Table")
        st.dataframe(filtered_records, use_container_width=True)
