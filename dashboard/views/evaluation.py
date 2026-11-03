"""Evaluation: trained accuracy of each model on our dataset, and speed benchmarks. Model comparison lives on the Models page."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from algorithms.detection import model_registry as mr
from dashboard import ui


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _model_comparison():
    with ui.panel():
        st.markdown("### Which detection model?")
        st.markdown('<div class="small">Review 1 planned YOLOv8; the project moved to YOLO26 (January 2026). The '
                    '<b>Models</b> page lists YOLO26, YOLO11, YOLOv10, YOLOv8 and YOLOv5 with their published accuracy '
                    'and speed, compares any of them on our own video, and sets the live model.</div>', unsafe_allow_html=True)
        st.button("Open the Models page", on_click=lambda: st.session_state.update(nav="Models"))


def _accuracy():
    with ui.panel():
        st.markdown("### Accuracy after training on our dataset (helmet, no helmet, number plate)")
        found = {}
        for p in sorted(Path("data").glob("evaluation_results_*.json")):
            data = _load(p)
            if data:
                found[p.stem.replace("evaluation_results_", "")] = data
        if not found and Path("data/evaluation_results.json").exists():
            data = _load(Path("data/evaluation_results.json"))
            if data:
                found[mr.key_for_path(data.get("model", "yolo26n"))] = data
        if not found:
            st.markdown('<div class="note-amber"><b>Not trained yet.</b> The Colab notebook '
                        '(<code>training/train_colab.ipynb</code>) trains the models listed in <code>MODELS_TO_TRAIN</code> '
                        '(YOLO26n and YOLOv8n by default) on the same dataset. Copy each '
                        '<b>evaluation_results_&lt;model&gt;.json</b> from Google Drive (traffic_model_output) into '
                        '<b>data/</b> and they appear here side by side.</div>', unsafe_allow_html=True)
            return
        rows = []
        for key, data in found.items():
            o = data.get("overall") or {}
            p, r = o.get("precision"), o.get("recall")
            f1 = 2 * p * r / (p + r) if isinstance(p, (int, float)) and isinstance(r, (int, float)) and p + r else None
            rows.append({"Model": mr.short(key), "Precision": p, "Recall": r, "F1": f1,
                         "mAP@0.50": o.get("mAP50"), "mAP@0.50:0.95": o.get("mAP50_95")})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in df.columns if c != "Model"})
        for key, data in found.items():
            per = data.get("per_class") or {}
            if isinstance(per, dict) and per:
                with st.expander(f"Per-class results · {mr.short(key)}"):
                    st.dataframe(pd.DataFrame.from_dict(per, orient="index").reset_index().rename(columns={"index": "Class"}),
                                 hide_index=True, use_container_width=True)


def _benchmark():
    with ui.panel():
        st.markdown("### Hardware speed benchmark (detection + tracking only)")
        rows = {}
        for p in sorted(Path("data").glob("benchmark_results_*.json")):
            b = _load(p)
            if b:
                rows[p.stem.replace("benchmark_results_", "")] = b
        if not rows and Path("data/benchmark_results.json").exists():
            b = _load(Path("data/benchmark_results.json")) or {}
            rows[mr.key_for_path(b.get("model", "yolo26n"))] = b
        if not rows:
            st.markdown('<div class="note-blue">No benchmark yet. Run <code>python evaluation/benchmark.py --model all '
                        '--frames 50 --size 640</code> and refresh.</div>', unsafe_allow_html=True)
            return
        body = "".join(
            f'<tr><td><b>{ui.esc(mr.short(k))}</b></td><td>{ui.esc(str(b.get("device", "?")).upper())}</td>'
            f'<td>{b.get("average_fps")} FPS</td><td>{b.get("mean_latency_ms")} ms</td><td>{b.get("p95_latency_ms")} ms</td>'
            f'<td>{ui.esc(b.get("input_resolution", "?"))}</td><td>{ui.esc(b.get("iterations", "?"))}</td>'
            f'<td class="small">{ui.esc(b.get("timestamp", ""))}</td></tr>' for k, b in rows.items())
        st.markdown('<table class="data"><tr><th>Model</th><th>Device</th><th>Average speed</th><th>Mean latency</th>'
                    f'<th>P95 latency</th><th>Input</th><th>Frames</th><th>Measured</th></tr>{body}</table>',
                    unsafe_allow_html=True)
        st.markdown('<div class="small">Synthetic frames, vehicle/person model + tracker only. '
                    '<code>python evaluation/benchmark.py --model all</code> times every downloaded model.</div>',
                    unsafe_allow_html=True)


def render():
    ui.page_header("Evaluation", "Which detection model, how accurate and how fast")
    _model_comparison()
    _accuracy()
    _benchmark()
