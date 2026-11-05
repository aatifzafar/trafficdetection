"""Models: pick detection models from a list, compare them on the same video, and choose which one goes live."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.calibration.camera_calibration import load_config_fresh, save_app_settings
from algorithms.detection import model_registry as mr
from backend.config.settings import settings
from backend.db_models.models import CameraSource
from dashboard import ui
from dashboard.data import session

COMPARE = Path("data/model_comparison.json")
HISTORY = Path("data/model_measurements.json")
PALETTE = ["#1F4E8C", "#E0A100", "#2E7D4F", "#C62828", "#7FA3CF", "#5E6975"]
LAYOUT = dict(paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=36, b=10),
              font=dict(family="Barlow, sans-serif", color="#5E6975", size=12))
MAX_COMPARE = 4
DEFAULT_SEL = ("yolo26n", "yolov8n")


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _live_key() -> str:
    return mr.key_for_path((load_config_fresh().get("detection") or {}).get("model_traffic", settings.MODEL_TRAFFIC))


def _go_live(key: str):
    err = mr.download(key)
    if err:
        st.session_state.models_msg = ("error", f"Could not download {mr.short(key)}: {err}")
        return
    save_app_settings({"model_traffic": mr.path(key)})
    st.session_state.models_msg = ("ok", f"{mr.short(key)} is now the live model. Live cameras, citizen uploads and "
                                         "run_video.py use it from the next run.")


def _download(key: str):
    err = mr.download(key)
    st.session_state.models_msg = ("error", f"Could not download {mr.short(key)}: {err}") if err else \
        ("ok", f"{mr.short(key)} downloaded to {mr.path(key)}.")


def _videos():
    out = {}
    with session() as db:
        for c in db.query(CameraSource).filter(CameraSource.source_type == "VIDEO_FILE").all():
            if c.source_url and Path(c.source_url).exists():
                out[Path(c.source_url).as_posix()] = f"{c.name} ({Path(c.source_url).name})"
    vdir = Path("data/videos")
    if vdir.exists():
        for p in sorted(vdir.glob("*")):
            if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"} and not p.name.startswith("processed_"):
                out.setdefault(p.as_posix(), p.name)
    return out


def _fmt(v, f):
    return "—" if v is None else f.format(v)


def _catalogue(live, history):
    rows = []
    for k, m in mr.MODELS.items():
        p, h = m["published"], history.get(k) or {}
        rows.append({
            "Compare": k in DEFAULT_SEL,  # starting ticks; the table remembers later changes
            "Model": m["short"] + ("  ● LIVE" if k == live else ""),
            "Year": m["year"],
            "Size": m["size"],
            "mAP50-95 (COCO)": p["mAP50_95"],
            "CPU ms / image": p["cpu_onnx_ms"],
            "GPU ms (as published)": f'{p["gpu_ms"]} ({p["gpu_hw"]})',
            "Params (M)": p["params_m"],
            "FLOPs (B)": p["flops_b"],
            "Our video: detect ms": h.get("detect_ms_mean"),
            "Our video: FPS": h.get("pipeline_fps"),
            "Downloaded": "yes" if mr.is_downloaded(k) else "no",
            "_key": k,
        })
    return pd.DataFrame(rows)


def _tradeoff_chart(live, sel):
    pts = [(k, m) for k, m in mr.MODELS.items() if m["published"]["cpu_onnx_ms"] is not None]
    fams = list(mr.FAMILIES)
    fig = go.Figure()
    for i, fam in enumerate(fams):
        fp = [(k, m) for k, m in pts if m["family"] == fam]
        if not fp:
            continue
        fig.add_scatter(x=[m["published"]["cpu_onnx_ms"] for _, m in fp], y=[m["published"]["mAP50_95"] for _, m in fp],
                        mode="lines+markers+text", name=fam, text=[m["short"] for _, m in fp], textposition="top center",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                        marker=dict(size=[16 if k == live else 12 if k in sel else 8 for k, _ in fp],
                                    line=dict(width=[3 if k == live else 0 for k, _ in fp], color="#18212B")),
                        hovertemplate="%{text}: %{y} mAP, %{x} ms CPU<extra></extra>")
    fig.update_layout(height=330, legend=dict(orientation="h", y=-0.2),
                      title=dict(text="Accuracy vs CPU speed (published, COCO) — up and to the left is better",
                                 font=dict(size=13)), **LAYOUT)
    fig.update_xaxes(title="CPU time per image, ms (lower is faster)", gridcolor="#EEF1F4")
    fig.update_yaxes(title="mAP50-95", gridcolor="#EEF1F4")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _bar(title, values, fmt, lower_better):
    keys = list(values)
    vals = [values[k] for k in keys]
    best = None
    if lower_better is not None and len(set(vals)) > 1:
        best = (min if lower_better else max)(vals)
    fig = go.Figure(go.Bar(x=[mr.short(k) for k in keys], y=vals, marker_color=PALETTE[:len(keys)],
                           text=[fmt.format(v) + (" " if v == best else "") for v in vals],
                           textposition="outside", cliponaxis=False))
    hint = "" if lower_better is None else (" (lower is better)" if lower_better else " (higher is better)")
    fig.update_layout(height=240, showlegend=False, title=dict(text=title + hint, font=dict(size=13)), **LAYOUT)
    fig.update_yaxes(visible=False, rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _results(rep, live):
    res = [r for r in rep["results"] if r["model"] in mr.MODELS]
    if not res:
        return
    st.markdown(f'<div class="small">Last comparison: <b>{ui.esc(Path(rep["video"]).name)}</b> · {res[0]["frames"]} frames per model '
                f'· {ui.esc(rep["device"]).upper()} · {ui.esc(rep["timestamp"])}. Same frames, same pipeline and settings — '
                'only the detection model changes. Practice mode: nothing was saved as a violation.</div>',
                unsafe_allow_html=True)
    metrics = [("Detection time per frame", "detect_ms_mean", "{:.1f} ms", True),
               ("Detection time, slowest 5% of frames", "detect_ms_p95", "{:.1f} ms", True),
               ("Full pipeline speed", "pipeline_fps", "{:.2f} FPS", False),
               ("Published accuracy (COCO mAP50-95)", None, "{:.1f}", False),
               ("Vehicles found per frame", "vehicles_per_frame", "{:.2f}", None),
               ("People found per frame", "people_per_frame", "{:.2f}", None),
               ("Mean confidence of found objects", "mean_confidence", "{:.2f}", None),
               ("Vehicles tracked (unique IDs)", "vehicles_tracked", "{}", None),
               ("Violations the rules would flag", "violations_flagged", "{}", None)]
    head = "".join(f"<th>{ui.esc(mr.short(r['model']))}{' ● live' if r['model'] == live else ''}</th>" for r in res)
    body = ""
    for name, k, fmt, lower in metrics:
        vals = [(r["model"], mr.MODELS[r["model"]]["published"]["mAP50_95"] if k is None else r[k]) for r in res]
        nums = [v for _, v in vals if v is not None]
        best = (min if lower else max)(nums) if lower is not None and len(set(nums)) > 1 else None
        body += f"<tr><td>{name}</td>" + "".join(
            f'<td style="font-weight:{700 if v == best else 400}">{_fmt(v, fmt)}{" " if v == best else ""}</td>'
            for _, v in vals) + "</tr>"
    st.markdown(f'<table class="data"><tr><th>Measured</th>{head}</tr>{body}</table>', unsafe_allow_html=True)
    g = st.columns(3)
    with g[0]:
        _bar("Detection time per frame (ms)", {r["model"]: r["detect_ms_mean"] for r in res}, "{:.1f}", True)
    with g[1]:
        _bar("Full pipeline speed (FPS)", {r["model"]: r["pipeline_fps"] for r in res}, "{:.2f}", False)
    with g[2]:
        _bar("Vehicles found per frame", {r["model"]: r["vehicles_per_frame"] for r in res}, "{:.2f}", None)
    frames = [r for r in res if r.get("sample_frame") and Path(r["sample_frame"]).exists()]
    if frames:
        st.markdown(f"**Same frame (#{frames[0]['sample_frame_index']}) from each model**")
        for i in range(0, len(frames), 2):
            cols = st.columns(2)
            for col, r in zip(cols, frames[i:i + 2]):
                col.image(r["sample_frame"], caption=mr.label(r["model"]), use_container_width=True)
    st.markdown("**Choose the live model from this comparison**")
    bcols = st.columns(len(res))
    for col, r in zip(bcols, res):
        if r["model"] == live:
            col.markdown(f'<div class="tag t-go" style="justify-content:center;width:100%">{ui.esc(mr.short(r["model"]))} is live</div>',
                         unsafe_allow_html=True)
        else:
            col.button(f"Go live with {mr.short(r['model'])}", key=f"golive_{r['model']}", use_container_width=True,
                       on_click=_go_live, args=(r["model"],))
    st.markdown('<div class="small">Detection counts and confidence show how the pretrained models see this video; they are not '
                'accuracy (that needs labelled ground truth — see System → Accuracy). Timings depend on the computer.</div>',
                unsafe_allow_html=True)


def render():
    live = _live_key()
    history = _load(HISTORY) or {}
    rep = _load(COMPARE)
    if rep:  # older comparisons (before the history file existed) still fill the table
        for r in rep.get("results", []):
            history.setdefault(r["model"], r)

    head, cur = st.columns([2.2, 1.4], vertical_alignment="center")
    with head:
        ui.page_header("AI models", "Pick models from the list, compare them on the same video, then choose the live model")
    with cur:
        lm = mr.MODELS.get(live)
        name = lm["short"] if lm else live
        detail = f'{lm["published"]["mAP50_95"]} mAP · {lm["family"]} · {lm["year"]}' if lm else mr.path(live)
        st.markdown(f'<div class="stat"><span>Live model now</span><b>{ui.esc(name)}</b>'
                    f'<span> · {ui.esc(detail)}</span></div>', unsafe_allow_html=True)
    msg = st.session_state.pop("models_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])

    # ---------- catalogue ----------
    with ui.panel():
        st.markdown("### 1. Model list")
        st.markdown('<div class="small">Tick up to four models in the <b>Compare</b> column. Published numbers are from the '
                    'Ultralytics model pages (COCO, 640 px). "Our video" columns fill in after you run a comparison.</div>',
                    unsafe_allow_html=True)
        df = _catalogue(live, history)
        edited = st.data_editor(
            df.drop(columns=["_key"]), hide_index=True, use_container_width=True, key="models_table",
            height=36 * (len(df) + 1) + 3,
            disabled=[c for c in df.columns if c not in ("Compare",)],
            column_config={
                "Compare": st.column_config.CheckboxColumn("Compare", width="small"),
                "mAP50-95 (COCO)": st.column_config.NumberColumn(format="%.1f"),
                "CPU ms / image": st.column_config.NumberColumn(format="%.1f"),
                "Our video: detect ms": st.column_config.NumberColumn(format="%.1f"),
                "Our video: FPS": st.column_config.NumberColumn(format="%.2f"),
            })
        chosen = [df["_key"][i] for i, v in enumerate(edited["Compare"]) if v]
        if len(chosen) > MAX_COMPARE:
            st.warning(f"Pick at most {MAX_COMPARE} models; only the first {MAX_COMPARE} ticked are compared.")
            chosen = chosen[:MAX_COMPARE]
        st.session_state.models_sel = chosen
        _tradeoff_chart(live, chosen)
        with st.expander("What is different about each family"):
            for fam, f in mr.FAMILIES.items():
                st.markdown(f"**{fam} ({f['year']})** — {f['note']} [source]({mr.SOURCES[fam]})")
            st.markdown('<div class="small">GPU times were published on different GPUs (T4 for YOLO26 / YOLO11 / YOLOv10, '
                        'A100 for YOLOv8 / YOLOv5), so compare GPU times only within the same GPU. Our cameras run on CPU or '
                        'edge devices, so the CPU column matters most.</div>', unsafe_allow_html=True)

    # ---------- compare ----------
    with ui.panel():
        st.markdown("### 2. Compare the ticked models on our video")
        if len(chosen) < 2:
            st.markdown('<div class="note-blue">Tick at least two models in the list above.</div>', unsafe_allow_html=True)
        else:
            st.markdown("Selected: " + " ".join(f'<span class="tag t-grey">{ui.esc(mr.short(k))}'
                                               f'{"" if mr.is_downloaded(k) else " · will download"}</span>' for k in chosen),
                        unsafe_allow_html=True)
            vids = _videos()
            if not vids:
                st.markdown('<div class="note-amber">No video found in data/videos/.</div>', unsafe_allow_html=True)
            else:
                a, b, c = st.columns([2.2, 1, 1.1], vertical_alignment="bottom")
                pref = next((v for v in vids if "traffic_junction" in v), next(iter(vids)))
                video = a.selectbox("Video", list(vids), index=list(vids).index(pref), format_func=vids.get)
                frames = b.number_input("Frames per model", 30, 1000, 150, step=30)
                if c.button(f"Compare {len(chosen)} models", type="primary", use_container_width=True):
                    from evaluation.compare_models import run_comparison
                    bar = st.progress(0.0, "Starting…")
                    try:
                        rep = run_comparison(video, int(frames), models=chosen,
                                             progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m))
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")
                    bar.empty()
                st.markdown('<div class="small">About 1 minute per model for 150 frames on a laptop CPU. '
                            'Models that are not downloaded yet are fetched first (needs internet).</div>',
                            unsafe_allow_html=True)
        if rep:
            _results(rep, live)

    # ---------- go live ----------
    with ui.panel():
        st.markdown("### 3. Choose the model to go live")
        a, b, c = st.columns([2.2, 1, 1], vertical_alignment="bottom")
        keys = list(mr.MODELS)
        pick = a.selectbox("Model", keys, index=keys.index(live) if live in keys else 0, format_func=mr.label,
                           key="golive_pick")
        b.button("Download", use_container_width=True, disabled=mr.is_downloaded(pick), on_click=_download, args=(pick,))
        c.button("Set as live model", type="primary", use_container_width=True, disabled=pick == live,
                 on_click=_go_live, args=(pick,))
        m = mr.MODELS[pick]
        st.markdown(f'<div class="small" style="margin-top:6px">{ui.esc(m["note"])} '
                    f'Published: {m["published"]["mAP50_95"]} mAP · '
                    f'{_fmt(m["published"]["cpu_onnx_ms"], "{} ms")} CPU · {m["published"]["params_m"]} M parameters. '
                    'The choice is saved in data/app_settings.yaml (same as System → Settings). Medium (m) models are '
                    'more accurate but several times slower on a laptop CPU.</div>', unsafe_allow_html=True)
