"""
compare_models.py — run detection models through the same pipeline on the same video and save the numbers.

    python evaluation/compare_models.py --video data/videos/traffic_junction_cctv.mp4 --frames 150
    python evaluation/compare_models.py --models yolo26n yolo11n yolov8n yolov5nu

Default models: YOLO26n (project) and YOLOv8n (Review 1 plan). Any key from
algorithms/detection/model_registry.py works; missing weights are downloaded first (needs internet).

Writes data/model_comparison.json and one annotated sample frame per model in data/model_comparison/.
The dashboard's Evaluation page shows the result. Nothing is written to the violations database
(the pipeline runs in practice mode).

What is measured, per model, on this computer:
  - detection time per frame (the YOLO model only), mean and 95th percentile
  - full pipeline speed (detection + tracking + plate reading + rules), frames per second
  - objects found per frame, split into vehicles and people, and their mean confidence
  - frames where at least one vehicle was found
  - vehicles tracked (unique track IDs) and violations the rules would have flagged
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from algorithms.detection import model_registry as mr

OUT_JSON = Path("data/model_comparison.json")
HISTORY_JSON = Path("data/model_measurements.json")  # latest measurement of every model ever compared
OUT_DIR = Path("data/model_comparison")
VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle", "auto", "autorickshaw"}


def _run_one(key: str, video: str, frames: int, signal: str, progress: Optional[Callable[[float, str], None]],
             step: float, base: float) -> Dict:
    from pipeline.traffic_pipeline import TrafficVisionPipeline

    pipe = TrafficVisionPipeline(location="Model comparison", persist=False, model_traffic=mr.path(key))
    det = pipe.detector
    per_frame = {"ms": [], "veh": [], "ppl": [], "conf": []}
    orig = det.detect

    def timed(frame):
        t0 = time.perf_counter()
        out = orig(frame)
        per_frame["ms"].append((time.perf_counter() - t0) * 1000.0)
        per_frame["veh"].append(sum(1 for d in out if d.class_name in VEHICLES))
        per_frame["ppl"].append(sum(1 for d in out if d.class_name == "person"))
        per_frame["conf"].extend(d.confidence for d in out if d.class_name in VEHICLES or d.class_name == "person")
        return out

    det.detect = timed
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = min(frames, total) if total else frames
    sample_at = max(1, limit // 2)
    n, t_pipe = 0, 0.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = OUT_DIR / f"{key}.jpg"
    # warm-up so the first-frame model load does not count against either model
    ok, first = cap.read()
    if ok:
        orig(first)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            t0 = time.perf_counter()
            annotated, _ = pipe.process_frame(frame, n, fps=fps, signal_state=signal)
            t_pipe += time.perf_counter() - t0
            if n == sample_at:
                cv2.imwrite(str(sample_path), annotated)
            if progress and n % 5 == 0:
                progress(base + step * n / max(1, limit), f"{mr.short(key)}: frame {n} of {limit}")
    finally:
        cap.release()
    ms = np.array(per_frame["ms"] or [0.0])
    veh = np.array(per_frame["veh"] or [0])
    return {
        "model": key,
        "label": mr.label(key),
        "weights": mr.path(key),
        "frames": n,
        "detect_ms_mean": round(float(ms.mean()), 1),
        "detect_ms_p95": round(float(np.percentile(ms, 95)), 1),
        "pipeline_fps": round(n / t_pipe, 2) if t_pipe else 0.0,
        "vehicles_per_frame": round(float(veh.mean()), 2),
        "people_per_frame": round(float(np.mean(per_frame["ppl"] or [0])), 2),
        "mean_confidence": round(float(np.mean(per_frame["conf"])), 3) if per_frame["conf"] else 0.0,
        "frames_with_vehicle_pct": round(100.0 * float((veh > 0).mean()), 1),
        "vehicles_tracked": len(pipe.total_vehicles_seen),
        "violations_flagged": int(sum(pipe.event_counts.values())),
        "violations_by_type": dict(pipe.event_counts),
        "sample_frame": str(sample_path) if sample_path.exists() else None,
        "sample_frame_index": sample_at,
    }


def run_comparison(video: str, frames: int = 150, models: List[str] = ("yolo26n", "yolov8n"), signal: str = "GREEN",
                   progress: Optional[Callable[[float, str], None]] = None) -> Dict:
    if not Path(video).exists():
        raise FileNotFoundError(video)
    models = list(models)
    for key in models:
        if progress:
            progress(0.0, f"Getting {mr.short(key)} weights…")
        err = mr.download(key) if key in mr.MODELS else None
        if err:
            raise RuntimeError(f"Could not download {mr.short(key)}: {err}")
    results = []
    step = 1.0 / len(models)
    for i, key in enumerate(models):
        results.append(_run_one(key, video, frames, signal, progress, step, i * step))
    import torch  # noqa: WPS433 (only for the device name)
    report = {
        "video": video,
        "frames_requested": frames,
        "signal": signal,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    history = {}
    if HISTORY_JSON.exists():
        try:
            history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        except Exception:
            history = {}
    for r in results:
        history[r["model"]] = {**r, "video": video, "device": report["device"], "timestamp": report["timestamp"]}
    HISTORY_JSON.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description="Compare YOLO26n and YOLOv8n on the same video")
    ap.add_argument("--video", default="data/videos/22.mp4")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--signal", default="GREEN", choices=["RED", "GREEN", "YELLOW", "AUTO"])
    ap.add_argument("--models", nargs="+", default=["yolo26n", "yolov8n"], choices=list(mr.MODELS),
                    help="Models to compare, e.g. --models yolo26n yolo11n yolov8n")
    args = ap.parse_args()
    rep = run_comparison(args.video, args.frames, models=args.models, signal=args.signal,
                         progress=lambda f, m: print(f"\r{m} ({f:.0%})", end="", flush=True))
    print()
    keys = ["detect_ms_mean", "detect_ms_p95", "pipeline_fps", "vehicles_per_frame", "people_per_frame",
            "mean_confidence", "frames_with_vehicle_pct", "vehicles_tracked", "violations_flagged"]
    print(f"{'metric':28}" + "".join(f"{r['model']:>12}" for r in rep["results"]))
    for k in keys:
        print(f"{k:28}" + "".join(f"{r[k]:>12}" for r in rep["results"]))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
