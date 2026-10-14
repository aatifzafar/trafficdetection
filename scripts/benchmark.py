import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings
from vision.detector.detector import TrafficDetector
from vision.tracker import create_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def run_benchmark(num_frames: int = 100, img_size: int = 640):
    logger.info("Initializing benchmark on device: %s", settings.DEVICE)
    detector = TrafficDetector(
        model_traffic_path=settings.MODEL_TRAFFIC,
        confidence=settings.CONFIDENCE,
        iou=settings.IOU,
        device=settings.DEVICE,
    )
    tracker = create_tracker(algorithm=settings.TRACKER_ALGO)

    # Warmup
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    for _ in range(5):
        _ = detector.detect(dummy)

    logger.info("Running %d iterations at %dx%d resolution...", num_frames, img_size, img_size)

    latencies = []
    start_total = time.time()

    for i in range(num_frames):
        # Generate varied synthetic frame
        frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        t0 = time.time()
        dets = detector.detect(frame)
        _ = tracker.update(frame, dets, i)
        t1 = time.time()

        latencies.append((t1 - t0) * 1000.0)

    total_time = time.time() - start_total
    avg_fps = num_frames / total_time
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    results = {
        "device": detector.device,
        "model": settings.MODEL_TRAFFIC,
        "tracker": settings.TRACKER_ALGO,
        "input_resolution": f"{img_size}x{img_size}",
        "iterations": num_frames,
        "total_time_sec": round(total_time, 2),
        "average_fps": round(avg_fps, 2),
        "mean_latency_ms": round(float(mean_lat), 2),
        "median_latency_ms": round(float(median_lat), 2),
        "p95_latency_ms": round(float(p95_lat), 2),
        "min_latency_ms": round(float(min_lat), 2),
        "max_latency_ms": round(float(max_lat), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = Path("data/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("        SURVEILLANCE AI HARDWARE BENCHMARK        ")
    print("=" * 50)
    print(f"Device               : {results['device']}")
    print(f"Model                : {results['model']}")
    print(f"Tracker              : {results['tracker']}")
    print(f"Frames Evaluated     : {results['iterations']}")
    print(f"Average FPS          : {results['average_fps']} FPS")
    print(f"Mean Latency         : {results['mean_latency_ms']} ms")
    print(f"Median Latency       : {results['median_latency_ms']} ms")
    print(f"95th Percentile Lat  : {results['p95_latency_ms']} ms")
    print("=" * 50)
    print(f"Results saved to: {out_path}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Traffic Surveillance Inference Benchmark")
    parser.add_argument("--frames", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--size", type=int, default=640, help="Input frame dimension")
    args = parser.parse_args()

    run_benchmark(num_frames=args.frames, img_size=args.size)
