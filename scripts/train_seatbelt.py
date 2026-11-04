"""
Train YOLO26 / YOLOv8 model on custom Seatbelt Dataset.
Saves model weights to models/seatbelt_best.pt and results to data/seatbelt/results/.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_seatbelt")


def train_seatbelt(
    data_yaml: str = "data/seatbelt/data.yaml",
    base_model: str = "models/yolo26n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,
    workers: int = 4,
    project_name: str = "runs/detect",
    run_name: str = "seatbelt_custom",
):
    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        logger.error(f"Dataset configuration not found at {yaml_path}")
        return False

    model_path = Path(base_model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    # Fallback to local yolo26n.pt or download
    if not model_path.exists():
        if (PROJECT_ROOT / "models" / "yolo26n.pt").exists():
            model_path = PROJECT_ROOT / "models" / "yolo26n.pt"
        elif (PROJECT_ROOT / "models" / "yolov8n.pt").exists():
            model_path = PROJECT_ROOT / "models" / "yolov8n.pt"
        else:
            model_path = "yolo26n.pt"

    from ultralytics import YOLO

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Target device: {device.upper()} (CUDA available: {torch.cuda.is_available()})")
    logger.info(f"Base model checkpoint: {model_path}")
    logger.info(f"Parameters: epochs={epochs}, batch={batch_size}, imgsz={img_size}, workers={workers}")

    model = YOLO(str(model_path))

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        project=project_name,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else PROJECT_ROOT / project_name / run_name
    best_weights = save_dir / "weights" / "best.pt"
    dest_weights = PROJECT_ROOT / "models" / "seatbelt_best.pt"

    if best_weights.exists():
        dest_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, dest_weights)
        logger.info(f"Deployed best checkpoint -> {dest_weights}")
    else:
        last_weights = save_dir / "weights" / "last.pt"
        if last_weights.exists():
            dest_weights.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(last_weights, dest_weights)
            logger.info(f"Deployed last checkpoint -> {dest_weights}")

    # Archive curves and metrics into data/seatbelt/results/
    out_results_dir = PROJECT_ROOT / "data" / "seatbelt" / "results"
    out_results_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
    ]
    for artifact in artifacts:
        src = save_dir / artifact
        if src.exists():
            shutil.copy2(src, out_results_dir / artifact)
            logger.info(f"Archived training artifact: {artifact}")

    logger.info("Training and artifact archival completed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO seatbelt detection model")
    parser.add_argument("--data", type=str, default="data/seatbelt/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="models/yolo26n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch count")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (e.g. 640 or 832)")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', etc.)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()
    train_seatbelt(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        workers=args.workers,
    )

