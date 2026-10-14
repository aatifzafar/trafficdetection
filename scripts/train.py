import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def train_model(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    project_name: str = "traffic_violation_train",
):
    """
    Fine-tunes YOLOv8 on custom Indian traffic or helmet datasets.
    """
    if not Path(data_yaml).exists():
        logger.error(f"Dataset config '{data_yaml}' not found. Please provide a valid data.yaml path.")
        return

    from ultralytics import YOLO

    logger.info(f"Loading base model: {base_model}")
    model = YOLO(base_model)

    logger.info(f"Starting training on {data_yaml} for {epochs} epochs (batch={batch_size}, img_size={img_size})...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=settings.DEVICE,
        project=project_name,
        name="run_1",
    )
    logger.info("Training finished. Check runs/ for model weights and metrics.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train custom YOLO model for traffic violation detection")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="Epoch count")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_model(args.data, base_model=args.model, epochs=args.epochs, batch_size=args.batch)
