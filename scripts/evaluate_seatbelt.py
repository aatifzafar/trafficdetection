"""
Evaluate trained seatbelt model on Validation and Held-Out Test splits separately.
Generates data/evaluation_results_seatbelt.json matching existing project format.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_seatbelt")


def evaluate_seatbelt(
    model_path: str = "models/seatbelt_best.pt",
    data_yaml: str = "data/seatbelt/data.yaml",
    img_size: int = 640,
    output_json: str = "data/evaluation_results_seatbelt.json",
):
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_file

    if not model_file.exists():
        logger.error(f"Weights file not found: {model_file}")
        return False

    yaml_file = Path(data_yaml)
    if not yaml_file.is_absolute():
        yaml_file = PROJECT_ROOT / yaml_file

    if not yaml_file.exists():
        logger.error(f"Dataset config not found: {yaml_file}")
        return False

    from ultralytics import YOLO

    logger.info(f"Loading seatbelt model: {model_file}")
    model = YOLO(str(model_file))

    # 1. Validation split
    logger.info("Evaluating on VALIDATION split...")
    val_res = model.val(data=str(yaml_file), split="val", imgsz=img_size, verbose=False)

    # 2. Test split
    logger.info("Evaluating on TEST split...")
    test_res = model.val(data=str(yaml_file), split="test", imgsz=img_size, verbose=False)

    def extract_metrics(res):
        mp = float(res.box.mp) if hasattr(res.box, "mp") else 0.0
        mr = float(res.box.mr) if hasattr(res.box, "mr") else 0.0
        map50 = float(res.box.map50) if hasattr(res.box, "map50") else 0.0
        map_all = float(res.box.map) if hasattr(res.box, "map") else 0.0
        f1_overall = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

        p_list = res.box.p if hasattr(res.box, "p") and len(res.box.p) > 0 else [0.0, 0.0]
        r_list = res.box.r if hasattr(res.box, "r") and len(res.box.r) > 0 else [0.0, 0.0]
        ap50_list = res.box.ap50 if hasattr(res.box, "ap50") and len(res.box.ap50) > 0 else [0.0, 0.0]
        ap_list = res.box.ap if hasattr(res.box, "ap") and len(res.box.ap) > 0 else [0.0, 0.0]

        classes_data = {}
        class_names = ["seatbelt", "no_seatbelt"]
        for idx, name in enumerate(class_names):
            p = float(p_list[idx]) if idx < len(p_list) else 0.0
            r = float(r_list[idx]) if idx < len(r_list) else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            a50 = float(ap50_list[idx]) if idx < len(ap50_list) else 0.0
            a95 = float(ap_list[idx]) if idx < len(ap_list) else 0.0
            classes_data[name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "F1": round(f1, 4),
                "mAP50": round(a50, 4),
                "mAP50_95": round(a95, 4),
            }

        return {
            "overall": {
                "precision": round(mp, 4),
                "recall": round(mr, 4),
                "F1": round(f1_overall, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map_all, 4),
            },
            "per_class": classes_data,
        }

    val_data = extract_metrics(val_res)
    test_data = extract_metrics(test_res)

    results_dict = {
        "model": "seatbelt_best.pt",
        "architecture": "YOLO26n / YOLOv8n fine-tuned on Seatbelt Dataset",
        "overall": val_data["overall"],
        "per_class": val_data["per_class"],
        "test": {
            "overall": test_data["overall"],
            "per_class": test_data["per_class"],
        },
        "split_counts": {
            "train": 3101,
            "val": 380,
            "test": 368,
        },
        "limitations": "Trained primarily on front-cabin windshield driver/passenger crops under standard illumination conditions. Real-world CCTV junction footage is frequently wide-angle and mounted at high elevation angles with lower effective resolution (sub-720p). Expect lower precision and recall on extreme wide-angle junctions without optical clarity; hence the pipeline strictly enforces windshield minimum crop dimensions (120x120px) and a Laplacian blur threshold prior to violation triggering.",
    }

    out_file = Path(output_json)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Evaluation results successfully saved to {out_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Seatbelt YOLO model on Val and Test splits")
    parser.add_argument("--model", type=str, default="models/seatbelt_best.pt", help="Path to seatbelt weights")
    parser.add_argument("--data", type=str, default="data/seatbelt/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--output", type=str, default="data/evaluation_results_seatbelt.json", help="Output JSON")

    args = parser.parse_args()
    evaluate_seatbelt(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.imgsz,
        output_json=args.output,
    )

