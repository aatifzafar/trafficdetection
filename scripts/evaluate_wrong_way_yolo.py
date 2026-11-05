"""
scripts/evaluate_wrong_way_yolo.py
Evaluation script for YOLO Wrong-Way Driving Detection Models.
Evaluates on Validation and Held-Out Test splits separately.
Generates data/evaluation_results_wrong_way_yolo.json matching project format.
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
logger = logging.getLogger("evaluate_wrong_way_yolo")


def evaluate_wrong_way_yolo(
    model_path: str = "models/wrong_way_yolo.pt",
    data_yaml: str = "datasets/wrong_way_roboflow/data.yaml",
    img_size: int = 640,
    output_json: str = "data/evaluation_results_wrong_way_yolo.json",
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

    logger.info(f"Loading YOLO wrong-way model: {model_file}")
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
        class_names = ["right_side", "wrong_side"]
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
        "model": "wrong_way_yolo.pt",
        "architecture": "Ultralytics YOLO (YOLOv8n / YOLO26n) fine-tuned on Wrong-Way Driving Dataset",
        "image_size": img_size,
        "overall": val_data["overall"],
        "per_class": val_data["per_class"],
        "test": {
            "overall": test_data["overall"],
            "per_class": test_data["per_class"],
        },
        "split_counts": {
            "train": 426,
            "val": 91,
            "test": 91,
        },
        "classes": {
            "0": "right-side (Normal compliant traffic flow)",
            "1": "wrong-side (Wrong-Way driving violation counterflow)",
        },
        "limitations": "Model performs visual detection of vehicle heading against designated corridor. For photo uploads, detections are flagged as UNVERIFIED pending review. For video feeds, combined with ByteTrack multi-object tracking and corridor kinematics.",
    }

    out_path = Path(output_json)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"✅ Exported evaluation results to: {out_path}")
    print("\n" + "=" * 65)
    print("           YOLO WRONG-WAY MODEL EVALUATION SUMMARY            ")
    print("=" * 65)
    print(f"Model Checkpoint : {model_file.name}")
    print(f"Validation Split : Precision={val_data['overall']['precision']:.1%}, Recall={val_data['overall']['recall']:.1%}, mAP50={val_data['overall']['mAP50']:.1%}")
    print(f"Held-Out Test    : Precision={test_data['overall']['precision']:.1%}, Recall={test_data['overall']['recall']:.1%}, mAP50={test_data['overall']['mAP50']:.1%}")
    print("-" * 65)
    print(f"  Class 'wrong_side' (Violation) : mAP50={val_data['per_class']['wrong_side']['mAP50']:.1%}, Precision={val_data['per_class']['wrong_side']['precision']:.1%}")
    print(f"  Class 'right_side' (Normal)    : mAP50={val_data['per_class']['right_side']['mAP50']:.1%}, Precision={val_data['per_class']['right_side']['precision']:.1%}")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO wrong-way driving model")
    parser.add_argument("--model", type=str, default="models/wrong_way_yolo.pt", help="Path to weights")
    parser.add_argument("--data", type=str, default="datasets/wrong_way_roboflow/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--output", type=str, default="data/evaluation_results_wrong_way_yolo.json", help="Output JSON path")

    args = parser.parse_args()
    evaluate_wrong_way_yolo(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.imgsz,
        output_json=args.output,
    )
