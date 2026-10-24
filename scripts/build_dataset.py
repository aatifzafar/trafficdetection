"""
build_dataset.py — Merge raw dataset archives into a unified YOLO dataset.

Final Classes:
    0: helmet
    1: no_helmet
    2: number_plate

Target Layout:
    datasets/merged/
        images/train/
        images/val/
        labels/train/
        labels/val/
        data.yaml
        preview.jpg
"""

import os
import sys
import glob
import shutil
import random
import hashlib
import zipfile
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# Final class mapping definition
CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
    2: "number_plate",
}

CLASS_COLORS = {
    0: (0, 220, 0),     # Green for helmet
    1: (0, 0, 240),     # Red for no_helmet
    2: (255, 180, 0),   # Cyan/Blue for number_plate (BGR)
}


class DatasetBuilder:
    def __init__(self, raw_dir: Path, output_dir: Path, seed: int = 42):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.seed = seed
        self.rng = random.Random(seed)
        self.seen_hashes = set()
        self.duplicate_count = 0
        
        # In-memory collection of all deduplicated items:
        # dict: {
        #   "source": str,
        #   "filename": str,
        #   "img_bytes": bytes,
        #   "boxes": list of (cls_id, xc, yc, w, h),
        #   "is_video": bool
        # }
        self.items = []
        
        # Stats tracking: source -> split -> {images, boxes_per_class}
        self.stats = {}

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _is_duplicate(self, img_bytes: bytes) -> bool:
        h = self._hash_bytes(img_bytes)
        if h in self.seen_hashes:
            self.duplicate_count += 1
            return True
        self.seen_hashes.add(h)
        return False

    @staticmethod
    def _voc_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int):
        """Converts VOC pixel bounding box to YOLO normalized xywh, clamped to [0, 1]."""
        if img_w <= 0 or img_h <= 0 or xmax <= xmin or ymax <= ymin:
            return None
        xmin = max(0.0, min(float(img_w), xmin))
        ymin = max(0.0, min(float(img_h), ymin))
        xmax = max(0.0, min(float(img_w), xmax))
        ymax = max(0.0, min(float(img_h), ymax))
        
        w_box = xmax - xmin
        h_box = ymax - ymin
        if w_box <= 0 or h_box <= 0:
            return None
        
        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        w_norm = w_box / img_w
        h_norm = h_box / img_h
        return xc, yc, w_norm, h_norm

    def inspect_archive_7(self):
        """Check archive (7).zip contents and determine if it has helmet or plate boxes."""
        zip_path = self.raw_dir / "archive (7).zip"
        if not zip_path.exists():
            print("[archive 7] Not found, skipping.")
            return
        
        print("\n--- Inspecting archive (7).zip ---")
        classes_found = set()
        with zipfile.ZipFile(zip_path) as z:
            xmls = [f for f in z.namelist() if f.endswith(".xml")]
            print(f"[archive 7] Found {len(z.namelist())} entries ({len(xmls)} XML annotations).")
            # Scan all xmls for object names
            for x in xmls:
                try:
                    root = ET.fromstring(z.read(x))
                    for obj in root.findall("object"):
                        n = obj.find("name")
                        if n is not None and n.text:
                            classes_found.add(n.text.strip())
                except Exception:
                    continue
        print(f"[archive 7] Object classes found: {classes_found}")
        has_helmet_or_plate = any(
            c.lower() in ["helmet", "with helmet", "without helmet", "plate", "number_plate", "licence_plate"]
            for c in classes_found
        )
        if not has_helmet_or_plate:
            print("[archive 7] Content identified as IITM-HeTra v2 (vehicle detection: Auto, Bus, Person, Car).")
            print("[archive 7] Contains NO helmet or number plate annotations. SKIPPING per instructions.\n")
        else:
            print("[archive 7] Helmet/plate found, including.\n")

    def process_training_dataset_files(self):
        """
        training_dataset_files.zip:
        - 01_indian_number_plates/yolo: class 0 = plate -> final class 2 (number_plate)
        - 02_helmet_riders/yolo: class 0 = helmet -> 0, class 1 = no_helmet -> 1
        - Ignore 03_traffic_light_state
        """
        zip_path = self.raw_dir / "training_dataset_files.zip"
        if not zip_path.exists():
            print(f"[skip] {zip_path.name} not found")
            return

        print(f"\nProcessing {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            # 1. 01_indian_number_plates/yolo
            plate_imgs = [f for f in z.namelist() if f.startswith("01_indian_number_plates/yolo/images/") and f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"  Found {len(plate_imgs)} plate images in 01_indian_number_plates")
            for img_path in plate_imgs:
                stem = Path(img_path).stem
                # Find matching label in train or val
                lbl_candidates = [
                    f"01_indian_number_plates/yolo/labels/train/{stem}.txt",
                    f"01_indian_number_plates/yolo/labels/val/{stem}.txt",
                ]
                lbl_path = next((c for c in lbl_candidates if c in z.namelist()), None)
                if not lbl_path:
                    continue
                
                img_bytes = z.read(img_path)
                if self._is_duplicate(img_bytes):
                    continue
                
                boxes = []
                lbl_content = z.read(lbl_path).decode("utf-8", errors="ignore")
                for line in lbl_content.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        boxes.append((2, xc, yc, w, h))  # 0 -> 2 (number_plate)
                
                if boxes:
                    self.items.append({
                        "source": "01_indian_plates",
                        "filename": f"inplates_{stem}.jpg",
                        "img_bytes": img_bytes,
                        "boxes": boxes,
                        "is_video": False,
                    })

            # 2. 02_helmet_riders/yolo
            helmet_imgs = [f for f in z.namelist() if f.startswith("02_helmet_riders/yolo/images/") and f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"  Found {len(helmet_imgs)} helmet images in 02_helmet_riders")
            for img_path in helmet_imgs:
                stem = Path(img_path).stem
                lbl_candidates = [
                    f"02_helmet_riders/yolo/labels/train/{stem}.txt",
                    f"02_helmet_riders/yolo/labels/val/{stem}.txt",
                ]
                lbl_path = next((c for c in lbl_candidates if c in z.namelist()), None)
                if not lbl_path:
                    continue
                
                img_bytes = z.read(img_path)
                if self._is_duplicate(img_bytes):
                    continue
                
                boxes = []
                lbl_content = z.read(lbl_path).decode("utf-8", errors="ignore")
                for line in lbl_content.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_src = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        # 0 -> helmet (0), 1 -> no_helmet (1)
                        if cls_src in (0, 1):
                            boxes.append((cls_src, xc, yc, w, h))
                
                if boxes:
                    self.items.append({
                        "source": "02_helmet_riders",
                        "filename": f"riders_{stem}.jpg",
                        "img_bytes": img_bytes,
                        "boxes": boxes,
                        "is_video": False,
                    })

    def process_archive_3(self):
        """
        archive (3).zip:
        - images/*.png + annotations/*.xml (Pascal VOC)
        - Names: 'With Helmet' -> 0 (helmet), 'Without Helmet' -> 1 (no_helmet)
        """
        zip_path = self.raw_dir / "archive (3).zip"
        if not zip_path.exists():
            print(f"[skip] {zip_path.name} not found")
            return

        print(f"\nProcessing {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            xml_dict = {}
            for f in z.namelist():
                if f.startswith("annotations/") and f.endswith(".xml"):
                    xml_dict[Path(f).stem] = f
            
            img_list = [f for f in z.namelist() if f.startswith("images/") and f.lower().endswith((".png", ".jpg", ".jpeg"))]
            print(f"  Found {len(img_list)} images, {len(xml_dict)} xmls")
            
            for img_path in img_list:
                stem = Path(img_path).stem
                if stem not in xml_dict:
                    continue
                
                img_bytes = z.read(img_path)
                if self._is_duplicate(img_bytes):
                    continue
                
                # Decode image to get dimensions safely
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                
                xml_content = z.read(xml_dict[stem])
                root = ET.fromstring(xml_content)
                boxes = []
                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    if name_elem is None or not name_elem.text:
                        continue
                    name = name_elem.text.strip().lower()
                    if name == "with helmet":
                        cls_id = 0
                    elif name == "without helmet":
                        cls_id = 1
                    else:
                        continue
                    
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    yolo_box = self._voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                    if yolo_box:
                        boxes.append((cls_id, *yolo_box))
                
                if boxes:
                    self.items.append({
                        "source": "archive_3_voc",
                        "filename": f"ar3_{stem}.png",
                        "img_bytes": img_bytes,
                        "boxes": boxes,
                        "is_video": False,
                    })

    def process_archive_4(self):
        """
        archive (4).zip:
        - HelmetViolations/ and HelmetViolationsV2/ (Roboflow YOLO, train/valid/test)
        - Read EACH data.yaml: names: ['Plate','WithHelmet','WithoutHelmet']
          Plate -> 2 (number_plate), WithHelmet -> 0 (helmet), WithoutHelmet -> 1 (no_helmet)
        - Extract 22.mp4 to data/videos/22.mp4
        - Ignore best.pt
        """
        zip_path = self.raw_dir / "archive (4).zip"
        if not zip_path.exists():
            print(f"[skip] {zip_path.name} not found")
            return

        print(f"\nProcessing {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            # Extract test video 22.mp4
            if "22.mp4" in z.namelist():
                videos_dir = Path("data/videos")
                videos_dir.mkdir(parents=True, exist_ok=True)
                test_video_path = videos_dir / "22.mp4"
                if not test_video_path.exists():
                    print(f"  Extracting test video 22.mp4 -> {test_video_path}")
                    with open(test_video_path, "wb") as f_out:
                        f_out.write(z.read("22.mp4"))
                else:
                    print(f"  Test video 22.mp4 already present at {test_video_path}")

            # Process both V1 and V2 folders
            for subfolder in ["HelmetViolations", "HelmetViolationsV2"]:
                yaml_path = f"{subfolder}/data.yaml"
                if yaml_path not in z.namelist():
                    continue
                
                # Parse data.yaml for class names
                yaml_content = z.read(yaml_path).decode("utf-8", errors="ignore")
                # Parse names list
                names_map = {}
                import yaml
                try:
                    data_cfg = yaml.safe_load(yaml_content)
                    raw_names = data_cfg.get("names", [])
                    if isinstance(raw_names, list):
                        for idx, name in enumerate(raw_names):
                            n = str(name).strip().lower()
                            if "plate" in n:
                                names_map[idx] = 2
                            elif "with" in n and "without" not in n:
                                names_map[idx] = 0
                            elif "without" in n:
                                names_map[idx] = 1
                    elif isinstance(raw_names, dict):
                        for idx, name in raw_names.items():
                            n = str(name).strip().lower()
                            if "plate" in n:
                                names_map[int(idx)] = 2
                            elif "with" in n and "without" not in n:
                                names_map[int(idx)] = 0
                            elif "without" in n:
                                names_map[int(idx)] = 1
                except Exception as e:
                    print(f"  Error parsing {yaml_path}: {e}, using fallback map")
                    names_map = {0: 2, 1: 0, 2: 1}
                
                print(f"  {subfolder} class mapping: {names_map}")
                
                # Collect images in this subfolder
                imgs = [
                    f for f in z.namelist()
                    if f.startswith(f"{subfolder}/") and "/images/" in f and f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
                print(f"  Found {len(imgs)} images in {subfolder}")
                
                for img_path in imgs:
                    lbl_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                    if lbl_path not in z.namelist():
                        continue
                    
                    img_bytes = z.read(img_path)
                    if self._is_duplicate(img_bytes):
                        continue
                    
                    boxes = []
                    lbl_content = z.read(lbl_path).decode("utf-8", errors="ignore")
                    for line in lbl_content.strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_src = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            if cls_src in names_map:
                                boxes.append((names_map[cls_src], xc, yc, w, h))
                    
                    if boxes:
                        stem = Path(img_path).name
                        self.items.append({
                            "source": f"archive_4_{subfolder}",
                            "filename": f"ar4_{subfolder[:6]}_{stem}",
                            "img_bytes": img_bytes,
                            "boxes": boxes,
                            "is_video": False,
                        })

    def process_archive_8(self):
        """
        archive (8).zip:
        - images/ + labels/ (YOLO, class 0 = plate -> final class 2: number_plate)
        - Match label to image by filename stem; skip images with no label file
        """
        zip_path = self.raw_dir / "archive (8).zip"
        if not zip_path.exists():
            print(f"[skip] {zip_path.name} not found")
            return

        print(f"\nProcessing {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            labels_dict = {}
            for f in z.namelist():
                if f.startswith("labels/") and f.endswith(".txt"):
                    labels_dict[Path(f).stem] = f
            
            imgs = [f for f in z.namelist() if f.startswith("images/") and f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"  Found {len(imgs)} images, {len(labels_dict)} labels")
            
            matched = 0
            for img_path in imgs:
                stem = Path(img_path).stem
                if stem not in labels_dict:
                    continue  # skip images with no label file
                
                img_bytes = z.read(img_path)
                if self._is_duplicate(img_bytes):
                    continue
                
                boxes = []
                lbl_content = z.read(labels_dict[stem]).decode("utf-8", errors="ignore")
                for line in lbl_content.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        boxes.append((2, xc, yc, w, h))  # 0 -> 2 (number_plate)
                
                if boxes:
                    matched += 1
                    ext = Path(img_path).suffix
                    self.items.append({
                        "source": "archive_8_plates",
                        "filename": f"ar8_{stem}{ext}",
                        "img_bytes": img_bytes,
                        "boxes": boxes,
                        "is_video": False,
                    })
            print(f"  Added {matched} images from archive (8)")

    def process_archive_9(self):
        """
        archive (9).zip:
        - vid-1/frame-*.jpg + .txt (YOLO, classes.txt = number_plate)
        - Class 0 -> final class 2 (number_plate)
        - All images from vid-1 must stay in ONE split (train)
        """
        zip_path = self.raw_dir / "archive (9).zip"
        if not zip_path.exists():
            print(f"[skip] {zip_path.name} not found")
            return

        print(f"\nProcessing {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            labels_dict = {}
            for f in z.namelist():
                if f.startswith("vid-1/") and f.endswith(".txt") and "classes" not in f:
                    labels_dict[Path(f).stem] = f
            
            imgs = [f for f in z.namelist() if f.startswith("vid-1/") and f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"  Found {len(imgs)} video frames in vid-1, {len(labels_dict)} labels")
            
            matched = 0
            for img_path in imgs:
                stem = Path(img_path).stem
                if stem not in labels_dict:
                    continue
                
                img_bytes = z.read(img_path)
                if self._is_duplicate(img_bytes):
                    continue
                
                boxes = []
                lbl_content = z.read(labels_dict[stem]).decode("utf-8", errors="ignore")
                for line in lbl_content.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        boxes.append((2, xc, yc, w, h))  # 0 -> 2 (number_plate)
                
                if boxes:
                    matched += 1
                    ext = Path(img_path).suffix
                    self.items.append({
                        "source": "archive_9_video1",
                        "filename": f"ar9_{stem}{ext}",
                        "img_bytes": img_bytes,
                        "boxes": boxes,
                        "is_video": True,  # Mark as video source so it stays in one split!
                    })
            print(f"  Added {matched} frames from archive (9)")

    def process_archive_10(self):
        """
        archive (10).zip:
        - VOC XML; keep only objects named 'number_plate' (ignore attributes such as 'rotation')
        - Pairs:
            Annotations/Annotations/<name>.xml <-> Indian_Number_Plates/Sample_Images/<name>.jpg
            number_plate_annos_ocr/.../<name>.xml <-> number_plate_images_ocr/.../<name>.jpg
        """
        zip_path = self.raw_dir / "archive (10).zip"
        if not zip_path.exists():
            print(f"[skip] {zip_path.name} not found")
            return

        print(f"\nProcessing {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as z:
            xml_dict = {}
            for f in z.namelist():
                if f.endswith(".xml"):
                    xml_dict[Path(f).stem] = f
            
            imgs = [f for f in z.namelist() if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            print(f"  Found {len(imgs)} images, {len(xml_dict)} xmls")
            
            matched = 0
            for img_path in imgs:
                stem = Path(img_path).stem
                if stem not in xml_dict:
                    continue
                
                img_bytes = z.read(img_path)
                if self._is_duplicate(img_bytes):
                    continue
                
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                
                root = ET.fromstring(z.read(xml_dict[stem]))
                boxes = []
                for obj in root.findall("object"):
                    # Strictly check direct child 'name'
                    name_elem = obj.find("name")
                    if name_elem is None or not name_elem.text:
                        continue
                    if name_elem.text.strip().lower() != "number_plate":
                        continue
                    
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    yolo_box = self._voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                    if yolo_box:
                        boxes.append((2, *yolo_box))  # class 2 = number_plate
                
                if boxes:
                    matched += 1
                    ext = Path(img_path).suffix
                    self.items.append({
                        "source": "archive_10_plates",
                        "filename": f"ar10_{stem}{ext}",
                        "img_bytes": img_bytes,
                        "boxes": boxes,
                        "is_video": False,
                    })
            print(f"  Added {matched} images from archive (10)")

    def split_and_write(self):
        """
        Splits 80/20 train/val with seed 42, keeping images from archive 9 (video) in train.
        Writes images and labels to datasets/merged/.
        """
        print(f"\n--- Splitting Dataset ---")
        print(f"Total deduplicated items collected: {len(self.items)} (skipped {self.duplicate_count} exact duplicates)")

        video_items = [item for item in self.items if item["is_video"]]
        non_video_items = [item for item in self.items if not item["is_video"]]

        # Shuffle non-video items reproducibly
        self.rng.shuffle(non_video_items)

        # Total dataset size
        total_count = len(self.items)
        val_target_count = int(round(0.20 * total_count))

        # Take val items from non-video items
        val_items = non_video_items[:val_target_count]
        train_items = non_video_items[val_target_count:] + video_items

        print(f"Train items: {len(train_items)} ({len(train_items)/total_count*100:.1f}%)")
        print(f"Val items:   {len(val_items)} ({len(val_items)/total_count*100:.1f}%)")
        print(f"Archive 9 video frames in train: {len(video_items)} (0 in val)")

        # Prepare directories
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Write items
        split_map = {"train": train_items, "val": val_items}
        for split_name, items in split_map.items():
            for item in items:
                src = item["source"]
                if src not in self.stats:
                    self.stats[src] = {
                        "train": {"images": 0, "boxes": {0: 0, 1: 0, 2: 0}},
                        "val": {"images": 0, "boxes": {0: 0, 1: 0, 2: 0}},
                    }
                self.stats[src][split_name]["images"] += 1

                # Write image
                img_out_path = self.output_dir / "images" / split_name / item["filename"]
                with open(img_out_path, "wb") as f_img:
                    f_img.write(item["img_bytes"])

                # Write label
                lbl_out_path = self.output_dir / "labels" / split_name / f"{Path(item['filename']).stem}.txt"
                lbl_lines = []
                for cls_id, xc, yc, w, h in item["boxes"]:
                    self.stats[src][split_name]["boxes"][cls_id] += 1
                    lbl_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                
                with open(lbl_out_path, "w", encoding="utf-8") as f_lbl:
                    f_lbl.write("\n".join(lbl_lines) + "\n")

        # Write data.yaml
        yaml_content = f"""# Merged Traffic Violation Dataset (YOLO format)
# Generated by scripts/build_dataset.py
path: .
train: images/train
val: images/val

nc: 3
names:
  0: helmet
  1: no_helmet
  2: number_plate
"""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f_yaml:
            f_yaml.write(yaml_content)
        print(f"\nWrote data.yaml to {yaml_path}")

    def print_summary_table(self):
        """Prints a detailed table of images and boxes per class, source, and split."""
        print("\n" + "=" * 90)
        print(f"{'Source':<26} | {'Split':<6} | {'Images':<7} | {'Helmet(0)':<9} | {'NoHelmet(1)':<11} | {'Plate(2)':<9} | {'Total Boxes':<11}")
        print("-" * 90)

        total_imgs = {"train": 0, "val": 0}
        total_boxes = {
            "train": {0: 0, 1: 0, 2: 0},
            "val": {0: 0, 1: 0, 2: 0},
        }

        for src, splits in sorted(self.stats.items()):
            for split in ["train", "val"]:
                s = splits[split]
                img_cnt = s["images"]
                b0 = s["boxes"][0]
                b1 = s["boxes"][1]
                b2 = s["boxes"][2]
                b_tot = b0 + b1 + b2

                total_imgs[split] += img_cnt
                total_boxes[split][0] += b0
                total_boxes[split][1] += b1
                total_boxes[split][2] += b2

                print(f"{src:<26} | {split:<6} | {img_cnt:<7} | {b0:<9} | {b1:<11} | {b2:<9} | {b_tot:<11}")
            print("-" * 90)

        # Totals per split
        for split in ["train", "val"]:
            t_imgs = total_imgs[split]
            t0 = total_boxes[split][0]
            t1 = total_boxes[split][1]
            t2 = total_boxes[split][2]
            print(f"{'SUBTOTAL ' + split.upper():<26} | {split:<6} | {t_imgs:<7} | {t0:<9} | {t1:<11} | {t2:<9} | {t0+t1+t2:<11}")
        print("=" * 90)

        grand_imgs = total_imgs["train"] + total_imgs["val"]
        g0 = total_boxes["train"][0] + total_boxes["val"][0]
        g1 = total_boxes["train"][1] + total_boxes["val"][1]
        g2 = total_boxes["train"][2] + total_boxes["val"][2]
        print(f"{'GRAND TOTAL':<26} | {'ALL':<6} | {grand_imgs:<7} | {g0:<9} | {g1:<11} | {g2:<9} | {g0+g1+g2:<11}")
        print("=" * 90 + "\n")

    def generate_preview_grid(self, n_samples: int = 12):
        """Draws 12 random images with boxes to datasets/merged/preview.jpg."""
        if not self.items:
            return

        print(f"Generating preview grid with {n_samples} samples...")
        sample_items = self.rng.sample(self.items, min(n_samples, len(self.items)))
        
        target_thumb_w = 480
        target_thumb_h = 360
        thumbnails = []

        for item in sample_items:
            nparr = np.frombuffer(item["img_bytes"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            for cls_id, xc, yc, bw, bh in item["boxes"]:
                x1 = int(round((xc - bw / 2.0) * w))
                y1 = int(round((yc - bh / 2.0) * h))
                x2 = int(round((xc + bw / 2.0) * w))
                y2 = int(round((yc + bh / 2.0) * h))
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = CLASS_NAMES.get(cls_id, str(cls_id))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img, (x1, max(0, y1 - txt_h - 4)), (x1 + txt_w + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, max(txt_h, y1 - 2)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            thumb = cv2.resize(img, (target_thumb_w, target_thumb_h), interpolation=cv2.INTER_AREA)
            thumbnails.append(thumb)

        # Arrange in a 3x4 grid (3 rows, 4 cols)
        rows = 3
        cols = 4
        # Pad with black thumbnail if fewer than 12
        while len(thumbnails) < rows * cols:
            thumbnails.append(np.zeros((target_thumb_h, target_thumb_w, 3), dtype=np.uint8))

        row_images = []
        for r in range(rows):
            row_images.append(np.hstack(thumbnails[r * cols : (r + 1) * cols]))
        grid = np.vstack(row_images)

        preview_path = self.output_dir / "preview.jpg"
        cv2.imwrite(str(preview_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"Saved preview grid to {preview_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "training dataset"
    output_dir = repo_root / "datasets" / "merged"

    builder = DatasetBuilder(raw_dir=raw_dir, output_dir=output_dir, seed=42)

    # 1. Inspect archive 7
    builder.inspect_archive_7()

    # 2. Process all datasets
    builder.process_training_dataset_files()
    builder.process_archive_3()
    builder.process_archive_4()
    builder.process_archive_8()
    builder.process_archive_9()
    builder.process_archive_10()

    # 3. Split and write merged dataset
    builder.split_and_write()

    # 4. Print summary table
    builder.print_summary_table()

    # 5. Generate preview image
    builder.generate_preview_grid(n_samples=12)

    print("\nPhase 1 dataset build complete!")


if __name__ == "__main__":
    main()
