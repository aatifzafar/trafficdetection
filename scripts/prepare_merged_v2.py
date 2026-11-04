import os
import shutil
import hashlib
import random
import zipfile
from pathlib import Path

DATASET_ROOT = Path("datasets/merged")
RAW_ARCHIVE_4 = Path("datasets/raw/archive (4).zip")

def get_md5(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def map_class(name: str):
    n = name.lower().strip()
    if "plate" in n or "licen" in n:
        return 2
    if any(k in n for k in ["without", "no helmet", "nohelmet", "not wearing", "head"]):
        return 1
    if "helmet" in n:
        return 0
    return None

def main():
    print("=== Step 1 & 2: Dataset Preparation and Harmonization ===")
    
    # 1. Verify original dataset exists
    orig_train_dir = DATASET_ROOT / "images" / "train"
    orig_val_dir = DATASET_ROOT / "images" / "val"
    orig_train_lbl_dir = DATASET_ROOT / "labels" / "train"
    orig_val_lbl_dir = DATASET_ROOT / "labels" / "val"
    
    assert orig_train_dir.exists(), f"Missing {orig_train_dir}"
    assert orig_val_dir.exists(), f"Missing {orig_val_dir}"
    
    test_img_dir = DATASET_ROOT / "images" / "test"
    test_lbl_dir = DATASET_ROOT / "labels" / "test"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    orig_train_imgs = sorted(list(orig_train_dir.glob("*.*")))
    orig_val_imgs = sorted(list(orig_val_dir.glob("*.*")))
    print(f"Original counts before split: {len(orig_train_imgs)} train, {len(orig_val_imgs)} val")
    
    # Check if 300 images were already sampled into test split
    existing_test = list(test_img_dir.glob("*.*"))
    if len(existing_test) == 0:
        print("Sampling 300 original train images into held-out test split (random.Random(0))...")
        rng = random.Random(0)
        sampled_300 = set(rng.sample(orig_train_imgs, 300))
        for img_p in sampled_300:
            stem = img_p.stem
            shutil.move(str(img_p), str(test_img_dir / img_p.name))
            lbl_p = orig_train_lbl_dir / f"{stem}.txt"
            if lbl_p.exists():
                shutil.move(str(lbl_p), str(test_lbl_dir / f"{stem}.txt"))
        print(f"Moved 300 original train images to test split.")
    else:
        print(f"Test split already contains {len(existing_test)} images.")

    # 2. Compute existing hashes for deduplication
    print("\nComputing MD5 hashes of all current images to prevent duplicates...")
    existing_hashes = set()
    for s in ["train", "val", "test"]:
        for img_p in (DATASET_ROOT / "images" / s).glob("*.*"):
            with open(img_p, "rb") as f:
                existing_hashes.add(get_md5(f.read()))
    print(f"Indexed {len(existing_hashes)} unique existing image hashes.")

    # 3. Read archive (4).zip and map classes
    print("\n--- Ingesting Extra Dataset: datasets/raw/archive (4).zip ---")
    orig_classes = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet"}
    mapped_classes = {cid: map_class(cname) for cid, cname in orig_classes.items()}
    target_names = {0: "helmet", 1: "no_helmet", 2: "number_plate"}
    print("Class Mapping for HelmetViolations (archive (4).zip):")
    for cid, cname in orig_classes.items():
        print(f"  Orig [{cid}: {cname}] -> Target [{mapped_classes[cid]}: {target_names[mapped_classes[cid]]}]")

    added_train = 0
    added_test = 0
    skipped_dup = 0
    skipped_no_labels = 0

    with zipfile.ZipFile(RAW_ARCHIVE_4, "r") as z:
        namelist = z.namelist()
        
        # Categorize entries
        for name in namelist:
            if not any(name.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                continue
            if name.startswith("__") or "/" not in name:
                continue

            # Determine destination split: train -> train, valid/test -> test
            is_train = "/train/" in name or name.startswith("train/")
            is_val_or_test = ("/valid/" in name or "/test/" in name or 
                              name.startswith("valid/") or name.startswith("test/"))
            if not (is_train or is_val_or_test):
                continue

            img_bytes = z.read(name)
            img_hash = get_md5(img_bytes)
            if img_hash in existing_hashes:
                skipped_dup += 1
                continue

            # Locate corresponding label in zip
            # Replace /images/ with /labels/ and .jpg with .txt
            lbl_name = name.replace("/images/", "/labels/")
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                if lbl_name.endswith(ext):
                    lbl_name = lbl_name[:-len(ext)] + ".txt"
                    break

            if lbl_name not in namelist:
                skipped_no_labels += 1
                continue

            raw_lbl = z.read(lbl_name).decode("utf-8", errors="ignore")
            converted_lines = []
            for line in raw_lbl.strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    orig_cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    continue

                if orig_cls_id not in mapped_classes or mapped_classes[orig_cls_id] is None:
                    continue
                target_cls = mapped_classes[orig_cls_id]

                if len(coords) == 4:
                    # Standard box [cx, cy, w, h]
                    cx, cy, w, h = coords
                elif len(coords) >= 6:
                    # Polygon -> bounding box
                    xs = coords[0::2]
                    ys = coords[1::2]
                    min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                    min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                    cx = (min_x + max_x) / 2.0
                    cy = (min_y + max_y) / 2.0
                    w = max(0.001, max_x - min_x)
                    h = max(0.001, max_y - min_y)
                else:
                    continue

                converted_lines.append(f"{target_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not converted_lines:
                skipped_no_labels += 1
                continue

            # Save to disk
            img_filename = f"ext_hv_{Path(name).name}"
            lbl_filename = f"ext_hv_{Path(lbl_name).name}"

            if is_train:
                dst_img = DATASET_ROOT / "images" / "train" / img_filename
                dst_lbl = DATASET_ROOT / "labels" / "train" / lbl_filename
                added_train += 1
            else:
                dst_img = DATASET_ROOT / "images" / "test" / img_filename
                dst_lbl = DATASET_ROOT / "labels" / "test" / lbl_filename
                added_test += 1

            with open(dst_img, "wb") as f:
                f.write(img_bytes)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(converted_lines) + "\n")

            existing_hashes.add(img_hash)

    print(f"\nExtra Data Merge Results:")
    print(f"  Added to train split: {added_train} images")
    print(f"  Added to test split:  {added_test} images")
    print(f"  Skipped exact duplicates: {skipped_dup}")
    print(f"  Skipped empty/no-label:   {skipped_no_labels}")

    # 4. Final Split Summary
    cur_tr = len(list((DATASET_ROOT / "images" / "train").glob("*.*")))
    cur_va = len(list((DATASET_ROOT / "images" / "val").glob("*.*")))
    cur_te = len(list((DATASET_ROOT / "images" / "test").glob("*.*")))
    print(f"\nFinal Dataset Split Counts:")
    print(f"  TRAIN split: {cur_tr} images")
    print(f"  VAL split:   {cur_va} images (strictly UNTOUCHED)")
    print(f"  TEST split:  {cur_te} images (held-out honest evaluation)")

    # 5. Write data.yaml
    abs_root = Path("datasets/merged").resolve().as_posix()
    yaml_content = f"""# YOLO Traffic Violation Detection Dataset
path: {abs_root}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: helmet
  1: no_helmet
  2: number_plate
"""
    yaml_path = DATASET_ROOT / "data.yaml"
    yaml_path.write_text(yaml_content.strip() + "\n", encoding="utf-8")
    print(f"\nWritten {yaml_path}:")
    print(yaml_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()

