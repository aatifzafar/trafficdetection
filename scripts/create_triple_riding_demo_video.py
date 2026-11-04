import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def make_smooth_sequence(img_path: str, num_frames: int = 35, out_w: int = 1280, out_h: int = 720) -> list:
    """Creates a smooth simulated CCTV trajectory from a static frame."""
    src = cv2.imread(str(img_path))
    if src is None:
        return []
    
    sh, sw = src.shape[:2]
    # Letterbox onto out_w, out_h
    scale = min(out_w / sw, out_h / sh) * 0.90
    nw, nh = int(sw * scale), int(sh * scale)
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    frames = []
    # Smooth motion: move across horizontal plane with minor vertical vibration
    start_x = int(out_w * 0.05)
    end_x = int(out_w * 0.25)
    base_y = int((out_h - nh) / 2)
    
    for i in range(num_frames):
        t = i / float(num_frames - 1)
        # Cosine ease
        ease_t = 0.5 * (1.0 - np.cos(t * np.pi))
        cur_x = int(start_x + ease_t * (end_x - start_x))
        cur_y = base_y + int(np.sin(i * 0.4) * 2) # subtle natural road wobble
        
        # Background: dark asphalt / city road tone
        bg = np.full((out_h, out_w, 3), 45, dtype=np.uint8)
        
        # Place motorcycle crop
        x1 = max(0, cur_x)
        y1 = max(0, cur_y)
        x2 = min(out_w, cur_x + nw)
        y2 = min(out_h, cur_y + nh)
        
        rx1 = 0
        ry1 = 0
        rx2 = x2 - x1
        ry2 = y2 - y1
        
        bg[y1:y2, x1:x2] = resized[ry1:ry2, rx1:rx2]
        
        # Add road perspective markings
        road_y = int(out_h * 0.88)
        cv2.line(bg, (0, road_y), (out_w, road_y), (180, 180, 180), 2)
        
        frames.append(bg)
        
    return frames

def build_demo_video():
    valid_dir = Path("datasets/raw/05_triple_riding/valid/images")
    key_images = [
        valid_dir / "103_jpg.rf.e614c5f3655c1e5e9fe6316b4b9aaea8.jpg",
        valid_dir / "130_jpg.rf.91962c6fbe7d26902dea957afd9abee9.jpg",
        valid_dir / "100_jpg.rf.1adbda0e75921ea1e7bd67534fee043d.jpg",
        valid_dir / "120_jpg.rf.9fdda545d2c2b8b618b151b050b5f48a.jpg",
    ]
    
    out_path = Path("data/videos/demo_triple_riding.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_w, out_h = 1280, 720
    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    
    total_written = 0
    for img_p in key_images:
        if not img_p.exists():
            print(f"Warning: {img_p} not found, skipping")
            continue
        seq = make_smooth_sequence(str(img_p), num_frames=35, out_w=out_w, out_h=out_h)
        for f in seq:
            writer.write(f)
            total_written += 1
            
    writer.release()
    print(f"Successfully generated {out_path} ({total_written} frames @ {fps} FPS, {total_written/fps:.1f}s)")

if __name__ == "__main__":
    build_demo_video()

