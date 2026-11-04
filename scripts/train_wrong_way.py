"""
scripts/train_wrong_way.py
Training Pipeline for Deep Learning Wrong-Way Driving Violation Detection Model.

Trains a Bidirectional LSTM with Attention on temporal trajectory kinematics sequences.
Outputs:
  - Model checkpoint: models/wrong_way_lstm.pt
  - Evaluation report: data/evaluation_results_wrong_way.json
  - Training metrics & loss history: data/training_results/wrong_way/
"""

import os
import sys
import math
import json
import time
import random
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from algorithms.violation_rules.wrong_way_net import (
    WrongWayLSTM,
    extract_trajectory_features,
    FEATURE_DIM,
    DEFAULT_SEQ_LEN,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_trajectory_dataset(num_samples: int = 6000, seq_len: int = DEFAULT_SEQ_LEN):
    """
    Synthesizes a realistic, diverse dataset of vehicle trajectory sequences
    under varying camera angles, speeds, lane directions, and real sensor jitter.
    """
    X_list = []
    y_list = []

    for idx in range(num_samples):
        # Random permitted lane angle in degrees [0, 360)
        lane_angle_deg = random.uniform(0, 360)
        lane_angle_rad = math.radians(lane_angle_deg)
        lane_vec = [math.cos(lane_angle_rad), math.sin(lane_angle_rad)]

        # 50% compliant, 50% wrong-way
        is_wrong_way = (idx % 2 == 1)

        # Base vehicle speed in pixels/frame (e.g. 3 to 25 px/frame)
        speed = random.uniform(4.0, 22.0)

        # Determine motion angle relative to lane vector
        if not is_wrong_way:
            # Legal flow: within -45° to +45° of allowed vector (includes lane changes)
            rel_angle = random.gauss(0, 18)
            rel_angle = max(-75, min(75, rel_angle))
            label = 0
        else:
            # Wrong-way: opposite direction (120° to 180° deviation)
            deviation = random.uniform(120.0, 180.0)
            sign = 1 if random.random() < 0.5 else -1
            rel_angle = sign * deviation
            label = 1

        motion_angle_rad = lane_angle_rad + math.radians(rel_angle)
        vx = speed * math.cos(motion_angle_rad)
        vy = speed * math.sin(motion_angle_rad)

        # Build trajectory points (seq_len + 3 points)
        start_x = random.uniform(100, 1180)
        start_y = random.uniform(100, 620)
        traj = [[start_x, start_y]]

        for s in range(seq_len + 2):
            prev_x, prev_y = traj[-1]
            # Speed acceleration/braking variation
            speed_mult = random.uniform(0.85, 1.15)
            # Jitter from detector / tracker bounding box noise (mean 0, std 1.2 px)
            jitter_x = random.gauss(0, 1.2)
            jitter_y = random.gauss(0, 1.2)

            next_x = prev_x + (vx * speed_mult) + jitter_x
            next_y = prev_y + (vy * speed_mult) + jitter_y
            traj.append([next_x, next_y])

        feat = extract_trajectory_features(traj, lane_vec, seq_len=seq_len)
        if feat is not None and feat.shape == (seq_len, FEATURE_DIM):
            X_list.append(feat)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train_model():
    print("=" * 65)
    print("   TRAINING DEEP LEARNING WRONG-WAY VIOLATION DETECTION MODEL   ")
    print("=" * 65)

    set_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Hardware Compute Device: {device}")

    # 1. Dataset Generation
    print("\n[1/5] Synthesizing & Augmenting Trajectory Sequence Dataset...")
    t0 = time.time()
    X, y = generate_synthetic_trajectory_dataset(num_samples=6000, seq_len=DEFAULT_SEQ_LEN)
    print(f"  Generated {len(X)} trajectory sequences in {time.time() - t0:.2f}s")
    print(f"  Feature shape per sequence: {X.shape[1:]} (SeqLen={DEFAULT_SEQ_LEN}, FeatDim={FEATURE_DIM})")
    print(f"  Class 0 (Compliant Flow) : {np.count_nonzero(y == 0)} samples")
    print(f"  Class 1 (Wrong-Way Offense): {np.count_nonzero(y == 1)} samples")

    # 2. Train / Val Split (80% / 20%)
    split_idx = int(0.80 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

    print(f"\n[2/5] Partitioned Datasets:")
    print(f"  Train Set : {len(X_train)} samples")
    print(f"  Val Set   : {len(X_val)} samples")

    # 3. Model Architecture
    print("\n[3/5] Instantiating Bi-LSTM Architecture (WrongWayLSTM)...")
    model = WrongWayLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=64,
        num_layers=2,
        num_classes=2,
        dropout=0.25,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 4. Training Loop
    epochs = 20
    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n[4/5] Training for {epochs} Epochs...")
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<12}{'Val Acc':<12}{'Val F1':<10}{'Time':<8}")
    print("-" * 65)

    best_weights = None

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss = total_train_loss / len(X_train)

        # Validation
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())

        val_loss = total_val_loss / len(X_val)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        val_acc = np.mean(all_preds == all_targets)

        # Precision, Recall, F1 for Class 1 (Wrong Way)
        tp = np.count_nonzero((all_preds == 1) & (all_targets == 1))
        fp = np.count_nonzero((all_preds == 1) & (all_targets == 0))
        fn = np.count_nonzero((all_preds == 0) & (all_targets == 1))

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(float(val_acc), 4))
        history["val_f1"].append(round(float(f1), 4))

        elapsed = time.time() - ep_start
        print(f"{epoch:<8}{train_loss:<14.4f}{val_loss:<12.4f}{val_acc*100:<11.2f}%{f1*100:<9.2f}%{elapsed:.2f}s")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("-" * 65)

    # 5. Save Artifacts
    print("\n[5/5] Exporting Trained Model Weights & Results...")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "wrong_way_lstm.pt"

    checkpoint = {
        "model_state_dict": best_weights,
        "input_dim": FEATURE_DIM,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_classes": 2,
        "seq_len": DEFAULT_SEQ_LEN,
        "best_val_f1": best_val_f1,
        "val_acc": history["val_acc"][-1],
        "timestamp": time.time(),
        "architecture": "Bidirectional LSTM with Temporal Attention",
    }
    torch.save(checkpoint, save_path)
    print(f"  -> Model saved successfully: {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")

    # Save Evaluation JSON
    eval_results = {
        "model_name": "wrong_way_lstm",
        "architecture": "Bi-LSTM + Temporal Attention",
        "epochs": epochs,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "validation_accuracy": round(float(history["val_acc"][-1]), 4),
        "validation_precision": round(float(precision), 4),
        "validation_recall": round(float(recall), 4),
        "validation_f1": round(float(best_val_f1), 4),
        "history": history,
    }

    eval_json_path = Path("data/evaluation_results_wrong_way.json")
    eval_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"  -> Evaluation results saved: {eval_json_path}")

    # Summary
    print("\n" + "=" * 65)
    print("                 TRAINING COMPLETED SUCCESSFULLY                 ")
    print("=" * 65)
    print(f"Final Validation Accuracy : {history['val_acc'][-1] * 100:.2f}%")
    print(f"Final Validation F1-Score : {best_val_f1 * 100:.2f}%")
    print(f"Model Checkpoint Path     : {save_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train_model()

