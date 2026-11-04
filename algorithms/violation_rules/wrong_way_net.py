"""
algorithms/violation_rules/wrong_way_net.py
Deep Learning Temporal Trajectory Classifier for Wrong-Way Driving Detection.

Architecture:
  - Input: Rolling 7-dimensional trajectory feature sequences over sliding time window
  - Feature Projection: Linear + LayerNorm + ReLU
  - Recurrent Backbone: 2-layer Bidirectional LSTM (Bi-LSTM) with temporal feature extraction
  - Classification Head: Multi-Layer Perceptron (MLP) with Dropout and Softmax
  - Classes:
      0: COMPLIANT_FLOW   (Legal lane flow, turning, or lane-change maneuvers)
      1: WRONG_WAY        (Retrograde motion, opposite-lane driving against vector)
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 7
DEFAULT_SEQ_LEN = 8


def extract_trajectory_features(
    trajectory: List[List[float]],
    lane_vector: List[float],
    seq_len: int = DEFAULT_SEQ_LEN,
    fps: float = 25.0,
) -> Optional[np.ndarray]:
    """
    Extracts normalized temporal kinematics vector from a vehicle's centroid trajectory.
    Returns array of shape [seq_len, FEATURE_DIM] or None if trajectory is too short.
    """
    if len(trajectory) < 3:
        return None

    # Take the last (seq_len + 1) centroid points
    pts = trajectory[-(seq_len + 1):]
    if len(pts) < 3:
        return None

    # Normalize reference lane vector
    rdx, rdy = lane_vector[0], lane_vector[1]
    r_norm = math.sqrt(rdx * rdx + rdy * rdy)
    if r_norm < 1e-4:
        rdx, rdy = 0.0, 1.0
        r_norm = 1.0
    else:
        rdx /= r_norm
        rdy /= r_norm

    features = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        disp = math.sqrt(dx * dx + dy * dy)

        if disp > 1e-4:
            ndx = dx / disp
            ndy = dy / disp
            dot = max(-1.0, min(1.0, ndx * rdx + ndy * rdy))
            cross = ndx * rdy - ndy * rdx
            angle_rad = math.acos(dot)
            angle_deg = math.degrees(angle_rad)
        else:
            dot = 1.0
            cross = 0.0
            angle_deg = 0.0

        speed_estimate = disp * fps

        features.append([
            float(dx / max(1.0, disp)),        # normalized dx
            float(dy / max(1.0, disp)),        # normalized dy
            float(min(50.0, disp) / 50.0),      # displacement scale
            float(dot),                        # cos(theta) against lane vector
            float(cross),                      # sin(theta) / lateral drift
            float(angle_deg / 180.0),          # normalized angular deviation [0, 1]
            float(min(120.0, speed_estimate) / 120.0),  # normalized speed
        ])

    # Pad if shorter than seq_len
    while len(features) < seq_len:
        features.insert(0, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    return np.array(features[-seq_len:], dtype=np.float32)


class WrongWayLSTM(nn.Module):
    """
    Bidirectional LSTM Neural Network for Wrong-Way Driving Sequence Classification.
    """
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)  # [B, T, 2 * hidden_dim]

        # Temporal Attention Pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [B, T, 1]
        pooled = torch.sum(lstm_out * attn_weights, dim=1)         # [B, 2 * hidden_dim]

        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

