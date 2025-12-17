"""Evaluation metrics for next-step IMU prediction.

All models (persistence, EKF, PF, RNN) must use this shared metric function
to ensure fair comparison. The task is:

  Input:  X_{t, t+T-1} ∈ ℝ^{T×6}  (window of IMU readings, 6 channels)
  Target: X_{t+T} ∈ ℝ^6           (next-step IMU readings)
  Metric: next-step RMSE (per-channel and mean)
"""

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute per-channel and mean RMSE for next-step prediction.
    
    Args:
        y_true: shape (N, 6) - ground truth next-step IMU readings
        y_pred: shape (N, 6) - predicted next-step IMU readings
    
    Returns:
        (per_channel_rmse, mean_rmse) where:
          - per_channel_rmse: shape (6,) - RMSE per channel
          - mean_rmse: scalar - mean RMSE across all channels
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    
    if y_true.ndim != 2 or y_true.shape[1] != 6:
        raise ValueError(
            f"Expected shape (N, 6), got {y_true.shape}"
        )
    
    # Compute squared error per sample, per channel
    se = (y_true - y_pred) ** 2  # (N, 6)
    
    # Mean over samples per channel
    mse_per_ch = np.mean(se, axis=0)  # (6,)
    
    # RMSE per channel
    rmse_per_ch = np.sqrt(mse_per_ch)
    
    # Mean RMSE across channels
    mean_rmse = float(np.mean(rmse_per_ch))
    
    return rmse_per_ch, mean_rmse


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model and return a metrics dict.
    
    Args:
        y_true: shape (N, 6) - ground truth next-step readings
        y_pred: shape (N, 6) - predicted next-step readings
    
    Returns:
        dict with keys: 'mean_rmse', 'rmse_ch0', ..., 'rmse_ch5'
    """
    rmse_per_ch, mean_rmse = compute_rmse(y_true, y_pred)
    
    metrics = {"mean_rmse": mean_rmse}
    for ch in range(6):
        metrics[f"rmse_ch{ch}"] = float(rmse_per_ch[ch])
    
    return metrics
