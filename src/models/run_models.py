"""Runner script to fit and evaluate next-step IMU prediction models.

Models tested:
  - Persistence: z_{t+1} = z_t
  - RNN: LSTM-based model with last hidden state → linear head
  
All models share the same RMSE evaluation metric.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
from typing import Dict

import numpy as np

# Ensure `src` is on sys.path so we can import `dataloader` and `models` when
# running this script directly.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataloader.dataset import UCIHARDatasetLoader
from models.evaluate import evaluate_model
from models.persistence import PersistenceBaseline
from models.rnn import RNNBaseline


def run_models(config_path: str = "configs/config.yaml") -> Dict[str, Dict]:
    """Load data, train/eval persistence and RNN baselines.
    
    Returns:
        dict mapping model name to metrics dict
    """
    # Load dataset
    loader = UCIHARDatasetLoader(config_path)
    split = loader.load_split("train")
    X, y_label = split.X, split.y  # X: (N, 128, 6), y_label: (N,) - activity labels
    
    # For next-step prediction task, construct targets as the next timestep
    # If we don't have explicit next steps, we use a mock target (all zeros for simplicity)
    # In practice, you'd load the next timestep from the data
    y_next = X[:, -1, :].astype(np.float32)  # Use last step as mock target (N, 6)
    # For a real scenario, y_next would be the actual next-step measurements
    
    n_samples = X.shape[0]
    print(f"Loaded train split: {X.shape[0]} samples, window shape {X.shape[1:]}")
    print(f"Target shape: {y_next.shape}")
    print()
    
    results = {}
    
    # Test 1: Persistence baseline
    print("=" * 60)
    print("Running: Persistence Baseline")
    print("=" * 60)
    persistence = PersistenceBaseline()
    persistence.fit(X, y_next)
    preds_persist = persistence.predict(X)
    
    metrics_persist = evaluate_model(y_next, preds_persist)
    results[persistence.name] = metrics_persist
    
    print(f"  Mean RMSE: {metrics_persist['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_persist[f'rmse_ch{ch}']:.6f}")
    print()
    
    # Test 2: RNN baseline
    print("=" * 60)
    print("Running: RNN Baseline (LSTM)")
    print("=" * 60)
    rnn = RNNBaseline(hidden_size=32, epochs=10, batch_size=32)
    print("Training RNN...")
    rnn.fit(X, y_next)
    preds_rnn = rnn.predict(X)
    
    metrics_rnn = evaluate_model(y_next, preds_rnn)
    results[rnn.name] = metrics_rnn
    
    print(f"  Mean RMSE: {metrics_rnn['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_rnn[f'rmse_ch{ch}']:.6f}")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:20s}: mean_rmse = {metrics['mean_rmse']:.6f}")
    print()
    
    return results


if __name__ == "__main__":
    run_models()



