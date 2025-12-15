"""Runner script to fit and evaluate next-step IMU prediction models.

Models tested:
  - Persistence: z_{t+1} = z_t
  - RNN: LSTM-based model with last hidden state → linear head
  
All models share the same RMSE evaluation metric.

Pipeline:
  1. Load train split → fit models
  2. Load test split → evaluate models on held-out data
  3. Save metrics to results/metrics/*.json
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
import json
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
    """Load data, train on train split, evaluate on test split.
    
    Task definition:
      - Input window: X_in = X[:, :-1, :] shape (N, T-1, 6)
      - Target next-step: y_next = X[:, -1, :] shape (N, 6)
      - Metric: RMSE (per-channel and mean)
    
    Returns:
        dict mapping model name to dict of train and test metrics
    """
    # Load train split
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    loader = UCIHARDatasetLoader(config_path)
    
    train_split = loader.load_split("train")
    X_train, y_train_label = train_split.X, train_split.y
    
    # Construct train task
    X_train_in = X_train[:, :-1, :].astype(np.float32)
    y_train_next = X_train[:, -1, :].astype(np.float32)
    
    test_split = loader.load_split("test")
    X_test, y_test_label = test_split.X, test_split.y
    
    # Construct test task
    X_test_in = X_test[:, :-1, :].astype(np.float32)
    y_test_next = X_test[:, -1, :].astype(np.float32)
    
    T = X_train.shape[1]
    print(f"Train split: {X_train_in.shape[0]} samples, shape (N, T-1={T-1}, 6)")
    print(f"Test split:  {X_test_in.shape[0]} samples, shape (N, T-1={T-1}, 6)")
    print()
    
    results = {}
    
    # Test 1: Persistence baseline
    print("=" * 60)
    print("Running: Persistence Baseline")
    print("=" * 60)
    persistence = PersistenceBaseline()
    persistence.fit(X_train_in, y_train_next)
    
    # Evaluate on train
    preds_persist_train = persistence.predict(X_train_in)
    metrics_persist_train = evaluate_model(y_train_next, preds_persist_train)
    
    # Evaluate on test
    preds_persist_test = persistence.predict(X_test_in)
    metrics_persist_test = evaluate_model(y_test_next, preds_persist_test)
    
    results[persistence.name] = {
        "train": metrics_persist_train,
        "test": metrics_persist_test,
    }
    
    print(f"  Train - Mean RMSE: {metrics_persist_train['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_persist_train[f'rmse_ch{ch}']:.6f}")
    print(f"  Test  - Mean RMSE: {metrics_persist_test['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_persist_test[f'rmse_ch{ch}']:.6f}")
    print()
    
    # Test 2: RNN baseline
    print("=" * 60)
    print("Running: RNN Baseline (LSTM)")
    print("=" * 60)
    rnn = RNNBaseline(hidden_size=32, epochs=10, batch_size=32)
    print("Training RNN on train split...")
    rnn.fit(X_train_in, y_train_next)
    
    # Evaluate on train
    preds_rnn_train = rnn.predict(X_train_in)
    metrics_rnn_train = evaluate_model(y_train_next, preds_rnn_train)
    
    # Evaluate on test
    preds_rnn_test = rnn.predict(X_test_in)
    metrics_rnn_test = evaluate_model(y_test_next, preds_rnn_test)
    
    results[rnn.name] = {
        "train": metrics_rnn_train,
        "test": metrics_rnn_test,
    }
    
    print(f"  Train - Mean RMSE: {metrics_rnn_train['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_rnn_train[f'rmse_ch{ch}']:.6f}")
    print(f"  Test  - Mean RMSE: {metrics_rnn_test['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_rnn_test[f'rmse_ch{ch}']:.6f}")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for model_name, metrics_dict in results.items():
        train_rmse = metrics_dict["train"]["mean_rmse"]
        test_rmse = metrics_dict["test"]["mean_rmse"]
        print(f"{model_name:20s}: train={train_rmse:.6f}  test={test_rmse:.6f}")
    print()
    
    # Save to JSON
    _save_metrics(results)
    
    return results


def _save_metrics(results: Dict[str, Dict]) -> None:
    """Save metrics to results/metrics/ as JSON files."""
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, metrics_dict in results.items():
        output_file = metrics_dir / f"{model_name}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        clean_metrics = {}
        for split_name, metrics in metrics_dict.items():
            clean_metrics[split_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        
        with open(output_file, "w") as f:
            json.dump(clean_metrics, f, indent=2)
        
        print(f"Saved metrics to: {output_file}")


if __name__ == "__main__":
    run_models()



