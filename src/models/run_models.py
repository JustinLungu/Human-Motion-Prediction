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
import yaml
import random

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
from models.ekf import EKFBaseline
from models.pf import PFBaseline


def run_models(config_path: str = "configs/config.yaml") -> Dict[str, Dict]:
    """Load data, train on train split, evaluate on test split.
    
    Task definition:
      - Input window: X_in = X[:, :-1, :] shape (N, T-1, 6)
      - Target next-step: y_next = X[:, -1, :] shape (N, 6)
      - Metric: RMSE (per-channel and mean)
    
    Returns:
        dict mapping model name to dict of train and test metrics
    """
    # Load config
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # Optional seeding
    seed = cfg.get("models", {}).get("run", {}).get("seed")
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

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
    
    T_train = X_train.shape[1]
    T_test = X_test.shape[1]
    print(f"Train split: {X_train_in.shape[0]} samples, shape (N, T-1={T_train-1}, 6) (T_train={T_train})")
    print(f"Test split:  {X_test_in.shape[0]} samples, shape (N, T-1={T_test-1}, 6) (T_test={T_test})")
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
    _sanity_check_preds(y_train_next, preds_persist_train, persistence.name, "train")
    metrics_persist_train = evaluate_model(y_train_next, preds_persist_train)
    
    # Evaluate on test
    preds_persist_test = persistence.predict(X_test_in)
    _sanity_check_preds(y_test_next, preds_persist_test, persistence.name, "test")
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
    # Build RNN from config
    rnn_cfg = cfg.get("models", {}).get("rnn", {})
    rnn_hidden = int(rnn_cfg.get("hidden_size", 32))
    rnn_epochs = int(rnn_cfg.get("epochs", 10))
    rnn_batch = int(rnn_cfg.get("batch_size", 32))
    rnn_lr = float(rnn_cfg.get("learning_rate", 1e-3))
    rnn_device = rnn_cfg.get("device", None)

    rnn = RNNBaseline(
        hidden_size=rnn_hidden,
        epochs=rnn_epochs,
        batch_size=rnn_batch,
        learning_rate=rnn_lr,
        device=rnn_device,
    )
    print("Training RNN on train split...")
    rnn.fit(X_train_in, y_train_next)
    
    # Evaluate on train
    preds_rnn_train = rnn.predict(X_train_in)
    _sanity_check_preds(y_train_next, preds_rnn_train, rnn.name, "train")
    metrics_rnn_train = evaluate_model(y_train_next, preds_rnn_train)
    
    # Evaluate on test
    preds_rnn_test = rnn.predict(X_test_in)
    _sanity_check_preds(y_test_next, preds_rnn_test, rnn.name, "test")
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
    
    # Test 3: EKF baseline
    print("=" * 60)
    print("Running: EKF Baseline")
    print("=" * 60)
    # Build EKF from config
    ekf_cfg = cfg.get("models", {}).get("ekf", {})
    ekf_Q_scale = float(ekf_cfg.get("Q_scale", 1e-4))
    ekf_R_scale = ekf_cfg.get("R_scale")
    if ekf_R_scale is not None:
        ekf_R_scale = float(ekf_R_scale)
    ekf_dt = float(ekf_cfg.get("dt", 1.0))

    ekf = EKFBaseline(Q_scale=ekf_Q_scale, R_scale=ekf_R_scale, dt=ekf_dt)
    print("Fitting EKF on train split...")
    ekf.fit(X_train_in, y_train_next)
    
    # Evaluate on train
    preds_ekf_train = ekf.predict(X_train_in)
    _sanity_check_preds(y_train_next, preds_ekf_train, ekf.name, "train")
    metrics_ekf_train = evaluate_model(y_train_next, preds_ekf_train)
    
    # Evaluate on test
    preds_ekf_test = ekf.predict(X_test_in)
    _sanity_check_preds(y_test_next, preds_ekf_test, ekf.name, "test")
    metrics_ekf_test = evaluate_model(y_test_next, preds_ekf_test)
    
    results[ekf.name] = {
        "train": metrics_ekf_train,
        "test": metrics_ekf_test,
    }
    
    print(f"  Train - Mean RMSE: {metrics_ekf_train['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_ekf_train[f'rmse_ch{ch}']:.6f}")
    print(f"  Test  - Mean RMSE: {metrics_ekf_test['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_ekf_test[f'rmse_ch{ch}']:.6f}")
    print()
    
    # Test 4: PF baseline
    print("=" * 60)
    print("Running: Particle Filter Baseline")
    print("=" * 60)
    # Build PF from config
    pf_cfg = cfg.get("models", {}).get("pf", {})
    pf_num_particles = int(pf_cfg.get("num_particles", 100))
    pf_Q_scale = float(pf_cfg.get("Q_scale", 1.0))
    pf_R_scale = pf_cfg.get("R_scale")
    if pf_R_scale is not None:
        pf_R_scale = float(pf_R_scale)
    pf_dt = float(pf_cfg.get("dt", 0.02))
    pf_resample_threshold = float(pf_cfg.get("resample_threshold", 0.5))
    pf_seed = cfg.get("models", {}).get("run", {}).get("seed")

    pf = PFBaseline(
        num_particles=pf_num_particles,
        Q_scale=pf_Q_scale,
        R_scale=pf_R_scale,
        dt=pf_dt,
        resample_threshold=pf_resample_threshold,
        seed=pf_seed,
    )
    print("Fitting PF on train split...")
    pf.fit(X_train_in, y_train_next)
    
    # Evaluate on train
    preds_pf_train = pf.predict(X_train_in)
    _sanity_check_preds(y_train_next, preds_pf_train, pf.name, "train")
    metrics_pf_train = evaluate_model(y_train_next, preds_pf_train)
    
    # Evaluate on test
    preds_pf_test = pf.predict(X_test_in)
    _sanity_check_preds(y_test_next, preds_pf_test, pf.name, "test")
    metrics_pf_test = evaluate_model(y_test_next, preds_pf_test)
    
    results[pf.name] = {
        "train": metrics_pf_train,
        "test": metrics_pf_test,
    }
    
    print(f"  Train - Mean RMSE: {metrics_pf_train['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_pf_train[f'rmse_ch{ch}']:.6f}")
    print(f"  Test  - Mean RMSE: {metrics_pf_test['mean_rmse']:.6f}")
    for ch in range(6):
        print(f"    ch{ch}: {metrics_pf_test[f'rmse_ch{ch}']:.6f}")
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
    
    # Save to JSON (location from config or default)
    metrics_dir = cfg.get("models", {}).get("run", {}).get("metrics_dir", "results/metrics")
    _save_metrics(results, metrics_dir)
    
    return results


def _save_metrics(results: Dict[str, Dict], metrics_dir: str = "results/metrics") -> None:
    """Save metrics to results/metrics/ as JSON files."""
    metrics_dir = Path(metrics_dir)
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


def _sanity_check_preds(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split: str) -> None:
    """Basic sanity checks before evaluation:
    - predictions shape (N, 6)
    - no NaNs in preds or targets
    Raises RuntimeError on failure.
    """
    # Shape check
    if y_pred.ndim != 2 or y_pred.shape[1] != 6 or y_pred.shape[0] != y_true.shape[0]:
        raise RuntimeError(
            f"Sanity check failed for {model_name} on {split}: expected preds shape (N,6) matching y_true, got {y_pred.shape} vs {y_true.shape}"
        )

    # NaN check
    if np.isnan(y_pred).any():
        raise RuntimeError(f"Sanity check failed for {model_name} on {split}: preds contain NaN")
    if np.isnan(y_true).any():
        raise RuntimeError(f"Sanity check failed for {model_name} on {split}: targets contain NaN")

    # Quick success print
    print(f"Sanity checks passed for {model_name} on {split}: preds shape {y_pred.shape}, no NaNs")


if __name__ == "__main__":
    run_models()



