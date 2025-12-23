"""
Runner script to fit and evaluate next-step IMU prediction models.

Models tested:
  - Persistence
  - RNN (LSTM)
  - EKF
  - PF

Pipeline:
  1. Load config
  2. Set random seeds
  3. Load train/test splits
  4. Fit & evaluate models
  5. Save metrics to JSON
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from dataloader.dataset import UCIHARDatasetLoader
from models.evaluate import evaluate_model

import PersistenceBaseline, EKFBaseline, RNNBaseline, PFBaseline


# ---------------------------------------------------------------------
# Configuration & Utilities
# ---------------------------------------------------------------------

CONFIG_PATH = "configs/config.yaml"


def load_config(path: str = CONFIG_PATH) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return

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


def split_xy(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split sequence into:
      inputs:  (N, T-1, C)
      targets: (N, C)  (next-step prediction)
    """
    return (
        X[:, :-1, :].astype(np.float32),
        X[:, -1, :].astype(np.float32),
    )


# ---------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------

def build_models(cfg: Dict) -> Dict[str, object]:
    models_cfg = cfg.get("models", {})

    # --- RNN ---
    rnn_cfg = models_cfg.get("rnn", {})
    rnn = RNNBaseline(
        hidden_size=int(rnn_cfg.get("hidden_size", 32)),
        epochs=int(rnn_cfg.get("epochs", 10)),
        batch_size=int(rnn_cfg.get("batch_size", 32)),
        learning_rate=float(rnn_cfg.get("learning_rate", 1e-3)),
        device=rnn_cfg.get("device"),
    )

    # --- EKF ---
    ekf_cfg = models_cfg.get("ekf", {})
    ekf = EKFBaseline(
        Q_scale=float(ekf_cfg.get("Q_scale", 1e-4)),
        R_scale=_maybe_float(ekf_cfg.get("R_scale")),
        dt=float(ekf_cfg.get("dt", 1.0)),
    )

    # --- PF ---
    pf_cfg = models_cfg.get("pf", {})
    pf = PFBaseline(
        num_particles=int(pf_cfg.get("num_particles", 100)),
        Q_scale=float(pf_cfg.get("Q_scale", 1.0)),
        R_scale=_maybe_float(pf_cfg.get("R_scale")),
        dt=float(pf_cfg.get("dt", 0.02)),
        resample_threshold=float(pf_cfg.get("resample_threshold", 0.5)),
        seed=models_cfg.get("run", {}).get("seed"),
    )

    return {
        "Persistence": PersistenceBaseline(),
        "EKF": ekf,
        "RNN": rnn,
        "PF": pf,
    }


def _maybe_float(x):
    return None if x is None else float(x)


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def load_data(cfg_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = UCIHARDatasetLoader(cfg_path)

    train_split = loader.load_split("train")
    test_split = loader.load_split("test")

    X_train_in, y_train = split_xy(train_split.X)
    X_test_in, y_test = split_xy(test_split.X)

    return X_train_in, y_train, X_test_in, y_test


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def sanity_check(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str,
) -> None:
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise RuntimeError(
            f"{model_name} ({split}): invalid prediction shape {y_pred.shape}"
        )
    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        raise RuntimeError(f"{model_name} ({split}): NaNs detected")


def run_single_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    model.fit(X_train, y_train)

    preds_train = model.predict(X_train)
    sanity_check(y_train, preds_train, model.name, "train")
    train_metrics = evaluate_model(y_train, preds_train)

    preds_test = model.predict(X_test)
    sanity_check(y_test, preds_test, model.name, "test")
    test_metrics = evaluate_model(y_test, preds_test)

    return {
        "train": train_metrics,
        "test": test_metrics,
    }


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def save_metrics(results: Dict[str, Dict], out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model_name, metrics in results.items():
        file = out_path / f"{model_name}.json"
        with open(file, "w") as f:
            json.dump(_to_python(metrics), f, indent=2)

        print(f"Saved metrics → {file}")


def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def run_experiment(config_path: str = CONFIG_PATH) -> Dict[str, Dict]:
    cfg = load_config(config_path)

    seed = cfg.get("models", {}).get("run", {}).get("seed")
    set_global_seed(seed)

    X_train, y_train, X_test, y_test = load_data(config_path)
    models = build_models(cfg)

    results = {}

    for name, model in models.items():
        print("=" * 60)
        print(f"Running model: {name}")
        print("=" * 60)

        results[model.name] = run_single_model(
            model, X_train, y_train, X_test, y_test
        )

        print(
            f"{model.name:15s} | "
            f"train RMSE={results[model.name]['train']['mean_rmse']:.6f} | "
            f"test RMSE={results[model.name]['test']['mean_rmse']:.6f}"
        )

    metrics_dir = cfg.get("models", {}).get("run", {}).get(
        "metrics_dir", "results/metrics"
    )
    save_metrics(results, metrics_dir)

    return results


if __name__ == "__main__":
    run_experiment()
