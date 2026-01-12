"""Factory functions for building models and loading data for experiments."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

# Ensure `src` is on sys.path
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataloader.dataset import UCIHARDatasetLoader
from models.persistence import PersistenceBaseline
from models.ekf import EKFBaseline
from models.pf import PFBaseline
from models.rnn import RNNBaseline


def split_xy(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split sequence into inputs (N, T-1, C) and targets (N, C)."""
    return X[:, :-1, :].astype(np.float32), X[:, -1, :].astype(np.float32)


def build_models(cfg: Dict) -> Dict[str, object]:
    """Build all models from configuration.
    
    Args:
        cfg: Configuration dictionary with model parameters
        
    Returns:
        Dictionary mapping model names to model instances
    """
    models_cfg = cfg.get("models", {})
    run_cfg = models_cfg.get("run", {})
    seed = run_cfg.get("seed", 42)
    
    persistence = PersistenceBaseline()
    
    rnn_cfg = models_cfg.get("rnn", {})
    rnn = RNNBaseline(
        hidden_size=int(rnn_cfg.get("hidden_size", 32)),
        epochs=int(rnn_cfg.get("epochs", 10)),
        batch_size=int(rnn_cfg.get("batch_size", 32)),
        learning_rate=float(rnn_cfg.get("learning_rate", 1e-3)),
        device=rnn_cfg.get("device"),
        seed=seed,
    )
    
    ekf_cfg = models_cfg.get("ekf", {})
    ekf = EKFBaseline(
        Q_scale=float(ekf_cfg.get("Q_scale", 1e-4)),
        R_scale=None if ekf_cfg.get("R_scale") is None else float(ekf_cfg.get("R_scale")),
        dt=float(ekf_cfg.get("dt", 1.0)),
    )
    
    pf_cfg = models_cfg.get("pf", {})
    pf = PFBaseline(
        num_particles=int(pf_cfg.get("num_particles", 100)),
        Q_scale=float(pf_cfg.get("Q_scale", 1.0)),
        R_scale=None if pf_cfg.get("R_scale") is None else float(pf_cfg.get("R_scale")),
        dt=float(pf_cfg.get("dt", 0.02)),
        resample_threshold=float(pf_cfg.get("resample_threshold", 0.5)),
        seed=seed,
    )
    
    return {"persistence": persistence, "ekf": ekf, "pf": pf, "rnn": rnn}


def load_data(cfg_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data for experiments.
    
    Args:
        cfg_path: Path to configuration file
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) where:
        - X_train: shape (N, T-1, 6) - training input sequences
        - y_train: shape (N, 6) - training targets
        - X_test: shape (N, T-1, 6) - test input sequences
        - y_test: shape (N, 6) - test targets
    """
    loader = UCIHARDatasetLoader(cfg_path)
    train_split = loader.load_split("train")
    test_split = loader.load_split("test")
    X_train_in, y_train = split_xy(train_split.X)
    X_test_in, y_test = split_xy(test_split.X)
    return X_train_in, y_train, X_test_in, y_test

