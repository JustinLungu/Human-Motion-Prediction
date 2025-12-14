"""Runner script to fit and evaluate simple models using the dataset loader.

Usage: run directly with Python to run the majority baseline as a smoke test.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
from typing import Tuple

import numpy as np

# Ensure `src` is on sys.path so we can import `dataloader` and `models` when
# running this script directly.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataloader.dataset import UCIHARDatasetLoader
from models.baseline import MajorityBaseline


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return simple accuracy metric."""
    return float(np.mean(y_true == y_pred))


def run_baseline(config_path: str = "configs/config.yaml") -> Tuple[float, int]:
    loader = UCIHARDatasetLoader(config_path)
    split = loader.load_split("train")
    X, y = split.X, split.y

    model = MajorityBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    acc = evaluate(y, preds)

    print("Baseline run completed:")
    print(f"  Model: {model.name}")
    print(f"  Train samples: {X.shape[0]}")
    print(f"  Accuracy (train): {acc:.4f}")

    return acc, X.shape[0]



