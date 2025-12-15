"""Sanity test for the degradation module.

This test implements the 5-line sanity check: RMSE for the persistence baseline
should increase as Gaussian noise sigma increases.

Run as: `python src/models/test_degrade.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src` is on sys.path so imports like `dataloader` resolve.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

import yaml

from dataloader.dataset import UCIHARDatasetLoader
from models.persistence import PersistenceBaseline
from models.evaluate import evaluate_model
from models.degrade import add_gaussian_noise, add_bias_drift, apply_dropout


def main() -> None:
    # Load config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    degrade_cfg = cfg.get("degrade", {})
    noise_sigmas = degrade_cfg.get("noise_sigmas", [0.01, 0.1])
    drift_rates = degrade_cfg.get("drift_rates", [0.05, 0.5])
    drop_cfg = degrade_cfg.get("dropout", {})
    drop_p = float(drop_cfg.get("p", 0.2))
    drop_max_gap = int(drop_cfg.get("max_gap", 5))
    drop_mode = drop_cfg.get("mode", "hold")
    drop_seed = drop_cfg.get("seed", 42)

    loader = UCIHARDatasetLoader("configs/config.yaml")
    split = loader.load_split("test")
    X_test = split.X.astype(np.float32)

    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]

    persistence = PersistenceBaseline()

    # --- Noise sanity check ---
    print("Running noise sanity check (persistence RMSE should increase with sigma)")
    preds_clean = persistence.predict(X_test_in)
    mean_clean = evaluate_model(y_test_next, preds_clean)["mean_rmse"]

    preds_small = persistence.predict(add_gaussian_noise(X_test_in, sigma=noise_sigmas[0], seed=0))
    mean_small = evaluate_model(y_test_next, preds_small)["mean_rmse"]

    preds_large = persistence.predict(add_gaussian_noise(X_test_in, sigma=noise_sigmas[1], seed=0))
    mean_large = evaluate_model(y_test_next, preds_large)["mean_rmse"]

    print(f"mean_rmse clean:  {mean_clean:.6f}")
    print(f"mean_rmse small:  {mean_small:.6f}")
    print(f"mean_rmse large:  {mean_large:.6f}")

    assert mean_small >= mean_clean - 1e-12, "small noise RMSE should not decrease"
    assert mean_large >= mean_small - 1e-12, "large noise RMSE should be >= small noise RMSE"
    print("Noise sanity passed.")

    # --- Drift sanity check ---
    print("Running drift sanity check (persistence RMSE should increase with drift_rate)")
    preds_d1 = persistence.predict(add_bias_drift(X_test_in, drift_rate=drift_rates[0], seed=1))
    mean_d1 = evaluate_model(y_test_next, preds_d1)["mean_rmse"]

    preds_d2 = persistence.predict(add_bias_drift(X_test_in, drift_rate=drift_rates[1], seed=1))
    mean_d2 = evaluate_model(y_test_next, preds_d2)["mean_rmse"]

    print(f"drift: clean={mean_clean:.6f}, d1={mean_d1:.6f}, d2={mean_d2:.6f}")
    assert mean_d1 >= mean_clean - 1e-12, "small drift should not reduce RMSE"
    assert mean_d2 >= mean_d1 - 1e-12, "larger drift should not reduce RMSE"
    print("Drift sanity passed.")

    # --- Dropout checks ---
    print("Running dropout checks (determinism, shape, no NaNs, RMSE non-decreasing)")
    out1 = apply_dropout(X_test_in, p=drop_p, max_gap=drop_max_gap, mode=drop_mode, seed=drop_seed)
    out2 = apply_dropout(X_test_in, p=drop_p, max_gap=drop_max_gap, mode=drop_mode, seed=drop_seed)
    assert np.array_equal(out1, out2), "apply_dropout should be deterministic with same seed"
    assert out1.shape == X_test_in.shape
    assert not np.isnan(out1).any(), "dropout must not introduce NaNs"

    mean_drop = evaluate_model(y_test_next, persistence.predict(out1))["mean_rmse"]
    print(f"dropout: clean={mean_clean:.6f}, drop={mean_drop:.6f}")
    assert mean_drop >= mean_clean - 1e-12, "dropout should not reduce RMSE"

    print("All degrade tests passed.")


if __name__ == "__main__":
    main()
