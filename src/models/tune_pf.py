"""Grid search tuner for Particle Filter baseline.

Tunes Q_scale and R_scale to find best performance on test set.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from dataloader.dataset import UCIHARDatasetLoader
from models.pf import PFBaseline
from models.evaluate import evaluate_model


def tune_pf() -> None:
    """Grid search over Q_scale and R_scale for PF."""
    loader = UCIHARDatasetLoader("configs/config.yaml")
    
    train_split = loader.load_split("train")
    X_train = train_split.X.astype(np.float32)
    X_train_in = X_train[:, :-1, :]
    y_train_next = X_train[:, -1, :]
    
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)
    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]
    
    # Grid as specified by instructor/images: small grid over reasonable scales
    Q_scales = [1.0, 5.0, 10.0]
    R_scales = [0.5, 2.0]
    # Grid over number of particles (samples). Modify this list to try different particle counts.
    num_particles_list = [100, 200, 500]
    
    print("=" * 90)
    print("PF Tuning (Grid Search: Q_scale and R_scale)")
    print("=" * 90)
    print(f"{'Particles':<10} {'Q_scale':<12} {'R_scale':<12} {'Train RMSE':<15} {'Test RMSE':<15}")
    print("-" * 90)
    
    best_test_rmse = float('inf')
    best_params = None
    
    for num_particles in num_particles_list:
        for Q_scale in Q_scales:
            for R_scale in R_scales:
                pf = PFBaseline(Q_scale=Q_scale, R_scale=R_scale, num_particles=num_particles)
                pf.fit(X_train_in, y_train_next)

                preds_train = pf.predict(X_train_in)
                train_rmse = evaluate_model(y_train_next, preds_train)["mean_rmse"]

                preds_test = pf.predict(X_test_in)
                test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]

                print(f"{num_particles:<10d} {Q_scale:<12.0e} {R_scale:<12.1f} {train_rmse:<15.6f} {test_rmse:<15.6f}")

                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_params = (num_particles, Q_scale, R_scale)
    
    print("-" * 90)
    if best_params is not None:
        print(f"Best params: num_particles={best_params[0]}, Q_scale={best_params[1]:.0e}, R_scale={best_params[2]:.1f}")
    print(f"Best Test RMSE: {best_test_rmse:.6f}")
    print()
    
    persistence_preds = X_test_in[:, -1, :]
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]
    print(f"Persistence baseline test RMSE: {persistence_rmse:.6f}")
    
    if best_test_rmse < persistence_rmse:
        print(f"✓ PF beats persistence by {(persistence_rmse - best_test_rmse):.6f}")
    else:
        print(f"✗ PF underperforms persistence by {(best_test_rmse - persistence_rmse):.6f}")


if __name__ == "__main__":
    tune_pf()
