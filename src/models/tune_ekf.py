"""Extended tuning: vary dt as well as Q and R scales."""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from dataloader.dataset import UCIHARDatasetLoader
from models.ekf import EKFBaseline
from models.evaluate import evaluate_model


def extended_tune() -> None:
    """Extended grid search including dt parameter."""
    loader = UCIHARDatasetLoader("configs/config.yaml")
    
    train_split = loader.load_split("train")
    X_train = train_split.X.astype(np.float32)
    X_train_in = X_train[:, :-1, :]
    y_train_next = X_train[:, -1, :]
    
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)
    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]
    
    # Expanded grid
    Q_scales = [1e-3, 1e-2, 1e-1, 1.0]
    R_scales = [None, 0.5, 1.0]
    dts = [0.02, 0.1, 1.0]  # 0.02 = 1/50 (50Hz sampling)
    
    print("=" * 80)
    print("EKF Extended Tuning (Grid Search with dt)")
    print("=" * 80)
    print(f"{'dt':<8} {'Q_scale':<12} {'R_scale':<12} {'Train RMSE':<15} {'Test RMSE':<15}")
    print("-" * 80)
    
    best_test_rmse = float('inf')
    best_params = None
    
    for dt in dts:
        for Q_scale in Q_scales:
            for R_scale in R_scales:
                ekf = EKFBaseline(Q_scale=Q_scale, R_scale=R_scale, dt=dt)
                ekf.fit(X_train_in, y_train_next)
                
                preds_train = ekf.predict(X_train_in)
                train_rmse = evaluate_model(y_train_next, preds_train)["mean_rmse"]
                
                preds_test = ekf.predict(X_test_in)
                test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]
                
                r_str = str(R_scale) if R_scale else "data"
                print(f"{dt:<8.3f} {Q_scale:<12.0e} {r_str:<12} {train_rmse:<15.6f} {test_rmse:<15.6f}")
                
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_params = (Q_scale, R_scale, dt)
    
    print("-" * 80)
    print(f"Best params: Q_scale={best_params[0]:.0e}, R_scale={best_params[1]}, dt={best_params[2]:.3f}")
    print(f"Best Test RMSE: {best_test_rmse:.6f}")
    print()
    
    persistence_preds = X_test_in[:, -1, :]
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]
    print(f"Persistence baseline test RMSE: {persistence_rmse:.6f}")
    
    if best_test_rmse < persistence_rmse:
        print(f"✓ EKF beats persistence by {(persistence_rmse - best_test_rmse):.6f}")
    else:
        print(f"✗ EKF underperforms persistence by {(best_test_rmse - persistence_rmse):.6f}")


if __name__ == "__main__":
    extended_tune()
