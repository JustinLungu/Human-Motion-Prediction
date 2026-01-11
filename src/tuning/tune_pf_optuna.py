"""Optuna-based tuner for Particle Filter baseline.

Replaces grid search with efficient Bayesian optimization.
Tunes num_particles, Q_scale, and R_scale with proper validation split.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import optuna
from optuna.samplers import TPESampler

from dataloader.dataset import UCIHARDatasetLoader
from models.pf import PFBaseline
from models.evaluate import evaluate_model


def create_validation_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple:
    """Split training data into train and validation sets."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)
    
    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    return (
        X[train_indices], y[train_indices],
        X[val_indices], y[val_indices]
    )


def tune_pf(n_trials: int = 50, seed: int = 42) -> None:
    """Tune PF using Optuna optimization.
    
    Args:
        n_trials: Number of optimization trials
        seed: Random seed for reproducibility
    """
    loader = UCIHARDatasetLoader("configs/config.yaml")
    
    # Load training data
    train_split = loader.load_split("train")
    X_train_full = train_split.X.astype(np.float32)
    X_train_in_full = X_train_full[:, :-1, :]
    y_train_next_full = X_train_full[:, -1, :]
    
    # Create validation split (important: don't optimize on test set!)
    X_train_in, y_train_next, X_val_in, y_val_next = create_validation_split(
        X_train_in_full, y_train_next_full, val_ratio=0.2, seed=seed
    )
    
    # Load test data
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)
    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]
    
    print("=" * 90)
    print("PF Tuning (Optuna Optimization)")
    print("=" * 90)
    print(f"Train samples:      {X_train_in.shape[0]}")
    print(f"Validation samples: {X_val_in.shape[0]}")
    print(f"Test samples:       {X_test_in.shape[0]}")
    print(f"Number of trials:   {n_trials}")
    print()
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        """Optimize on validation RMSE."""
        # Parameter ranges based on your original grid
        num_particles = trial.suggest_int("num_particles", 100, 500, log=True)
        Q_scale = trial.suggest_float("Q_scale", 1.0, 10.0, log=True)
        R_scale = trial.suggest_float("R_scale", 0.5, 2.0)
        
        try:
            pf = PFBaseline(
                Q_scale=Q_scale,
                R_scale=R_scale,
                num_particles=num_particles,
                dt=0.02,
                resample_threshold=0.5,
                seed=seed,
            )
            pf.fit(X_train_in, y_train_next)
            
            preds_val = pf.predict(X_val_in)
            val_rmse = evaluate_model(y_val_next, preds_val)["mean_rmse"]
            
            # Check for numerical issues
            if np.isnan(val_rmse) or np.isinf(val_rmse):
                return float('inf')
            
            return val_rmse
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    # Create and run study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    print("Running optimization...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print()
    print("=" * 90)
    print("Optimization Results")
    print("=" * 90)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation RMSE: {study.best_value:.6f}")
    print()
    
    best_params = study.best_params
    print("Best hyperparameters:")
    print(f"  num_particles: {best_params['num_particles']}")
    print(f"  Q_scale:       {best_params['Q_scale']:.2f}")
    print(f"  R_scale:       {best_params['R_scale']:.2f}")
    print()
    
    # Retrain on full training set with best params
    print("Retraining best model on full training set...")
    pf_best = PFBaseline(
        Q_scale=best_params['Q_scale'],
        R_scale=best_params['R_scale'],
        num_particles=best_params['num_particles'],
        dt=0.02,
        resample_threshold=0.5,
        seed=seed,
    )
    pf_best.fit(X_train_in_full, y_train_next_full)
    
    # Evaluate on all sets
    preds_train = pf_best.predict(X_train_in_full)
    train_rmse = evaluate_model(y_train_next_full, preds_train)["mean_rmse"]
    
    preds_test = pf_best.predict(X_test_in)
    test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]
    
    print()
    print("Final performance:")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print()
    
    # Compare with persistence
    persistence_preds = X_test_in[:, -1, :]
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]
    print(f"Persistence baseline test RMSE: {persistence_rmse:.6f}")
    
    if test_rmse < persistence_rmse:
        improvement = persistence_rmse - test_rmse
        print(f"✓ PF beats persistence by {improvement:.6f}")
    else:
        decline = test_rmse - persistence_rmse
        print(f"✗ PF underperforms persistence by {decline:.6f}")
    print()
    
    # Show top 5 trials
    print("Top 5 trials:")
    print(f"{'Trial':<8} {'Particles':<12} {'Q_scale':<12} {'R_scale':<12} {'Val RMSE':<15}")
    print("-" * 90)
    
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for trial in sorted_trials[:5]:
        if trial.value is not None and trial.value != float('inf'):
            params = trial.params
            print(f"{trial.number:<8d} {params['num_particles']:<12d} "
                  f"{params['Q_scale']:<12.2f} {params['R_scale']:<12.2f} "
                  f"{trial.value:<15.6f}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune PF hyperparameters using Optuna")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of optimization trials (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    tune_pf(n_trials=args.n_trials, seed=args.seed)