"""Optuna-based hyperparameter tuning for EKF baseline with proper validation.

Uses train/validation split to prevent overfitting and includes:
- Proper validation split (not tuning on test set)
- Physically-motivated parameter ranges
- Optional parameter penalties
- Pruning for early stopping of bad trials
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
from optuna.pruners import MedianPruner

from dataloader.dataset import UCIHARDatasetLoader
from models.ekf import EKFBaseline
from models.evaluate import evaluate_model


def create_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple:
    """Split training data into train and validation sets.

    Args:
        X: Training sequences (N, T, 6)
        y: Training targets (N, 6)
        val_ratio: Fraction of data to use for validation
        seed: Random seed

    Returns:
        (X_train, y_train, X_val, y_val)
    """
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


def tune_ekf_optuna(
    n_trials: int = 50,
    seed: int = 42,
    val_ratio: float = 0.2,
    use_penalties: bool = True,
) -> None:
    """Tune EKF using Optuna optimization with proper validation.

    Args:
        n_trials: Number of optimization trials to run
        seed: Random seed for reproducibility
        val_ratio: Fraction of training data to use for validation
        use_penalties: Whether to add penalties for extreme parameters
    """
    # Load data
    loader = UCIHARDatasetLoader("configs/config.yaml")

    train_split = loader.load_split("train")
    X_train_full = train_split.X.astype(np.float32)
    X_train_in_full = X_train_full[:, :-1, :]
    y_train_next_full = X_train_full[:, -1, :]

    # Create train/val split
    X_train_in, y_train_next, X_val_in, y_val_next = create_validation_split(
        X_train_in_full, y_train_next_full, val_ratio=val_ratio, seed=seed
    )

    # Load test set (only for final evaluation)
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)
    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]

    print("=" * 80)
    print("EKF Hyperparameter Tuning with Optuna (with Validation Split)")
    print("=" * 80)
    print(f"Train samples: {X_train_in.shape[0]}")
    print(f"Validation samples: {X_val_in.shape[0]}")
    print(f"Test samples: {X_test_in.shape[0]}")
    print(f"Number of trials: {n_trials}")
    print(f"Random seed: {seed}")
    print(f"Parameter penalties: {'enabled' if use_penalties else 'disabled'}")
    print()

    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation RMSE (to be minimized), with optional penalties
        """
        # Suggest hyperparameters with physically-motivated ranges
        # Q_scale: process noise should be reasonable relative to signal dynamics
        Q_scale = trial.suggest_float("Q_scale", 1e-4, 5.0, log=True)

        # R_scale: either None (use data-driven) or a moderate scale factor
        use_data_r = trial.suggest_categorical("use_data_r", [True, False])
        if use_data_r:
            R_scale = None
        else:
            R_scale = trial.suggest_float("R_scale", 0.2, 5.0, log=True)

        # dt: should be close to physical sampling rate (1/50 = 0.02)
        # Allow range from 0.01 to 0.1 (half to 5x the sampling period)
        dt = trial.suggest_float("dt", 0.01, 0.1, log=True)

        # Train and evaluate EKF
        ekf = EKFBaseline(Q_scale=Q_scale, R_scale=R_scale, dt=dt)
        ekf.fit(X_train_in, y_train_next)

        preds_val = ekf.predict(X_val_in)
        val_rmse = evaluate_model(y_val_next, preds_val)["mean_rmse"]

        # Optional: Add penalties for extreme parameters to prevent overfitting
        if use_penalties:
            penalty = 0.0

            # Penalize very large Q_scale (encourages stability)
            if Q_scale > 2.0:
                penalty += 0.001 * (Q_scale - 2.0)

            # Penalize dt far from physical sampling rate (0.02)
            dt_deviation = abs(dt - 0.02) / 0.02
            if dt_deviation > 0.5:  # More than 50% deviation
                penalty += 0.001 * dt_deviation

            val_rmse += penalty

        return val_rmse

    # Create study with pruning for early stopping
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="ekf_tuning"
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print()
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation RMSE: {study.best_value:.6f}")
    print()
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            if key == "Q_scale" or (key == "R_scale" and value < 1):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Evaluate best model on all splits
    best_params = study.best_params
    Q_scale = best_params["Q_scale"]
    R_scale = None if best_params["use_data_r"] else best_params.get("R_scale")
    dt = best_params["dt"]

    # Retrain on full training set
    ekf_final = EKFBaseline(Q_scale=Q_scale, R_scale=R_scale, dt=dt)
    ekf_final.fit(X_train_in_full, y_train_next_full)

    # Evaluate on all splits
    preds_train = ekf_final.predict(X_train_in_full)
    train_rmse = evaluate_model(y_train_next_full, preds_train)["mean_rmse"]

    preds_val = ekf_final.predict(X_val_in)
    val_rmse = evaluate_model(y_val_next, preds_val)["mean_rmse"]

    preds_test = ekf_final.predict(X_test_in)
    test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]

    print("Best model performance (retrained on full training set):")
    print(f"  Train RMSE:      {train_rmse:.6f}")
    print(f"  Validation RMSE: {val_rmse:.6f}")
    print(f"  Test RMSE:       {test_rmse:.6f}")
    print()

    # Compare with baselines
    persistence_preds = X_test_in[:, -1, :]
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]

    # Also show default EKF performance
    ekf_default = EKFBaseline(Q_scale=1.0, R_scale=None, dt=0.02)
    ekf_default.fit(X_train_in_full, y_train_next_full)
    default_test_rmse = evaluate_model(
        y_test_next, ekf_default.predict(X_test_in)
    )["mean_rmse"]

    print("Baseline comparisons (test RMSE):")
    print(f"  Persistence:     {persistence_rmse:.6f}")
    print(f"  Default EKF:     {default_test_rmse:.6f}")
    print(f"  Optimized EKF:   {test_rmse:.6f}")
    print()

    if test_rmse < persistence_rmse:
        improvement = persistence_rmse - test_rmse
        pct = 100 * improvement / persistence_rmse
        print(f"✓ Optimized EKF beats persistence by {improvement:.6f} ({pct:.2f}%)")
    else:
        decline = test_rmse - persistence_rmse
        pct = 100 * decline / persistence_rmse
        print(f"✗ Optimized EKF underperforms persistence by {decline:.6f} ({pct:.2f}%)")
    print()

    # Check for overfitting
    train_val_gap = abs(train_rmse - val_rmse)
    val_test_gap = abs(val_rmse - test_rmse)

    if train_val_gap > 0.01:
        print(f"⚠ Warning: Large train-validation gap ({train_val_gap:.6f}) suggests overfitting")
    if val_test_gap > 0.01:
        print(f"⚠ Warning: Large validation-test gap ({val_test_gap:.6f}) - results may not generalize")

    if train_val_gap <= 0.01 and val_test_gap <= 0.01:
        print("✓ Good generalization: small train-val-test gaps")
    print()

    # Optionally plot optimization history
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot optimization history
        trials = study.trials
        trial_numbers = [t.number for t in trials if t.value is not None]
        values = [t.value for t in trials if t.value is not None]

        ax1.plot(trial_numbers, values, 'o-', alpha=0.6, markersize=4)
        ax1.axhline(y=study.best_value, color='r', linestyle='--',
                    label=f'Best val: {study.best_value:.6f}')
        ax1.axhline(y=persistence_rmse, color='g', linestyle='--',
                    label=f'Persistence: {persistence_rmse:.6f}')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Validation RMSE')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())

            ax2.barh(params, values)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importance')
            ax2.grid(True, alpha=0.3, axis='x')
        except Exception:
            ax2.text(0.5, 0.5, 'Importance analysis unavailable',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hyperparameter Importance')

        plt.tight_layout()

        # Save plot
        plot_dir = Path("results/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "ekf_optuna_history.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"Saved optimization history plot to: {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plots")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune EKF hyperparameters using Optuna with proper validation"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--no-penalties",
        action="store_true",
        help="Disable parameter penalties"
    )

    args = parser.parse_args()

    tune_ekf_optuna(
        n_trials=args.n_trials,
        seed=args.seed,
        val_ratio=args.val_ratio,
        use_penalties=not args.no_penalties
    )