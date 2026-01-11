"""
Comprehensive experiments runner for next-step IMU prediction models.

Models tested:
  - Persistence
  - EKF
  - PF
  - RNN

Conditions tested:
  - Nominal (baseline)
  - Noise (various sigma values)
  - Dropout (various p values)
  - Drift (various drift rates)

Outputs:
  - results/metrics/metrics.json
  - Plots: error vs noise, error vs dropout, filtered trace plots
"""

from __future__ import annotations

import json
import os, sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure `src` is on sys.path so we can import `dataloader` and `models` when
# running this script directly.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataloader.dataset import UCIHARDatasetLoader
from models.evaluate import evaluate_model
from models.persistence import PersistenceBaseline
from models.ekf import EKFBaseline
from models.pf import PFBaseline
from models.rnn import RNNBaseline
from models.degrade import add_gaussian_noise, add_bias_drift, apply_dropout


# ---------------------------------------------------------------------
# Configuration & Utilities
# ---------------------------------------------------------------------

CONFIG_PATH = "configs/config.yaml"


def load_config(path: str = CONFIG_PATH) -> Dict:
    """Load YAML configuration file."""
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(path):
        project_root = Path(__file__).resolve().parents[2]  # Go up from src/models/experiments.py
        path = str(project_root / path)
    
    if not os.path.exists(path):
        print(f"Warning: Config file not found at {path}, using defaults")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def set_global_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility (numpy and Python random).
    
    Note: PyTorch seeding is handled separately by the RNN model if needed,
    to avoid PyTorch dependency in this module.
    """
    if seed is None:
        return

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_xy(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split sequence into inputs (N, T-1, C) and targets (N, C)."""
    return (
        X[:, :-1, :].astype(np.float32),
        X[:, -1, :].astype(np.float32),
    )


# ---------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------

def build_models(cfg: Dict) -> Dict[str, object]:
    """Build all models from configuration."""
    models_cfg = cfg.get("models", {})

    # Persistence
    persistence = PersistenceBaseline()

    # RNN
    rnn_cfg = models_cfg.get("rnn", {})
    rnn_seed = models_cfg.get("run", {}).get("seed")
    rnn = RNNBaseline(
        hidden_size=int(rnn_cfg.get("hidden_size", 32)),
        epochs=int(rnn_cfg.get("epochs", 10)),
        batch_size=int(rnn_cfg.get("batch_size", 32)),
        learning_rate=float(rnn_cfg.get("learning_rate", 1e-3)),
        device=rnn_cfg.get("device"),
        seed=rnn_seed,
    )

    # EKF
    ekf_cfg = models_cfg.get("ekf", {})
    ekf = EKFBaseline(
        Q_scale=float(ekf_cfg.get("Q_scale", 1e-4)),
        R_scale=_maybe_float(ekf_cfg.get("R_scale")),
        dt=float(ekf_cfg.get("dt", 1.0)),
    )

    # PF
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
        "persistence": persistence,
        "ekf": ekf,
        "pf": pf,
        "rnn": rnn,
    }


def _maybe_float(x):
    """Convert to float or return None."""
    return None if x is None else float(x)


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def load_data(cfg_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data."""
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
    """Basic sanity checks."""
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise RuntimeError(
            f"{model_name} ({split}): invalid prediction shape {y_pred.shape}"
        )
    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        raise RuntimeError(f"{model_name} ({split}): NaNs detected")


def evaluate_model_on_data(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Evaluate a model on test data."""
    preds = model.predict(X_test)
    sanity_check(y_test, preds, model.name, "test")
    return evaluate_model(y_test, preds)


# ---------------------------------------------------------------------
# Degradation Experiments
# ---------------------------------------------------------------------

def run_nominal_experiments(
    models: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    """Run experiments on nominal (clean) data."""
    print("\n" + "=" * 60)
    print("NOMINAL CONDITIONS")
    print("=" * 60)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nFitting {model_name}...")
        model.fit(X_train, y_train)
        
        metrics = evaluate_model_on_data(model, X_test, y_test)
        results[model_name] = {
            "condition": "nominal",
            "mean_rmse": metrics["mean_rmse"],
            "metrics": metrics,
        }
        print(f"  {model_name}: RMSE = {metrics['mean_rmse']:.6f}")

    return results


def run_noise_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_sigmas: List[float],
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run experiments with Gaussian noise."""
    print("\n" + "=" * 60)
    print("NOISE EXPERIMENTS")
    print("=" * 60)

    results = {}
    
    for sigma in noise_sigmas:
        print(f"\nNoise sigma = {sigma:.4f}")
        X_test_noisy = add_gaussian_noise(X_test, sigma=sigma, seed=seed)
        
        for model_name, model in models.items():
            metrics = evaluate_model_on_data(model, X_test_noisy, y_test)
            
            key = f"{model_name}_noise_{sigma:.4f}"
            results[key] = {
                "condition": "noise",
                "sigma": sigma,
                "model": model_name,
                "mean_rmse": metrics["mean_rmse"],
                "metrics": metrics,
            }
            print(f"  {model_name}: RMSE = {metrics['mean_rmse']:.6f}")

    return results


def run_dropout_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    dropout_ps: List[float],
    max_gap: int = 5,
    mode: str = "hold",
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run experiments with dropout."""
    print("\n" + "=" * 60)
    print("DROPOUT EXPERIMENTS")
    print("=" * 60)

    results = {}
    
    for p in dropout_ps:
        print(f"\nDropout p = {p:.4f}")
        X_test_dropped = apply_dropout(X_test, p=p, max_gap=max_gap, mode=mode, seed=seed)
        
        for model_name, model in models.items():
            metrics = evaluate_model_on_data(model, X_test_dropped, y_test)
            
            key = f"{model_name}_dropout_{p:.4f}"
            results[key] = {
                "condition": "dropout",
                "p": p,
                "model": model_name,
                "mean_rmse": metrics["mean_rmse"],
                "metrics": metrics,
            }
            print(f"  {model_name}: RMSE = {metrics['mean_rmse']:.6f}")

    return results


def run_drift_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    drift_rates: List[float],
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run experiments with bias drift."""
    print("\n" + "=" * 60)
    print("DRIFT EXPERIMENTS")
    print("=" * 60)

    results = {}
    
    for drift_rate in drift_rates:
        print(f"\nDrift rate = {drift_rate:.4f}")
        X_test_drifted = add_bias_drift(X_test, drift_rate=drift_rate, seed=seed)
        
        for model_name, model in models.items():
            metrics = evaluate_model_on_data(model, X_test_drifted, y_test)
            
            key = f"{model_name}_drift_{drift_rate:.4f}"
            results[key] = {
                "condition": "drift",
                "drift_rate": drift_rate,
                "model": model_name,
                "mean_rmse": metrics["mean_rmse"],
                "metrics": metrics,
            }
            print(f"  {model_name}: RMSE = {metrics['mean_rmse']:.6f}")

    return results


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_error_vs_noise(
    all_results: Dict[str, Dict],
    noise_sigmas: List[float],
    models: List[str],
    output_path: str,
) -> None:
    """Plot error vs noise sigma for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name in models:
        rmse_values = []
        sigmas_used = []
        
        for sigma in noise_sigmas:
            key = f"{model_name}_noise_{sigma:.4f}"
            if key in all_results:
                rmse_values.append(all_results[key]["mean_rmse"])
                sigmas_used.append(sigma)
        
        if rmse_values:
            ax.plot(sigmas_used, rmse_values, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel("Noise Sigma", fontsize=12)
    ax.set_ylabel("Mean RMSE", fontsize=12)
    ax.set_title("Error vs Noise Level", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_error_vs_dropout(
    all_results: Dict[str, Dict],
    dropout_ps: List[float],
    models: List[str],
    output_path: str,
) -> None:
    """Plot error vs dropout probability for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name in models:
        rmse_values = []
        ps_used = []
        
        for p in dropout_ps:
            key = f"{model_name}_dropout_{p:.4f}"
            if key in all_results:
                rmse_values.append(all_results[key]["mean_rmse"])
                ps_used.append(p)
        
        if rmse_values:
            ax.plot(ps_used, rmse_values, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel("Dropout Probability (p)", fontsize=12)
    ax.set_ylabel("Mean RMSE", fontsize=12)
    ax.set_title("Error vs Dropout Level", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_filtered_trace(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    example_idx: int = 0,
    channel: int = 0,
    output_path: str = "results/plots/filtered_traces.png",
) -> None:
    """Plot filtered trace for one example sequence per model."""
    # Get one example sequence
    seq = X_test[example_idx, :, :]  # (T, 6)
    true_next = y_test[example_idx, channel]
    
    # Reconstruct full sequence including next step for visualization
    full_seq = np.concatenate([seq, y_test[example_idx:example_idx+1, :]], axis=0)
    time_steps = np.arange(full_seq.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_names = ["persistence", "ekf", "pf", "rnn"]
    
    for idx, model_name in enumerate(model_names):
        if model_name not in models:
            continue
            
        ax = axes[idx]
        model = models[model_name]
        
        # Get predictions for the sequence
        seq_input = seq.reshape(1, seq.shape[0], seq.shape[1])  # (1, T, 6)
        pred_next = model.predict(seq_input)[0, channel]
        
        # Plot true sequence
        ax.plot(time_steps[:-1], full_seq[:-1, channel], 'b-', label='True', linewidth=2, alpha=0.7)
        ax.plot(time_steps[-1], true_next, 'b*', markersize=12, label='True (next)', alpha=0.7)
        
        # Plot prediction
        ax.plot(time_steps[-1], pred_next, 'r^', markersize=12, label='Predicted', alpha=0.7)
        
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel(f"Channel {channel}", fontsize=10)
        ax.set_title(f"{model_name.upper()}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Filtered Traces - Example {example_idx}, Channel {channel}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


# ---------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------

def save_metrics_json(
    all_results: Dict[str, Dict],
    output_path: str,
) -> None:
    """Save all results to a single JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean_results = _to_python(all_results)
    
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nSaved metrics to: {output_path}")


# ---------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------

def run_experiment(config_path: str = CONFIG_PATH) -> Dict[str, Dict]:
    """Run comprehensive experiments across all models and conditions."""
    cfg = load_config(config_path)

    # Set seed
    seed = cfg.get("models", {}).get("run", {}).get("seed", 42)
    set_global_seed(seed)

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(config_path)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Build models
    print("\nBuilding models...")
    models = build_models(cfg)

    # Get degradation parameters from config
    degrade_cfg = cfg.get("degrade", {})
    noise_sigmas = degrade_cfg.get("noise_sigmas", [0.01, 0.1, 0.5])
    drift_rates = degrade_cfg.get("drift_rates", [0.05, 0.5])
    dropout_cfg = degrade_cfg.get("dropout", {})
    
    # Handle dropout probabilities: if single value, create a range; if list, use it
    dropout_p_config = dropout_cfg.get("p", 0.2)
    if isinstance(dropout_p_config, (int, float)):
        # Create a range around the configured value
        dropout_ps = [0.1, dropout_p_config, 0.3]
    else:
        # Assume it's a list
        dropout_ps = dropout_p_config if isinstance(dropout_p_config, list) else [0.1, 0.2, 0.3]
    
    dropout_max_gap = dropout_cfg.get("max_gap", 5)
    dropout_mode = dropout_cfg.get("mode", "hold")

    # Run experiments
    all_results = {}

    # 1. Nominal conditions
    nominal_results = run_nominal_experiments(models, X_train, y_train, X_test, y_test)
    all_results.update(nominal_results)

    # 2. Noise experiments
    noise_results = run_noise_experiments(models, X_test, y_test, noise_sigmas, seed=seed)
    all_results.update(noise_results)

    # 3. Dropout experiments
    dropout_results = run_dropout_experiments(
        models, X_test, y_test, dropout_ps, 
        max_gap=dropout_max_gap, mode=dropout_mode, seed=seed
    )
    all_results.update(dropout_results)

    # 4. Drift experiments
    drift_results = run_drift_experiments(models, X_test, y_test, drift_rates, seed=seed)
    all_results.update(drift_results)

    # Save results
    metrics_path = cfg.get("models", {}).get("run", {}).get(
        "metrics_dir", "results/metrics"
    )
    metrics_path = Path(metrics_path) / "metrics.json"
    save_metrics_json(all_results, metrics_path)

    # Create plots
    plots_dir = Path(cfg.get("paths", {}).get("plot_output_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["persistence", "ekf", "pf", "rnn"]
    
    # Plot error vs noise
    plot_error_vs_noise(
        all_results, noise_sigmas, model_names,
        str(plots_dir / "error_vs_noise.png")
    )

    # Plot error vs dropout
    plot_error_vs_dropout(
        all_results, dropout_ps, model_names,
        str(plots_dir / "error_vs_dropout.png")
    )

    # Plot filtered traces
    plot_filtered_trace(
        models, X_test, y_test,
        example_idx=0, channel=0,
        output_path=str(plots_dir / "filtered_traces.png")
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_experiment()
