"""
Comprehensive experiments runner for next-step IMU prediction models.

Models tested: Persistence, EKF, PF, RNN
Conditions tested: Nominal, Noise, Dropout, Drift
Outputs: results/metrics/metrics.json

Note: Plotting functionality has been moved to degradation_plotter.py
"""

from __future__ import annotations

import json
import os, sys
import random
from pathlib import Path
from typing import Dict, List, Callable, Optional

import numpy as np
import yaml
from scipy.stats import chi2

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.evaluate import evaluate_model
from models.ekf import EKFBaseline
from models.pf import PFBaseline
from models.degrade import add_gaussian_noise, add_bias_drift, apply_dropout
from models.experiments.factory import build_models, load_data
from utils.progress import ProgressTracker

CONFIG_PATH = "configs/config.yaml"

def load_config(path: str = CONFIG_PATH) -> Dict:
    """Load YAML configuration file."""
    if not os.path.isabs(path):
        project_root = Path(__file__).resolve().parents[2]
        path = str(project_root / path)
    if not os.path.exists(path):
        print(f"Warning: Config file not found at {path}, using defaults")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def set_global_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    np.random.seed(int(seed))
    random.seed(int(seed))



def sanity_check(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split: str) -> None:
    """Basic sanity checks."""
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise RuntimeError(f"{model_name} ({split}): invalid prediction shape {y_pred.shape}")
    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        raise RuntimeError(f"{model_name} ({split}): NaNs detected")

def compute_residual_autocorrelation(y_true: np.ndarray, y_pred: np.ndarray, lag: int = 1) -> float:
    """Compute residual autocorrelation at specified lag (smoothness metric)."""
    residuals = y_true - y_pred
    autocorrs = []
    for ch in range(6):
        res_ch = residuals[:, ch]
        if len(res_ch) > lag and np.var(res_ch) > 1e-10:
            centered = res_ch - np.mean(res_ch)
            autocorr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
            if not np.isnan(autocorr):
                autocorrs.append(autocorr)
    return float(np.mean(autocorrs)) if len(autocorrs) > 0 else 0.0

def compute_nis_ekf(ekf: EKFBaseline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute NIS (Normalized Innovation Squared) for EKF."""
    _ = ekf.predict(X_test, log_diagnostics=True)
    mahal_dists_list = getattr(ekf, '_all_mahalanobis', getattr(ekf, '_mahalanobis_distances', []))
    if len(mahal_dists_list) == 0:
        return {"nis_mean": np.nan, "nis_std": np.nan, "nis_pct_below_chi2_95": np.nan}
    mahal_dists = np.array([d for d in mahal_dists_list if not (np.isnan(d) or np.isinf(d))])
    nis_values = mahal_dists ** 2
    if len(nis_values) == 0:
        return {"nis_mean": np.nan, "nis_std": np.nan, "nis_pct_below_chi2_95": np.nan}
    chi2_95 = chi2.ppf(0.95, df=6)
    return {
        "nis_mean": float(np.mean(nis_values)),
        "nis_std": float(np.std(nis_values)),
        "nis_pct_below_chi2_95": float(np.mean(nis_values < chi2_95) * 100),
    }

def compute_nis_pf(pf: PFBaseline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute NIS (Normalized Innovation Squared) for PF."""
    preds = pf.predict(X_test)
    residuals = y_test - preds
    R_diag = np.maximum(np.diag(pf.R), 1e-8)
    nis_values = [np.sum(res ** 2 / R_diag) for res in residuals 
                  if not (np.isnan(np.sum(res ** 2 / R_diag)) or np.isinf(np.sum(res ** 2 / R_diag)))]
    if len(nis_values) == 0:
        return {"nis_mean": np.nan, "nis_std": np.nan, "nis_pct_below_chi2_95": np.nan}
    nis_array = np.array(nis_values)
    chi2_95 = chi2.ppf(0.95, df=6)
    return {
        "nis_mean": float(np.mean(nis_array)),
        "nis_std": float(np.std(nis_array)),
        "nis_pct_below_chi2_95": float(np.mean(nis_array < chi2_95) * 100),
    }

def evaluate_model_on_data(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate a model on test data."""
    preds = model.predict(X_test)
    sanity_check(y_test, preds, model.name, "test")
    metrics = evaluate_model(y_test, preds)
    metrics["residual_autocorr"] = compute_residual_autocorrelation(y_test, preds)
    if isinstance(model, EKFBaseline):
        X_test_full = np.concatenate([X_test, y_test[:, None, :]], axis=1)
        metrics.update(compute_nis_ekf(model, X_test_full, y_test))
    elif isinstance(model, PFBaseline):
        X_test_full = np.concatenate([X_test, y_test[:, None, :]], axis=1)
        metrics.update(compute_nis_pf(model, X_test_full, y_test))
    return metrics

def _run_degradation_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    condition: str,
    param_name: str,
    param_values: List[float],
    degrade_func: Callable,
    degrade_kwargs: Dict,
    progress: Optional[ProgressTracker] = None,
) -> Dict[str, Dict]:
    """Generic function to run degradation experiments."""
    print(f"\n{'=' * 60}\n{condition.upper()} EXPERIMENTS\n{'=' * 60}")
    results = {}
    for param_val in param_values:
        print(f"\n{param_name} = {param_val:.4f}")
        X_test_degraded = degrade_func(X_test, **{param_name: param_val}, **degrade_kwargs)
        for model_name, model in models.items():
            metrics = evaluate_model_on_data(model, X_test_degraded, y_test)
            key = f"{model_name}_{condition}_{param_val:.4f}"
            results[key] = {
                "condition": condition,
                param_name: param_val,
                "model": model_name,
                "mean_rmse": metrics["mean_rmse"],
                "metrics": metrics,
            }
            print(f"  {model_name}: RMSE = {metrics['mean_rmse']:.6f}")
            if progress:
                progress.update(1, f"Completed: {model_name} ({condition} {param_name}={param_val:.4f})")
    return results

def run_nominal_experiments(
    models: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    progress: Optional[ProgressTracker] = None,
) -> Dict[str, Dict]:
    """Run experiments on nominal (clean) data."""
    print(f"\n{'=' * 60}\nNOMINAL CONDITIONS\n{'=' * 60}")
    results = {}
    for model_name, model in models.items():
        print(f"\nFitting {model_name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model_on_data(model, X_test, y_test)
        results[model_name] = {"condition": "nominal", "mean_rmse": metrics["mean_rmse"], "metrics": metrics}
        print(f"  {model_name}: RMSE = {metrics['mean_rmse']:.6f}")
        if progress:
            progress.update(1, f"Completed: {model_name} (nominal)")
    return results

def run_noise_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_sigmas: List[float],
    seed: int = 42,
    progress: Optional[ProgressTracker] = None,
) -> Dict[str, Dict]:
    """Run experiments with Gaussian noise."""
    return _run_degradation_experiments(
        models, X_test, y_test, "noise", "sigma", noise_sigmas,
        add_gaussian_noise, {"seed": seed}, progress
    )

def run_dropout_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    dropout_ps: List[float],
    max_gap: int = 5,
    mode: str = "hold",
    seed: int = 42,
    progress: Optional[ProgressTracker] = None,
) -> Dict[str, Dict]:
    """Run experiments with dropout."""
    return _run_degradation_experiments(
        models, X_test, y_test, "dropout", "p", dropout_ps,
        apply_dropout, {"max_gap": max_gap, "mode": mode, "seed": seed}, progress
    )

def run_drift_experiments(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    drift_rates: List[float],
    seed: int = 42,
    progress: Optional[ProgressTracker] = None,
) -> Dict[str, Dict]:
    """Run experiments with bias drift."""
    return _run_degradation_experiments(
        models, X_test, y_test, "drift", "drift_rate", drift_rates,
        add_bias_drift, {"seed": seed}, progress
    )

def save_metrics_json(all_results: Dict[str, Dict], output_path: str) -> None:
    """Save all results to a single JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(output_path, "w") as f:
        json.dump(_to_python(all_results), f, indent=2)
    print(f"\nSaved metrics to: {output_path}")

def run_experiment(config_path: str = CONFIG_PATH) -> Dict[str, Dict]:
    """Run comprehensive experiments across all models and conditions."""
    cfg = load_config(config_path)
    seed = cfg.get("models", {}).get("run", {}).get("seed", 42)
    set_global_seed(seed)
    
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(config_path)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\nBuilding models...")
    models = build_models(cfg)
    
    degrade_cfg = cfg.get("degrade", {})
    noise_sigmas = degrade_cfg.get("noise_sigmas", [0.01, 0.1, 0.5])
    drift_rates = degrade_cfg.get("drift_rates", [0.05, 0.5])
    dropout_cfg = degrade_cfg.get("dropout", {})
    dropout_p_config = dropout_cfg.get("p", 0.2)
    dropout_ps = ([0.1, dropout_p_config, 0.3] if isinstance(dropout_p_config, (int, float))
                  else (dropout_p_config if isinstance(dropout_p_config, list) else [0.1, 0.2, 0.3]))
    dropout_max_gap = dropout_cfg.get("max_gap", 5)
    dropout_mode = dropout_cfg.get("mode", "hold")
    
    num_models = len(models)
    total_experiments = (num_models + num_models * len(noise_sigmas) + 
                         num_models * len(dropout_ps) + num_models * len(drift_rates))
    progress = ProgressTracker(total_experiments)
    progress.update(0)
    
    all_results = {}
    all_results.update(run_nominal_experiments(models, X_train, y_train, X_test, y_test, progress=progress))
    all_results.update(run_noise_experiments(models, X_test, y_test, noise_sigmas, seed=seed, progress=progress))
    all_results.update(run_dropout_experiments(models, X_test, y_test, dropout_ps, 
                                                max_gap=dropout_max_gap, mode=dropout_mode, seed=seed, progress=progress))
    all_results.update(run_drift_experiments(models, X_test, y_test, drift_rates, seed=seed, progress=progress))
    progress.finish()
    
    metrics_path = Path(cfg.get("models", {}).get("run", {}).get("metrics_dir", "results/metrics")) / "metrics.json"
    save_metrics_json(all_results, metrics_path)
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    return all_results

if __name__ == "__main__":
    run_experiment()
