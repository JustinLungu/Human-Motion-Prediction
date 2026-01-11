"""EKF Diagnostics Runner

Simple script to run EKF and get innovation statistics and diagnostic plots.

Checks for:
- Mahalanobis distance consistently >5: Filter diverging or Q/R badly tuned
- Trace(P) monotonically increasing: Process noise too high
- Trace(P) → 0: Filter overconfident, ignoring measurements
- Negative eigenvalues in P: Numerical instability
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src` is on sys.path
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import yaml
from dataloader.dataset import UCIHARDatasetLoader
from models.ekf import EKFBaseline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    # Resolve path relative to project root (two levels up from src/models/)
    project_root = SRC_ROOT.parent
    config_path = project_root / config_path
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def split_xy(X: np.ndarray):
    """Split sequence into inputs (N, T-1, C) and targets (N, C)."""
    return (
        X[:, :-1, :].astype(np.float32),
        X[:, -1, :].astype(np.float32),
    )


def main():
    """Run EKF diagnostics."""
    print("=" * 70)
    print("EKF DIAGNOSTICS RUNNER")
    print("=" * 70)
    print()
    
    # Load config
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    
    # Get EKF parameters from config
    ekf_cfg = cfg.get("models", {}).get("ekf", {})
    Q_scale = float(ekf_cfg.get("Q_scale", 1e-4))
    R_scale = ekf_cfg.get("R_scale")
    if R_scale is not None:
        R_scale = float(R_scale)
    dt = float(ekf_cfg.get("dt", 0.02))
    
    print(f"EKF Parameters:")
    print(f"  Q_scale: {Q_scale:.6f}")
    print(f"  R_scale: {R_scale if R_scale is not None else 'auto (from data)'}")
    print(f"  dt: {dt:.6f}")
    print()
    
    # Load data (use config path as string for loader)
    print("Loading data...")
    loader = UCIHARDatasetLoader(str(SRC_ROOT.parent / "configs/config.yaml"))
    train_split = loader.load_split("train")
    X_train_full = train_split.X  # (N, T, 6) where T=128 (full window)
    
    # Split sequences for fitting (same format as run_models.py)
    X_train_in, y_train = split_xy(X_train_full)  # X_train_in: (N, T-1, 6), y_train: (N, 6)
    
    # For diagnostics, use the split format (matches actual usage)
    # The filter will process T-1 timesteps, then predict the T-th timestep
    X_diag = X_train_in[0:1, :, :]  # (1, T-1, 6) - first sequence for diagnostics
    
    print(f"Training data shape: {X_train_full.shape}")
    print(f"Input sequences for fit/predict: {X_train_in.shape}")
    print(f"Sequence for diagnostics: {X_diag.shape}")
    print()
    
    # Build and fit EKF (same as run_models.py)
    print("Fitting EKF...")
    ekf = EKFBaseline(Q_scale=Q_scale, R_scale=R_scale, dt=dt)
    ekf.fit(X_train_in, y_train)  # fit() accepts (N, T, 6) or (N, T-1, 6), uses X for R estimation
    print("✓ EKF fitted")
    print()
    
    # Run prediction with diagnostics on first sequence
    # predict() will filter through T-1 timesteps, then predict the next step
    print("Running filter with diagnostics (first sequence)...")
    print("-" * 70)
    
    # Run on sequence with diagnostics enabled
    predictions = ekf.predict(X_diag, log_diagnostics=True)
    
    print("✓ Filter run complete")
    print()
    
    # Get innovation statistics
    print("INNOVATION STATISTICS")
    print("-" * 70)
    stats = ekf.get_innovation_statistics()
    
    if stats:
        print(f"Innovation Mean (per channel): {[f'{x:.6f}' for x in stats.get('innovation_mean', [])]}")
        print(f"Innovation Std (per channel):  {[f'{x:.6f}' for x in stats.get('innovation_std', [])]}")
        print(f"Innovation Max Norm: {stats.get('innovation_max_norm', 'N/A'):.6f}")
        print(f"Innovation Mean Norm: {stats.get('innovation_mean_norm', 'N/A'):.6f}")
        
        if 'mahalanobis_mean' in stats:
            print(f"Mahalanobis Distance - Mean: {stats['mahalanobis_mean']:.4f}")
            print(f"Mahalanobis Distance - Max: {stats['mahalanobis_max']:.4f}")
            print(f"Mahalanobis Distance - % above 5: {stats.get('mahalanobis_pct_above_5', 0):.2f}%")
        
        if 'P_trace_initial' in stats and stats['P_trace_initial'] is not None:
            print(f"Trace(P) - Initial: {stats['P_trace_initial']:.6f}")
            print(f"Trace(P) - Final: {stats['P_trace_final']:.6f}")
            if stats.get('P_trace_change') is not None:
                change_pct = (stats['P_trace_change'] / stats['P_trace_initial']) * 100
                print(f"Trace(P) - Change: {stats['P_trace_change']:+.6f} ({change_pct:+.2f}%)")
                print(f"Trace(P) - Monotonic increasing: {stats.get('P_trace_is_monotonic_increasing', False)}")
            
            if stats.get('P_trace_trending_to_zero'):
                print("⚠ Trace(P) → 0 detected!")
        
        if 'P_min_eigenvalue_min' in stats:
            print(f"P Min Eigenvalue - Mean: {stats['P_min_eigenvalue_mean']:.6f}")
            print(f"P Min Eigenvalue - Min: {stats['P_min_eigenvalue_min']:.6f}")
            if stats.get('P_has_negative_eigenvalues'):
                print("⚠ Negative eigenvalues detected in P!")
    else:
        print("No diagnostics available")
    
    print()
    
    # Check filter health
    print("FILTER HEALTH CHECKS")
    print("-" * 70)
    health = ekf.check_filter_health()
    
    if health["status"] == "no_diagnostics":
        print("No diagnostics available. Run predict() with log_diagnostics=True first.")
    elif health["status"] == "healthy":
        print("✓ Filter appears healthy - no warnings detected")
    else:
        print(f"⚠ {len(health['warnings'])} warning(s) detected:")
        for i, warning in enumerate(health["warnings"], 1):
            print(f"  {i}. {warning}")
    
    print()
    
    # Generate and save diagnostic plot
    print("Generating diagnostic plot...")
    project_root = SRC_ROOT.parent
    plots_dir = project_root / "results/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "ekf_diagnostics.png"
    fig = ekf.plot_diagnostics(save_path=str(plot_path))
    
    if fig is not None:
        print(f"✓ Diagnostic plot saved to: {plot_path}")
        print()
        print("Plot shows:")
        print("  - Row 1: Innovation magnitude, Mahalanobis distance, Trace(P)")
        print("  - Row 2: P min eigenvalues, Per-channel innovations, Summary stats")
    else:
        print("⚠ Failed to generate plot")
    
    print()
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

