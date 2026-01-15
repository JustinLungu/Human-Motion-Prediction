"""PF Diagnostics Runner

Simple script to run PF and get ESS statistics and diagnostic plots.

ESS (Effective Sample Size) indicates particle degeneracy:
- ESS close to num_particles: Good diversity, no resampling needed
- ESS << num_particles: Degeneracy, frequent resampling
- ESS consistently below threshold: Filter may be struggling
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src` is on sys.path
# File is at src/models/experiments/run_pf_diagnostics.py
# parents[0] = src/models/experiments/
# parents[1] = src/models/
# parents[2] = src/
# parents[3] = project root
SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import yaml
from dataloader.dataset import UCIHARDatasetLoader
from models.pf import PFBaseline
from utils.progress import ProgressTracker


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
    """Run PF diagnostics."""
    print("=" * 70)
    print("PF DIAGNOSTICS RUNNER")
    print("=" * 70)
    print()
    
    # Load config
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    
    # Get PF parameters from config
    pf_cfg = cfg.get("models", {}).get("pf", {})
    num_particles = 1000
    Q_scale = float(pf_cfg.get("Q_scale", 1.0))
    R_scale = pf_cfg.get("R_scale")
    if R_scale is not None:
        R_scale = float(R_scale)
    dt = float(pf_cfg.get("dt", 0.02))
    resample_threshold = float(pf_cfg.get("resample_threshold", 0.5))
    seed = cfg.get("models", {}).get("run", {}).get("seed", 42)
    
    print(f"PF Parameters:")
    print(f"  num_particles: {num_particles}")
    print(f"  Q_scale: {Q_scale:.6f}")
    print(f"  R_scale: {R_scale if R_scale is not None else 'auto (from data)'}")
    print(f"  dt: {dt:.6f}")
    print(f"  resample_threshold: {resample_threshold:.2f}")
    print(f"  seed: {seed}")
    print()
    
    # Load data (use config path as string for loader)
    print("Loading data...")
    loader = UCIHARDatasetLoader(str(SRC_ROOT.parent / "configs/config.yaml"))
    train_split = loader.load_split("train")
    test_split = loader.load_split("test")
    X_train_full = train_split.X  # (N, T, 6) where T=128 (full window)
    X_test_full = test_split.X  # (N, T, 6) - all test sequences
    
    print(f"Training data shape: {X_train_full.shape}")
    print(f"Test data shape: {X_test_full.shape}")
    num_test_seqs = X_test_full.shape[0]
    print(f"Using all {num_test_seqs} test sequences for diagnostics")
    print()
    
    # Build and fit PF (PF expects full sequences)
    print("Fitting PF...")
    pf = PFBaseline(
        num_particles=num_particles,
        Q_scale=Q_scale,
        R_scale=R_scale,
        dt=dt,
        resample_threshold=resample_threshold,
        seed=seed,
    )
    # PF.fit() expects (N, T, 6) for X, y is not used but required by interface
    pf.fit(X_train_full, X_train_full[:, -1, :])
    print("✓ PF fitted")
    print()
    
    # Initialize progress tracker
    progress = ProgressTracker(num_test_seqs)
    progress.update(0)
    
    # Run prediction with diagnostics on all test sequences
    print(f"Running filter with diagnostics (all {num_test_seqs} test sequences)...")
    print("-" * 70)
    
    # Process sequences one by one to track progress
    # Initialize diagnostic storage
    all_ess_accumulated = []
    predictions = np.zeros((num_test_seqs, 6), dtype=np.float32)
    
    for n in range(num_test_seqs):
        seq = X_test_full[n:n+1, :, :]  # (1, T, 6)
        pred = pf.predict(seq, log_diagnostics=True)
        predictions[n, :] = pred[0, :]
        
        # Aggregate ESS from this sequence (predict() resets _all_ess, so preserve it manually)
        if hasattr(pf, '_all_ess') and len(pf._all_ess) > 0:
            all_ess_accumulated.extend(pf._all_ess)
        
        # Update progress every 50 sequences or at the end for efficiency
        if (n + 1) % 50 == 0 or (n + 1) == num_test_seqs:
            increment = 50 if (n + 1) % 50 == 0 else ((n + 1) % 50)
            progress.update(increment, f"Processed {n + 1}/{num_test_seqs} sequences")
    
    # Restore accumulated ESS for statistics
    pf._all_ess = all_ess_accumulated
    
    progress.finish()
    print()
    
    # Get ESS statistics
    print("ESS STATISTICS")
    print("-" * 70)
    stats = pf.get_ess_statistics()
    
    if stats:
        print(f"ESS Mean: {stats.get('mean_ess', 'N/A'):.2f}")
        print(f"ESS Min: {stats.get('min_ess', 'N/A'):.2f}")
        print(f"ESS Max: {stats.get('max_ess', 'N/A'):.2f}")
        print(f"ESS Std: {stats.get('std_ess', 'N/A'):.2f}")
        print(f"ESS % Below Threshold: {stats.get('ess_pct_below_threshold', 'N/A'):.2f}%")
        print(f"Number of Resamples: {stats.get('num_resamples', 'N/A')}")
        print()
        
        threshold = resample_threshold * num_particles
        print(f"Resample Threshold: {threshold:.1f} ({resample_threshold:.0%} of {num_particles} particles)")
        print()
        
        # Health assessment
        mean_ess = stats.get('mean_ess', np.nan)
        if not np.isnan(mean_ess):
            if mean_ess > 0.8 * num_particles:
                print("✓ ESS is high - good particle diversity, minimal resampling needed")
            elif mean_ess > 0.5 * num_particles:
                print("⚠ ESS is moderate - some resampling occurring")
            else:
                print("⚠ ESS is low - frequent resampling, possible degeneracy")
    else:
        print("No ESS data available")
    
    print()
    
    # Generate and save diagnostic plot
    print("Generating ESS curve plot...")
    project_root = SRC_ROOT.parent
    
    # Get plot directory from config, default to results/plots
    plot_output_dir = cfg.get("paths", {}).get("plot_output_dir", "results/plots")
    plots_dir = project_root / plot_output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / "pf_diagnostics.png"
    fig = pf.plot_ess_curves(save_path=str(plot_path))
    
    if fig is not None:
        print(f"✓ ESS curve plot saved to: {plot_path}")
        print()
        print("Plot shows:")
        print("  - ESS over time for each sequence")
        print("  - Resample threshold line (red dashed)")
        print("  - Maximum ESS line (green dotted)")
    else:
        print("⚠ Failed to generate plot")
    
    print()
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

