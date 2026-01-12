"""Grid search tuner for Particle Filter.

Simple tuning script that evaluates all configurations without ESS checks.
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
from utils.progress import ProgressTracker


def tune_pf_extended() -> None:
    """Extended grid search with better hyperparameter ranges."""
    loader = UCIHARDatasetLoader("configs/config.yaml")
    
    train_split = loader.load_split("train")
    X_train = train_split.X.astype(np.float32)  # (N, T, 6) - full sequences
    
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)  # (N, T, 6) - full sequences
    y_test_next = X_test[:, -1, :]  # For evaluation only
    
    # Reduced grid for quick testing
    Q_scales = [1.0]  # Just 2 values
    R_scales = [2.0]  # Just 2 values
    num_particles_list = [500]  # Just 1 value
    sample_percentages = [1.0]  # Just 1 value
    dt = 0.02  # Just 1 value
    
    total_configs = len(num_particles_list) * len(Q_scales) * len(R_scales) * len(sample_percentages)
    
    print("=" * 110)
    print("PF Tuning")
    print("=" * 110)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Total configurations: {total_configs}")
    print()
    
    # Initialize progress tracker
    progress = ProgressTracker(total_configs)
    progress.update(0)
    
    results = []
    
    print(f"{'N':<6} {'dt':<6} {'Q':<10} {'R':<8} {'resample_percentage':<10} {'Train':<10} {'Test':<10}")
    print("-" * 70)
    
    best_test_rmse = float('inf')
    best_params = None
    
    for num_particles in num_particles_list:
        for resample_percentage in sample_percentages:
            for Q_scale in Q_scales:
                for R_scale in R_scales:
                    # Build and fit
                    pf = PFBaseline(
                        num_particles=num_particles,
                        Q_scale=Q_scale,
                        R_scale=R_scale,
                        dt=dt,
                        resample_threshold=resample_percentage,  # Resample when ESS < 50% of N
                        seed=42  # For reproducibility
                    )
                    
                    try:
                        # PF.fit() expects full sequences (N, T, 6), y is not used but required by interface
                        pf.fit(X_train, X_train[:, -1, :])
                    except Exception as e:
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {str(R_scale):<8} {resample_percentage:<10.2f} "
                              f"{'FIT FAIL':<10} {'---':<10}")
                        progress.update(1, f"Failed: N={num_particles}, dt={dt}, Q={Q_scale:.0e}, R={R_scale:.1f}, resample_percentage={resample_percentage:.2f}")
                        continue
                    
                    # Evaluate performance (no ESS checks, just run everything)
                    try:
                        # PF.predict() expects full sequences (N, T, 6) and predicts next step
                        preds_train = pf.predict(X_train)
                        train_rmse = evaluate_model(X_train[:, -1, :], preds_train)["mean_rmse"]
                        
                        preds_test = pf.predict(X_test)
                        test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]
                        
                        r_str = f"{R_scale:.1f}"
                        
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {r_str:<8} {resample_percentage:<10.2f} "
                              f"{train_rmse:<10.6f} {test_rmse:<10.6f}")
                        
                        # Track results
                        results.append({
                            "num_particles": num_particles,
                            "dt": dt,
                            "Q_scale": Q_scale,
                            "R_scale": R_scale,
                            "resample_percentage": resample_percentage,
                            "train_rmse": train_rmse,
                            "test_rmse": test_rmse,
                        })
                        
                        # Update best
                        if test_rmse < best_test_rmse:
                            best_test_rmse = test_rmse
                            best_params = (num_particles, dt, Q_scale, R_scale, resample_percentage)
                        
                        # Update progress
                        progress.update(1, f"Completed: N={num_particles}, dt={dt}, Q={Q_scale:.0e}, R={R_scale:.1f}, resample_percentage={resample_percentage:.2f}, RMSE={test_rmse:.6f}")
                            
                    except Exception as e:
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {str(R_scale):<8} {resample_percentage:<10.2f} "
                              f"{'PRED FAIL':<10} {'---':<10}")
                        progress.update(1, f"Failed: N={num_particles}, dt={dt}, Q={Q_scale:.0e}, R={R_scale:.1f}, resample_percentage={resample_percentage:.2f}")
                        continue
    
    # Finalize progress
    progress.finish()
    
    # Summary
    print("=" * 110)
    print("TUNING SUMMARY")
    print("=" * 110)
    print()
    
    if best_params:
        N_best, dt_best, Q_best, R_best, resample_percentage_best = best_params
        print(f"Best Configuration:")
        print(f"  num_particles: {N_best}")
        print(f"  dt: {dt_best:.3f}")
        print(f"  Q_scale: {Q_best:.0e}")
        print(f"  R_scale: {R_best:.1f}")
        print(f"  resample_percentage: {resample_percentage_best:.2f}")
        print(f"  Test RMSE: {best_test_rmse:.6f}")
        print()
    else:
        print("⚠ No valid configuration found!")
        print()
    
    # Persistence baseline
    persistence_preds = X_test[:, -1, :]  # Last timestep of each sequence
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]
    print(f"Persistence Baseline Test RMSE: {persistence_rmse:.6f}")
    print()
    
    if best_params and persistence_rmse > 0:
        if best_test_rmse < persistence_rmse:
            improvement = ((persistence_rmse - best_test_rmse) / persistence_rmse) * 100
            print(f"✓ PF beats persistence by {improvement:.2f}%")
        else:
            degradation = ((best_test_rmse - persistence_rmse) / persistence_rmse) * 100
            print(f"✗ PF underperforms persistence by {degradation:.2f}%")
    elif best_params and persistence_rmse == 0:
        print(f"⚠ Persistence RMSE is 0 (perfect match), cannot compute relative improvement")
    print()
    
    # Top 5 configurations
    if results:
        results.sort(key=lambda x: x["test_rmse"])
        
        print("Top 5 Configurations:")
        print(f"{'Rank':<6} {'N':<6} {'dt':<6} {'Q':<10} {'R':<8} {'resample_percentage':<10} {'Test RMSE':<12}")
        print("-" * 50)
        for i, r in enumerate(results[:5], 1):
            r_str = f"{r['R_scale']:.1f}"
            print(f"{i:<6} {r['num_particles']:<6d} {r['dt']:<6.2f} {r['Q_scale']:<10.0e} {r_str:<8} {r['resample_percentage']:<10.2f} "
                  f"{r['test_rmse']:<12.6f}")
    
    print()
    print("=" * 110)
    
    # Analysis and recommendations
    print()
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 110)
    
    if results:
        # Analyze Q_scale trend
        q_rmses = {}
        for r in results:
            q = r["Q_scale"]
            if q not in q_rmses:
                q_rmses[q] = []
            q_rmses[q].append(r["test_rmse"])
        
        print("Average Test RMSE by Q_scale:")
        for q in sorted(q_rmses.keys()):
            avg_rmse = np.mean(q_rmses[q])
            print(f"  Q={q:.0e}: {avg_rmse:.6f} ({len(q_rmses[q])} configs)")
        print()
        
        # Analyze R_scale trend
        r_rmses = {}
        for r in results:
            r_scale = r["R_scale"]
            if r_scale not in r_rmses:
                r_rmses[r_scale] = []
            r_rmses[r_scale].append(r["test_rmse"])
        
        print("Average Test RMSE by R_scale:")
        for r_scale in sorted(r_rmses.keys()):
            avg_rmse = np.mean(r_rmses[r_scale])
            print(f"  R={r_scale:.1f}: {avg_rmse:.6f} ({len(r_rmses[r_scale])} configs)")
        print()
    
    print("=" * 110)


if __name__ == "__main__":
    tune_pf_extended()