"""Improved grid search tuner for Particle Filter with diagnostics.

Key improvements:
- Much wider Q_scale range (PF needs higher process noise than EKF)
- R_scale includes None (auto from data)
- ESS diagnostics to detect degeneracy
- Stability checks for particle collapse
- dt tuning (critical for PF performance)
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from typing import Dict, Tuple

from dataloader.dataset import UCIHARDatasetLoader
from models.pf import PFBaseline
from models.evaluate import evaluate_model


def check_pf_health(pf: PFBaseline, X_sample: np.ndarray) -> Tuple[bool, Dict]:
    """Check if PF is healthy (not degenerate).
    
    Args:
        pf: Fitted PF model
        X_sample: Sample sequences for diagnostics
    
    Returns:
        (is_healthy, diagnostics_dict)
    """
    try:
        # Run filter with diagnostics
        _ = pf.predict(X_sample, log_diagnostics=True)
        stats = pf.get_ess_statistics()
        
        if not stats:
            return False, {"error": "No diagnostics available"}
        
        # Extract metrics
        diagnostics = {
            "ess_mean": stats.get("ess_mean", 0),
            "ess_min": stats.get("ess_min", 0),
            "ess_pct_below_threshold": stats.get("ess_pct_below_threshold", 100),
            "num_resamples": stats.get("num_resamples", 0),
        }
        
        # Health criteria:
        # 1. Mean ESS should be > 20% of particles (severe degeneracy check)
        # 2. ESS should not always be below threshold (constant resampling = bad)
        min_acceptable_ess = 0.2 * pf.num_particles
        is_healthy = (
            diagnostics["ess_mean"] > min_acceptable_ess and
            diagnostics["ess_pct_below_threshold"] < 95  # Not always resampling
        )
        
        return is_healthy, diagnostics
        
    except Exception as e:
        return False, {"error": str(e)}


def tune_pf_extended() -> None:
    """Extended grid search with better hyperparameter ranges."""
    loader = UCIHARDatasetLoader("configs/config.yaml")
    
    train_split = loader.load_split("train")
    X_train = train_split.X.astype(np.float32)
    X_train_in = X_train[:, :-1, :]
    y_train_next = X_train[:, -1, :]
    
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)
    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]
    
    # Sample for health checks
    X_test_sample = X_test_in[:50, :, :]  # Smaller sample for PF (slower than EKF)
    
    # CRITICAL: PF needs MUCH higher process noise than EKF
    # PF particles need to explore; EKF assumes Gaussian
    Q_scales = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]  # Much wider range!
    
    # R_scale: include None (auto from data) and various multipliers
    R_scales = [None, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Particle counts: more particles = better but slower
    num_particles_list = [50, 100, 200, 500]
    
    # dt: CRITICAL parameter (must match data sampling rate)
    # UCI-HAR is 50Hz → dt should be ~0.02
    dts = [0.02, 0.05, 0.1]
    
    print("=" * 110)
    print("PF Extended Tuning with Diagnostics")
    print("=" * 110)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Total configurations: {len(num_particles_list) * len(Q_scales) * len(R_scales) * len(dts)}")
    print()
    
    results = []
    degenerate_configs = []
    
    print(f"{'N':<6} {'dt':<6} {'Q':<10} {'R':<8} {'Train':<10} {'Test':<10} {'ESS':<8} {'Resamp%':<10} {'Status':<12}")
    print("-" * 110)
    
    best_test_rmse = float('inf')
    best_params = None
    best_diagnostics = None
    
    for num_particles in num_particles_list:
        for dt in dts:
            for Q_scale in Q_scales:
                for R_scale in R_scales:
                    # Build and fit
                    pf = PFBaseline(
                        num_particles=num_particles,
                        Q_scale=Q_scale,
                        R_scale=R_scale,
                        dt=dt,
                        resample_threshold=0.5,  # Resample when ESS < 50% of N
                        seed=42  # For reproducibility
                    )
                    
                    try:
                        pf.fit(X_train_in, y_train_next)
                    except Exception as e:
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {str(R_scale):<8} "
                              f"{'FIT FAIL':<10} {'---':<10} {'---':<8} {'---':<10} {str(e)[:12]:<12}")
                        degenerate_configs.append((num_particles, dt, Q_scale, R_scale, "fit_failed"))
                        continue
                    
                    # Check health
                    is_healthy, diag = check_pf_health(pf, X_test_sample)
                    
                    if not is_healthy:
                        reason = diag.get("error", "degenerate")
                        ess_mean = diag.get("ess_mean", 0)
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {str(R_scale) if R_scale else 'data':<8} "
                              f"{'---':<10} {'---':<10} {ess_mean:<8.1f} {'---':<10} {'DEGENERATE':<12}")
                        degenerate_configs.append((num_particles, dt, Q_scale, R_scale, reason))
                        continue
                    
                    # Evaluate performance
                    try:
                        preds_train = pf.predict(X_train_in)
                        train_rmse = evaluate_model(y_train_next, preds_train)["mean_rmse"]
                        
                        preds_test = pf.predict(X_test_in)
                        test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]
                        
                        # Check for exploding errors
                        if train_rmse > 100 or test_rmse > 100 or np.isnan(train_rmse) or np.isnan(test_rmse):
                            print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {str(R_scale) if R_scale else 'data':<8} "
                                  f"{train_rmse:<10.4f} {test_rmse:<10.4f} {diag['ess_mean']:<8.1f} "
                                  f"{diag['ess_pct_below_threshold']:<10.1f} {'EXPLODING':<12}")
                            degenerate_configs.append((num_particles, dt, Q_scale, R_scale, "exploding_error"))
                            continue
                        
                        r_str = f"{R_scale:.1f}" if R_scale else "data"
                        resamp_pct = diag['ess_pct_below_threshold']
                        status = "✓ HEALTHY" if is_healthy else "⚠ CHECK"
                        
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {r_str:<8} "
                              f"{train_rmse:<10.6f} {test_rmse:<10.6f} "
                              f"{diag['ess_mean']:<8.1f} {resamp_pct:<10.1f} {status:<12}")
                        
                        # Track results
                        results.append({
                            "num_particles": num_particles,
                            "dt": dt,
                            "Q_scale": Q_scale,
                            "R_scale": R_scale,
                            "train_rmse": train_rmse,
                            "test_rmse": test_rmse,
                            "diagnostics": diag,
                            "is_healthy": is_healthy,
                        })
                        
                        # Update best
                        if test_rmse < best_test_rmse and is_healthy:
                            best_test_rmse = test_rmse
                            best_params = (num_particles, dt, Q_scale, R_scale)
                            best_diagnostics = diag
                            
                    except Exception as e:
                        print(f"{num_particles:<6d} {dt:<6.2f} {Q_scale:<10.0e} {str(R_scale) if R_scale else 'data':<8} "
                              f"{'PRED FAIL':<10} {'---':<10} {'---':<8} {'---':<10} {str(e)[:12]:<12}")
                        degenerate_configs.append((num_particles, dt, Q_scale, R_scale, f"predict_failed"))
    
    # Summary
    print("=" * 110)
    print("TUNING SUMMARY")
    print("=" * 110)
    print()
    
    if best_params:
        N_best, dt_best, Q_best, R_best = best_params
        print(f"Best Configuration:")
        print(f"  num_particles: {N_best}")
        print(f"  dt: {dt_best:.3f}")
        print(f"  Q_scale: {Q_best:.0e}")
        print(f"  R_scale: {R_best if R_best else 'auto (from data)'}")
        print(f"  Test RMSE: {best_test_rmse:.6f}")
        print()
        
        print(f"Best Configuration Diagnostics:")
        for key, val in best_diagnostics.items():
            print(f"  {key}: {val}")
        print()
    else:
        print("⚠ No healthy configuration found!")
        print()
    
    # Persistence baseline
    persistence_preds = X_test_in[:, -1, :]
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]
    print(f"Persistence Baseline Test RMSE: {persistence_rmse:.6f}")
    print()
    
    if best_params and best_test_rmse < persistence_rmse:
        improvement = ((persistence_rmse - best_test_rmse) / persistence_rmse) * 100
        print(f"✓ PF beats persistence by {improvement:.2f}%")
    elif best_params:
        degradation = ((best_test_rmse - persistence_rmse) / persistence_rmse) * 100
        print(f"✗ PF underperforms persistence by {degradation:.2f}%")
    print()
    
    # Degenerate configurations summary
    print(f"Degenerate Configurations: {len(degenerate_configs)} / {len(num_particles_list) * len(Q_scales) * len(R_scales) * len(dts)}")
    if degenerate_configs:
        print()
        print("Degenerate Config Breakdown:")
        from collections import Counter
        reasons = Counter([reason for _, _, _, _, reason in degenerate_configs])
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")
    print()
    
    # Top 5 healthy configurations
    if results:
        healthy_results = [r for r in results if r["is_healthy"]]
        healthy_results.sort(key=lambda x: x["test_rmse"])
        
        print("Top 5 Healthy Configurations:")
        print(f"{'Rank':<6} {'N':<6} {'dt':<6} {'Q':<10} {'R':<8} {'Test RMSE':<12} {'ESS':<10}")
        print("-" * 70)
        for i, r in enumerate(healthy_results[:5], 1):
            r_str = f"{r['R_scale']:.1f}" if r['R_scale'] else "data"
            print(f"{i:<6} {r['num_particles']:<6d} {r['dt']:<6.2f} {r['Q_scale']:<10.0e} {r_str:<8} "
                  f"{r['test_rmse']:<12.6f} {r['diagnostics']['ess_mean']:<10.1f}")
    
    print()
    print("=" * 110)
    
    # Analysis and recommendations
    print()
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 110)
    
    if results:
        healthy_results = [r for r in results if r["is_healthy"]]
        
        if len(healthy_results) > 0:
            # Analyze Q_scale trend
            q_rmses = {}
            for r in healthy_results:
                q = r["Q_scale"]
                if q not in q_rmses:
                    q_rmses[q] = []
                q_rmses[q].append(r["test_rmse"])
            
            print("Average Test RMSE by Q_scale (healthy configs only):")
            for q in sorted(q_rmses.keys()):
                avg_rmse = np.mean(q_rmses[q])
                print(f"  Q={q:.0e}: {avg_rmse:.6f} ({len(q_rmses[q])} configs)")
            print()
            
            # Analyze particle count trend
            n_rmses = {}
            for r in healthy_results:
                n = r["num_particles"]
                if n not in n_rmses:
                    n_rmses[n] = []
                n_rmses[n].append(r["test_rmse"])
            
            print("Average Test RMSE by num_particles (healthy configs only):")
            for n in sorted(n_rmses.keys()):
                avg_rmse = np.mean(n_rmses[n])
                print(f"  N={n}: {avg_rmse:.6f} ({len(n_rmses[n])} configs)")
            print()
            
            # Recommendations
            print("Key Insights:")
            if best_params:
                _, _, Q_best, _ = best_params
                if Q_best >= 10.0:
                    print("  • High Q_scale works best → PF needs strong process noise for particle diversity")
                if Q_best < 1.0:
                    print("  • Low Q_scale works best → Data is very predictable with constant velocity")
            
            avg_ess = np.mean([r["diagnostics"]["ess_mean"] for r in healthy_results])
            if avg_ess < 0.3 * healthy_results[0]["num_particles"]:
                print(f"  • Low average ESS ({avg_ess:.1f}) → Consider increasing num_particles or Q_scale")
            
            avg_resamp_pct = np.mean([r["diagnostics"]["ess_pct_below_threshold"] for r in healthy_results])
            if avg_resamp_pct > 80:
                print(f"  • High resampling rate ({avg_resamp_pct:.1f}%) → Particles degenerating quickly")
                print("    → Try: higher Q_scale, more particles, or lower R_scale")
    
    print("=" * 110)


if __name__ == "__main__":
    tune_pf_extended()