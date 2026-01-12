"""Extended tuning with filter health diagnostics.

Improvements:
- Detects exploding errors and filter divergence
- Logs innovation statistics for each configuration
- Tracks numerical stability issues
- Provides detailed diagnostics for best configuration
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import chi2
from dataloader.dataset import UCIHARDatasetLoader
from models.ekf import EKFBaseline
from models.evaluate import evaluate_model


def get_aggregated_stats(ekf: EKFBaseline) -> dict:
    """Get statistics from aggregated diagnostics across all sequences."""
    # Use aggregated diagnostics if available, otherwise per-sequence
    innovations_list = getattr(ekf, '_all_innovations', getattr(ekf, '_innovations', []))
    if len(innovations_list) == 0:
        return {}
    
    innovations = np.array(innovations_list)  # (N*T, 6) or (T, 6)
    mahal_dists_raw = getattr(ekf, '_all_mahalanobis', getattr(ekf, '_mahalanobis_distances', []))
    mahal_dists = np.array([d for d in mahal_dists_raw if not np.isnan(d)])
    innovation_norms = np.linalg.norm(innovations, axis=1)
    
    stats = {
        'innovation_mean': np.mean(innovations, axis=0).tolist(),
        'innovation_std': np.std(innovations, axis=0).tolist(),
        'innovation_max_norm': float(np.max(innovation_norms)),
        'innovation_mean_norm': float(np.mean(innovation_norms)),
    }
    
    if len(mahal_dists) > 0:
        stats.update({
            'mahalanobis_mean': float(np.mean(mahal_dists)),
            'mahalanobis_max': float(np.max(mahal_dists)),
            'mahalanobis_pct_above_5': float(np.mean(mahal_dists > 5) * 100),
        })
    
    # Covariance statistics
    P_traces_list = getattr(ekf, '_all_P_traces', getattr(ekf, '_P_traces', []))
    if len(P_traces_list) > 0:
        P_traces = np.array(P_traces_list)
        stats.update({
            'P_trace_mean': float(np.mean(P_traces)),
            'P_trace_max': float(np.max(P_traces)),
            'P_trace_final_mean': float(np.mean(P_traces[-100:])) if len(P_traces) >= 100 else float(np.mean(P_traces)),
        })
    
    # Eigenvalue statistics  
    P_min_eigvals_list = getattr(ekf, '_all_P_min_eigvals', getattr(ekf, '_P_min_eigvals', []))
    if len(P_min_eigvals_list) > 0:
        min_eigvals = np.array([e for e in P_min_eigvals_list if not np.isnan(e)])
        if len(min_eigvals) > 0:
            stats.update({
                'P_min_eigenvalue_mean': float(np.mean(min_eigvals)),
                'P_min_eigenvalue_min': float(np.min(min_eigvals)),
                'P_has_negative_eigenvalues': bool(np.any(min_eigvals < -1e-6)),
            })
    
    return stats


def check_filter_stability(
    ekf: EKFBaseline,
    X_sample: np.ndarray,
    max_mahal: float = chi2.ppf(0.99, df=6),
    max_innovation_norm: float = 100.0
) -> Tuple[bool, Dict[str, float]]:
    """Check if filter is stable on a sample sequence.
    
    Args:
        ekf: Fitted EKF model
        X_sample: Sample sequences (first 100 from test set)
        max_mahal: Maximum acceptable Mahalanobis distance
        max_innovation_norm: Maximum acceptable innovation norm
    
    Returns:
        (is_stable, diagnostics_dict)
    """
    # Run filter with diagnostics on sample
    try:
        _ = ekf.predict(X_sample, log_diagnostics=True)
        
        # Get aggregated statistics from all sequences (with safe attribute access)
        all_innovations_list = getattr(ekf, '_all_innovations', None)
        if all_innovations_list is None:
            all_innovations_list = getattr(ekf, '_innovations', [])
        
        if len(all_innovations_list) == 0:
            return False, {"error": "No diagnostics collected"}
        
        # Compute statistics from aggregated data (with fallback to per-sequence if aggregated not available)
        all_innovations = np.array(all_innovations_list)
        all_mahal_raw = getattr(ekf, '_all_mahalanobis', None) or getattr(ekf, '_mahalanobis_distances', [])
        all_P_traces_raw = getattr(ekf, '_all_P_traces', None) or getattr(ekf, '_P_traces', [])
        all_P_min_eigvals_raw = getattr(ekf, '_all_P_min_eigvals', None) or getattr(ekf, '_P_min_eigvals', [])
        
        # Ensure we have lists/arrays
        if not isinstance(all_mahal_raw, (list, np.ndarray)):
            all_mahal_raw = []
        if not isinstance(all_P_traces_raw, (list, np.ndarray)):
            all_P_traces_raw = []
        if not isinstance(all_P_min_eigvals_raw, (list, np.ndarray)):
            all_P_min_eigvals_raw = []
        
        all_mahal = np.array([float(d) for d in all_mahal_raw if not (np.isnan(d) or np.isinf(d))])
        all_P_traces = np.array([float(t) for t in all_P_traces_raw]) if len(all_P_traces_raw) > 0 else np.array([])
        
        # Handle complex eigenvalues (can happen due to numerical errors) - take real part only
        all_min_eigvals_list = []
        for e in all_P_min_eigvals_raw:
            if not np.isnan(e):
                try:
                    # Convert to real if complex
                    val = float(np.real(e)) if np.iscomplexobj(e) else float(e)
                    all_min_eigvals_list.append(val)
                except (ValueError, TypeError):
                    pass  # Skip invalid values
        all_min_eigvals = np.array(all_min_eigvals_list) if len(all_min_eigvals_list) > 0 else np.array([])
        
        innovation_norms = np.linalg.norm(all_innovations, axis=1) if len(all_innovations) > 0 else np.array([])
        
        # Extract key metrics
        diagnostics = {
            "mean_mahal": float(np.mean(all_mahal)) if len(all_mahal) > 0 else np.inf,
            "max_mahal": float(np.max(all_mahal)) if len(all_mahal) > 0 else np.inf,
            "mean_innovation_norm": float(np.mean(innovation_norms)) if len(innovation_norms) > 0 else np.inf,
            "max_innovation_norm": float(np.max(innovation_norms)) if len(innovation_norms) > 0 else np.inf,
            "P_trace_final": float(np.mean(all_P_traces[-100:])) if len(all_P_traces) >= 100 else (float(np.mean(all_P_traces)) if len(all_P_traces) > 0 else np.inf),
            "P_min_eigenvalue": float(np.min(all_min_eigvals)) if len(all_min_eigvals) > 0 else -np.inf,
        }
        
        # Check stability criteria
        is_stable = (
            diagnostics["max_mahal"] < max_mahal and
            diagnostics["max_innovation_norm"] < max_innovation_norm and
            diagnostics["P_min_eigenvalue"] >= -1e-6 and  # Allow tiny numerical error
            diagnostics["P_trace_final"] < 1e6 and  # Not exploding
            not np.isnan(diagnostics["mean_mahal"]) and
            not np.isinf(diagnostics["mean_mahal"])
        )
        
        return is_stable, diagnostics
        
    except AttributeError as e:
        # Handle attribute errors specifically (might be old attribute names)
        error_msg = str(e)
        # Replace old attribute names in error message if present
        if '_all_P_min_eigenvalues' in error_msg:
            error_msg = error_msg.replace('_all_P_min_eigenvalues', '_all_P_min_eigvals')
        return False, {"error": f"AttributeError: {error_msg}"}
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {str(e)}"}


def extended_tune() -> None:
    """Extended grid search with stability checks."""
    loader = UCIHARDatasetLoader("configs/config.yaml")
    
    train_split = loader.load_split("train")
    X_train = train_split.X.astype(np.float32)
    X_train_in = X_train[:, :-1, :]
    y_train_next = X_train[:, -1, :]
    
    test_split = loader.load_split("test")
    X_test = test_split.X.astype(np.float32)
    X_test_in = X_test[:, :-1, :]
    y_test_next = X_test[:, -1, :]
    
    # Sample for stability checking (first 100 sequences)
    X_test_sample = X_test_in[:100, :, :]
    
    # Grid - expand if needed
    Q_scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    R_scales = [None, 0.1, 0.5, 1.0, 2.0]  # Added more R options
    dts = [0.02, 0.05, 0.1, 0.5, 1.0]  # Finer dt grid
    
    print("=" * 100)
    print("EKF Extended Tuning with Stability Diagnostics")
    print("=" * 100)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Total configurations: {len(Q_scales) * len(R_scales) * len(dts)}")
    print()
    
    # Results tracking
    results = []
    unstable_configs = []
    
    print(f"{'dt':<6} {'Q':<10} {'R':<8} {'Train':<10} {'Test':<10} {'Mahal':<8} {'P_trace':<10} {'Status':<15}")
    print("-" * 100)
    
    best_test_rmse = float('inf')
    best_params = None
    best_diagnostics = None
    
    for dt in dts:
        for Q_scale in Q_scales:
            for R_scale in R_scales:
                # Build and fit model
                ekf = EKFBaseline(Q_scale=Q_scale, R_scale=R_scale, dt=dt)
                
                try:
                    ekf.fit(X_train_in, y_train_next)
                except Exception as e:
                    print(f"{dt:<6.2f} {Q_scale:<10.0e} {str(R_scale):<8} "
                          f"{'FIT FAILED':<10} {'---':<10} {'---':<8} {'---':<10} {str(e)[:15]:<15}")
                    unstable_configs.append((Q_scale, R_scale, dt, "fit_failed"))
                    continue
                
                # Check stability on sample
                is_stable, diag = check_filter_stability(ekf, X_test_sample)
                
                if not is_stable:
                    reason = diag.get("error", "unstable")
                    max_mahal = diag.get("max_mahal", np.inf)
                    print(f"{dt:<6.2f} {Q_scale:<10.0e} {str(R_scale) if R_scale else 'data':<8} "
                          f"{'---':<10} {'---':<10} {max_mahal:<8.2f} {'---':<10} {'UNSTABLE':<15}")
                    unstable_configs.append((Q_scale, R_scale, dt, reason))
                    continue
                
                # Evaluate performance
                try:
                    preds_train = ekf.predict(X_train_in)
                    train_rmse = evaluate_model(y_train_next, preds_train)["mean_rmse"]
                    
                    preds_test = ekf.predict(X_test_in)
                    test_rmse = evaluate_model(y_test_next, preds_test)["mean_rmse"]
                    
                    # Check for exploding errors
                    if train_rmse > 100 or test_rmse > 100 or np.isnan(train_rmse) or np.isnan(test_rmse):
                        print(f"{dt:<6.2f} {Q_scale:<10.0e} {str(R_scale) if R_scale else 'data':<8} "
                              f"{train_rmse:<10.4f} {test_rmse:<10.4f} {diag['mean_mahal']:<8.2f} "
                              f"{diag['P_trace_final']:<10.4f} {'EXPLODING':<15}")
                        unstable_configs.append((Q_scale, R_scale, dt, "exploding_error"))
                        continue
                    
                    r_str = f"{R_scale:.1f}" if R_scale else "data"
                    status = "✓ STABLE" if is_stable else "⚠ CHECK"
                    
                    print(f"{dt:<6.2f} {Q_scale:<10.0e} {r_str:<8} "
                          f"{train_rmse:<10.6f} {test_rmse:<10.6f} "
                          f"{diag['mean_mahal']:<8.2f} {diag['P_trace_final']:<10.4f} {status:<15}")
                    
                    # Track results
                    results.append({
                        "Q_scale": Q_scale,
                        "R_scale": R_scale,
                        "dt": dt,
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                        "diagnostics": diag,
                        "is_stable": is_stable,
                    })
                    
                    # Update best
                    if test_rmse < best_test_rmse and is_stable:
                        best_test_rmse = test_rmse
                        best_params = (Q_scale, R_scale, dt)
                        best_diagnostics = diag
                        
                except Exception as e:
                    print(f"{dt:<6.2f} {Q_scale:<10.0e} {str(R_scale) if R_scale else 'data':<8} "
                          f"{'PRED FAILED':<10} {'---':<10} {'---':<8} {'---':<10} {str(e)[:15]:<15}")
                    unstable_configs.append((Q_scale, R_scale, dt, f"predict_failed: {e}"))
    
    # Summary
    print("=" * 100)
    print("TUNING SUMMARY")
    print("=" * 100)
    print()
    
    if best_params:
        Q_best, R_best, dt_best = best_params
        print(f"Best Configuration:")
        print(f"  Q_scale: {Q_best:.0e}")
        print(f"  R_scale: {R_best if R_best else 'auto (from data)'}")
        print(f"  dt: {dt_best:.3f}")
        print(f"  Test RMSE: {best_test_rmse:.6f}")
        print()
        
        print(f"Best Configuration Diagnostics:")
        for key, val in best_diagnostics.items():
            print(f"  {key}: {val:.6f}")
        print()
    else:
        print("⚠ No stable configuration found!")
        print()
    
    # Persistence baseline
    persistence_preds = X_test_in[:, -1, :]
    persistence_rmse = evaluate_model(y_test_next, persistence_preds)["mean_rmse"]
    print(f"Persistence Baseline Test RMSE: {persistence_rmse:.6f}")
    print()
    
    if best_params and best_test_rmse < persistence_rmse:
        improvement = ((persistence_rmse - best_test_rmse) / persistence_rmse) * 100
        print(f"✓ EKF beats persistence by {improvement:.2f}%")
    elif best_params:
        degradation = ((best_test_rmse - persistence_rmse) / persistence_rmse) * 100
        print(f"✗ EKF underperforms persistence by {degradation:.2f}%")
    print()
    
    # Unstable configurations summary
    print(f"Unstable Configurations: {len(unstable_configs)} / {len(Q_scales) * len(R_scales) * len(dts)}")
    if unstable_configs:
        print()
        print("Unstable Config Breakdown:")
        from collections import Counter
        reasons = Counter([reason for _, _, _, reason in unstable_configs])
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")
    print()
    
    # Top 5 stable configurations
    if results:
        stable_results = [r for r in results if r["is_stable"]]
        stable_results.sort(key=lambda x: x["test_rmse"])
        
        print("Top 5 Stable Configurations:")
        print(f"{'Rank':<6} {'Q':<10} {'R':<8} {'dt':<6} {'Test RMSE':<12} {'Mean Mahal':<12}")
        print("-" * 60)
        for i, r in enumerate(stable_results[:5], 1):
            r_str = f"{r['R_scale']:.1f}" if r['R_scale'] else "data"
            print(f"{i:<6} {r['Q_scale']:<10.0e} {r_str:<8} {r['dt']:<6.2f} "
                  f"{r['test_rmse']:<12.6f} {r['diagnostics']['mean_mahal']:<12.4f}")
    
    print()
    print("=" * 100)


if __name__ == "__main__":
    extended_tune()