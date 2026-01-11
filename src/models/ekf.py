"""Extended Kalman Filter baseline for next-step IMU prediction.

State model (constant velocity):
  - State: [x1, x2, ..., x6, v1, v2, ..., v6] (12-dim)
  - Dynamics: x_{k+1} = x_k + dt * v_k; v_{k+1} = v_k (constant velocity)
  - Process noise: Q (12x12, diagonal)

Measurement model:
  - Measurement: z = [x1, x2, ..., x6] + noise (6-dim)
  - Measurement matrix: H = [I_6x6 | 0_6x6] (observe position, not velocity)
  - Measurement noise: R (6x6, diagonal)

Next-step prediction:
  - Filter over entire window, then predict one step ahead
  - Return the predicted x (position) as the next-step estimate
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseModel


class EKFBaseline(BaseModel):
    """Extended Kalman Filter baseline using constant-velocity state model.

    The filter processes a sequence of observations and predicts the next step.
    For simplicity, we use linear dynamics and measurement, so this is just
    a standard Kalman Filter (but we call it EKF for generality).
    """

    def __init__(
        self,
        Q_scale: float = 1e-4,
        R_scale: Optional[float] = None,
        dt: float = 1.0,
    ) -> None:
        """Initialize EKF baseline.

        Args:
            Q_scale: scale for process noise covariance Q = q * I_12x12
            R_scale: scale for measurement noise covariance R = r * I_6x6.
                     If None, R will be computed from training data.
            dt: time step for constant-velocity model
        """
        super().__init__(name="ekf_baseline")
        self.Q_scale = Q_scale
        self.R_scale = R_scale
        self.dt = dt

        # Will be set during fit()
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self._R_diag_from_data: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit EKF parameters (Q and R) from training data.

        Args:
            X: shape (N, T, 6) - training sequences
            y: shape (N, 6) - next-step targets (not used for EKF, but for interface)
        """
        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6, "Expected 6 channels"

        # Estimate R from measurement noise (per-channel variance of differences)
        diffs = np.diff(X, axis=1)  # (N, T-1, 6) - finite differences as proxy for noise
        meas_var = np.var(diffs, axis=(0, 1))  # (6,)
        # Ensure R is not singular: add small epsilon to prevent zero variance
        meas_var = np.maximum(meas_var, 1e-10)
        self._R_diag_from_data = meas_var

        # Initialize Q (process noise covariance, 12x12 diagonal)
        self.Q = np.eye(12) * self.Q_scale

        # Initialize R (measurement noise covariance, 6x6 diagonal)
        if self.R_scale is not None:
            self.R = np.eye(6) * self.R_scale
        else:
            # Use measurement variance directly (unscaled); tuning can adjust via R_scale
            self.R = np.diag(self._R_diag_from_data)

    def predict(self, X: np.ndarray, log_diagnostics: bool = False) -> np.ndarray:
        """Run EKF filter over sequences and predict next step.
        
        Args:
            X: shape (N, T, 6) - input sequences
            log_diagnostics: if True, collect innovation statistics
        
        Returns:
            shape (N, 6) - predicted next-step IMU readings
        """
        if self.Q is None or self.R is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6, "Expected 6 channels"

        predictions = np.zeros((N, 6), dtype=np.float32)
        
        # Initialize diagnostic storage
        if log_diagnostics:
            self._all_innovations = []
            self._all_innovation_norms = []
            self._all_mahalanobis = []
            self._all_P_traces = []
            self._all_P_min_eigvals = []

        for n in range(N):
            seq = X[n, :, :]  # (T, 6)
            # Run filter and predict
            x_pred = self._filter_and_predict(seq, log_diagnostics=log_diagnostics)
            predictions[n, :] = x_pred
            
            # Aggregate diagnostics from this sequence
            if log_diagnostics and hasattr(self, '_innovations'):
                self._all_innovations.extend(self._innovations)
                if hasattr(self, '_innovation_norms'):
                    self._all_innovation_norms.extend(self._innovation_norms)
                if hasattr(self, '_mahalanobis_distances'):
                    self._all_mahalanobis.extend(self._mahalanobis_distances)
                if hasattr(self, '_P_traces'):
                    self._all_P_traces.extend(self._P_traces)
                if hasattr(self, '_P_min_eigvals'):
                    self._all_P_min_eigvals.extend(self._P_min_eigvals)

        return predictions

    def _filter_and_predict(self, seq: np.ndarray, log_diagnostics: bool = True) -> np.ndarray:
        """Run EKF over a single sequence and return next-step prediction.
        
        Args:
            seq: shape (T, 6) - single sequence
            log_diagnostics: if True, store and return diagnostics
        
        Returns:
            shape (6,) - predicted next-step IMU readings
        """
        T, C = seq.shape
        assert C == 6
        
        # Diagnostics storage
        if log_diagnostics:
            self._innovations = []  # List of (6,) arrays
            self._innovation_norms = []  # List of scalars
            self._mahalanobis_distances = []  # List of scalars
            self._P_traces = []  # Trace of covariance matrix over time
            self._P_min_eigvals = []  # Minimum eigenvalue of P at each step
        
        # Initialize state
        x_hat = np.zeros(12, dtype=np.float64)
        x_hat[:6] = seq[0, :]
        x_hat[6:] = 0.0
        P = np.eye(12, dtype=np.float64) * 1.0
        
        # Update with z0
        z0 = seq[0, :]
        x_hat, P = self._update_step(x_hat, P, z0, log_diagnostics=log_diagnostics)
        
        # Main filter loop
        for t in range(1, T):
            z = seq[t, :]
            
            # Predict
            x_hat, P = self._predict_step(x_hat, P)
            
            # Update
            x_hat, P = self._update_step(x_hat, P, z, log_diagnostics=log_diagnostics)
            
            # Log covariance trace (measure of total uncertainty)
            if log_diagnostics:
                self._P_traces.append(np.trace(P))
                # Check for negative eigenvalues (numerical instability indicator)
                try:
                    # Ensure P is symmetric before eigenvalue computation (extra safeguard)
                    P_sym = (P + P.T) / 2
                    eigvals = np.linalg.eigvals(P_sym)
                    # Take real part (should be real for symmetric matrices, but numerical errors can make them complex)
                    eigvals = np.real(eigvals)
                    # Filter out any remaining NaN/Inf values
                    eigvals = eigvals[np.isfinite(eigvals)]
                    if len(eigvals) > 0:
                        self._P_min_eigvals.append(float(np.min(eigvals)))
                    else:
                        self._P_min_eigvals.append(np.nan)
                except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                    self._P_min_eigvals.append(np.nan)
        
        # Final predict
        x_next, P_next = self._predict_step(x_hat, P)
        
        return x_next[:6]

    def _predict_step(
        self, x_hat: np.ndarray, P: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """EKF predict step.

        Dynamics: x[0:6] += dt * x[6:12]; x[6:12] unchanged (constant velocity)

        Args:
            x_hat: shape (12,) - current state estimate
            P: shape (12, 12) - current covariance

        Returns:
            (x_hat_pred, P_pred): predicted state and covariance
        """
        # State transition matrix F for constant velocity model
        F = np.eye(12, dtype=np.float64)
        for i in range(6):
            F[i, 6 + i] = self.dt  # position += dt * velocity

        # Jacobian of dynamics (linear, so F_k = F)
        # x_hat_pred = F @ x_hat
        x_hat_pred = F @ x_hat

        # P_pred = F @ P @ F^T + Q
        P_pred = F @ P @ F.T + self.Q
        
        # Ensure P remains symmetric (numerical errors can break symmetry)
        P_pred = (P_pred + P_pred.T) / 2

        return x_hat_pred, P_pred

    def _update_step(
        self, x_hat: np.ndarray, P: np.ndarray, z: np.ndarray,
        log_diagnostics: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """EKF update step.

        Measurement: z = H @ x + noise, where H = [I_6x6 | 0_6x6]

        Args:
            x_hat: shape (12,) - predicted state
            P: shape (12, 12) - predicted covariance
            z: shape (6,) - measurement

        Returns:
            (x_hat_upd, P_upd): updated state and covariance
        """
        # Measurement matrix (observe position, not velocity)
        H = np.zeros((6, 12), dtype=np.float64)
        H[:6, :6] = np.eye(6)
        
        # Innovation (residual)
        z_hat = H @ x_hat
        innovation = z - z_hat 
        
        # Innovation covariance
        S = H @ P @ H.T + self.R
        S = (S + S.T) / 2
        S += np.eye(6) * 1e-8  # Proactive regularization
        
        # Mahalanobis distance: innovation^T @ S^(-1) @ innovation
        # Measures "how many sigmas" away the measurement is
        if log_diagnostics:
            try:
                mahal_dist = np.sqrt(innovation @ np.linalg.solve(S, innovation))
                self._mahalanobis_distances.append(mahal_dist)
            except:
                self._mahalanobis_distances.append(np.nan)
            
            self._innovations.append(innovation.copy())
            self._innovation_norms.append(np.linalg.norm(innovation))
        
        # Kalman gain
        try:
            K = np.linalg.solve(S, H @ P).T
        except np.linalg.LinAlgError:
            K = P @ H.T @ np.linalg.pinv(S)
            if log_diagnostics:
                print(f"Warning: Singular S matrix, using pseudo-inverse")
        
        # Update
        x_hat_upd = x_hat + K @ innovation
        P_upd = (np.eye(12) - K @ H) @ P
        # Ensure P remains symmetric (numerical errors can break symmetry)
        P_upd = (P_upd + P_upd.T) / 2
        
        return x_hat_upd, P_upd

    def get_innovation_statistics(self) -> dict:
        """Return statistics about innovations from last filter run.
        
        Call after predict() with log_diagnostics=True.
        """
        # Use aggregated diagnostics if available (multiple sequences), otherwise per-sequence
        innovations_list = getattr(self, '_all_innovations', getattr(self, '_innovations', []))
        mahal_dists_list = getattr(self, '_all_mahalanobis', getattr(self, '_mahalanobis_distances', []))
        innovation_norms_list = getattr(self, '_all_innovation_norms', getattr(self, '_innovation_norms', []))
        P_traces_list = getattr(self, '_all_P_traces', getattr(self, '_P_traces', []))
        P_min_eigvals_list = getattr(self, '_all_P_min_eigvals', getattr(self, '_P_min_eigvals', []))
        
        if len(innovations_list) == 0:
            return {}
        
        innovations = np.array(innovations_list)  # (T, 6) or (N*T, 6) if aggregated
        mahal_dists = np.array([d for d in mahal_dists_list if not np.isnan(d)])
        
        stats = {
            'innovation_mean': np.mean(innovations, axis=0).tolist(),  # Should be ~0
            'innovation_std': np.std(innovations, axis=0).tolist(),
        }
        
        if len(innovation_norms_list) > 0:
            stats.update({
                'innovation_max_norm': float(np.max(innovation_norms_list)),
                'innovation_mean_norm': float(np.mean(innovation_norms_list)),
            })
        else:
            # Compute from innovations if norms not available
            norms = np.linalg.norm(innovations, axis=1)
            if len(norms) > 0:
                stats.update({
                    'innovation_max_norm': float(np.max(norms)),
                    'innovation_mean_norm': float(np.mean(norms)),
                })
        
        if len(mahal_dists) > 0:
            stats.update({
                'mahalanobis_mean': float(np.mean(mahal_dists)),
                'mahalanobis_max': float(np.max(mahal_dists)),
                'mahalanobis_pct_above_5': float(np.mean(mahal_dists > 5) * 100),
            })
        
        # Covariance statistics
        if len(P_traces_list) > 0:
            P_traces = np.array(P_traces_list)
            stats.update({
                'P_trace_initial': float(P_traces[0]) if len(P_traces) > 0 else None,
                'P_trace_final': float(P_traces[-1]) if len(P_traces) > 0 else None,
                'P_trace_change': float(P_traces[-1] - P_traces[0]) if len(P_traces) > 1 else None,
                'P_trace_is_monotonic_increasing': bool(np.all(np.diff(P_traces) >= -1e-10)) if len(P_traces) > 1 else None,
            })
            
            # Check if Trace(P) is approaching zero
            if len(P_traces) > 10:
                recent_traces = P_traces[-10:]
                stats['P_trace_trending_to_zero'] = bool(np.max(recent_traces) < 0.01)
        
        # Eigenvalue statistics
        if len(P_min_eigvals_list) > 0:
            min_eigvals = np.array([e for e in P_min_eigvals_list if not np.isnan(e)])
            if len(min_eigvals) > 0:
                stats.update({
                    'P_min_eigenvalue_mean': float(np.mean(min_eigvals)),
                    'P_min_eigenvalue_min': float(np.min(min_eigvals)),
                    'P_has_negative_eigenvalues': bool(np.any(min_eigvals < -1e-6)),
                })
        
        return stats
    
    def check_filter_health(self) -> dict:
        """Check for common filter health issues.
        
        Returns:
            Dictionary with health checks and warnings
        """
        warnings_list = []
        
        # Use aggregated diagnostics if available, otherwise per-sequence
        mahal_dists_list = getattr(self, '_all_mahalanobis', getattr(self, '_mahalanobis_distances', []))
        P_traces_list = getattr(self, '_all_P_traces', getattr(self, '_P_traces', []))
        P_min_eigvals_list = getattr(self, '_all_P_min_eigvals', getattr(self, '_P_min_eigvals', []))
        
        if len(mahal_dists_list) == 0:
            return {"status": "no_diagnostics", "warnings": []}
        
        # Check Mahalanobis distance consistently > 5
        mahal_dists = np.array([d for d in mahal_dists_list if not np.isnan(d)])
        if len(mahal_dists) > 0:
            pct_above_5 = np.mean(mahal_dists > 5) * 100
            if pct_above_5 > 50:  # More than 50% above 5
                warnings_list.append(
                    f"Mahalanobis distance consistently >5 ({pct_above_5:.1f}% of time). "
                    "Filter may be diverging or Q/R badly tuned."
                )
        
        # Check Trace(P) monotonically increasing
        if len(P_traces_list) > 1:
            P_traces = np.array(P_traces_list)
            is_monotonic = np.all(np.diff(P_traces) >= -1e-10)
            if is_monotonic and P_traces[-1] > P_traces[0] * 1.1:  # 10% increase
                warnings_list.append(
                    f"Trace(P) monotonically increasing ({P_traces[0]:.4f} -> {P_traces[-1]:.4f}). "
                    "Process noise (Q) may be too high."
                )
            
            # Check if Trace(P) → 0
            if len(P_traces) > 10:
                recent_traces = P_traces[-10:]
                if np.max(recent_traces) < 0.01:
                    warnings_list.append(
                        f"Trace(P) → 0 (recent values: {np.max(recent_traces):.6f}). "
                        "Filter may be overconfident, ignoring measurements."
                    )
        
        # Check for negative eigenvalues
        if len(P_min_eigvals_list) > 0:
            min_eigvals = np.array([e for e in P_min_eigvals_list if not np.isnan(e)])
            if len(min_eigvals) > 0 and np.any(min_eigvals < -1e-6):
                n_negative = np.sum(min_eigvals < -1e-6)
                warnings_list.append(
                    f"Negative eigenvalues detected in P ({n_negative}/{len(min_eigvals)} steps). "
                    "Possible numerical instability."
                )
        
        return {
            "status": "healthy" if len(warnings_list) == 0 else "warnings",
            "warnings": warnings_list,
        }

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """Plot innovation and covariance diagnostics."""
        import matplotlib.pyplot as plt
        
        # Use aggregated diagnostics if available, otherwise per-sequence
        innovations_list = getattr(self, '_all_innovations', getattr(self, '_innovations', []))
        innovation_norms_list = getattr(self, '_all_innovation_norms', getattr(self, '_innovation_norms', []))
        mahal_dists_list = getattr(self, '_all_mahalanobis', getattr(self, '_mahalanobis_distances', []))
        P_traces_list = getattr(self, '_all_P_traces', getattr(self, '_P_traces', []))
        P_min_eigvals_list = getattr(self, '_all_P_min_eigvals', getattr(self, '_P_min_eigvals', []))
        
        if len(innovations_list) == 0:
            print("No diagnostics available. Run predict() with log_diagnostics=True first.")
            return None
        
        # Compute innovation norms from innovations if not available
        if len(innovation_norms_list) == 0 and len(innovations_list) > 0:
            innovations_array = np.array(innovations_list)
            innovation_norms_list = np.linalg.norm(innovations_array, axis=1).tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Innovation norms over time
        if len(innovation_norms_list) > 0:
            axes[0, 0].plot(innovation_norms_list, linewidth=1.5)
        axes[0, 0].set_ylabel('Innovation Norm')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_title('Innovation Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mahalanobis distance
        if len(mahal_dists_list) > 0:
            axes[0, 1].plot(mahal_dists_list, label='Mahalanobis', linewidth=1.5)
        axes[0, 1].axhline(y=3, color='r', linestyle='--', label='3-sigma', alpha=0.7)
        axes[0, 1].axhline(y=5, color='orange', linestyle='--', label='5-sigma (warning)', alpha=0.7)
        axes[0, 1].set_ylabel('Mahalanobis Distance')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_title('Normalized Innovation (Mahalanobis Distance)')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Covariance trace
        if len(P_traces_list) > 0:
            axes[0, 2].plot(P_traces_list, linewidth=1.5, color='green')
            if len(P_traces_list) > 1:
                # Highlight if monotonically increasing
                diff = np.diff(P_traces_list)
                if np.all(diff >= -1e-10) and P_traces_list[-1] > P_traces_list[0] * 1.1:
                    axes[0, 2].plot(P_traces_list, 'r--', alpha=0.5, label='Monotonic increasing!')
                    axes[0, 2].legend(fontsize=8)
        axes[0, 2].set_ylabel('Trace(P)')
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_title('Total Uncertainty (Trace of P)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # P minimum eigenvalues (numerical stability check)
        if len(P_min_eigvals_list) > 0:
            min_eigvals = np.array(P_min_eigvals_list)
            valid_mask = ~np.isnan(min_eigvals)
            if np.any(valid_mask):
                axes[1, 0].plot(min_eigvals, linewidth=1.5, color='purple')
                axes[1, 0].axhline(y=0, color='r', linestyle='--', label='Zero (instability)', alpha=0.7)
                if np.any(min_eigvals < -1e-6):
                    axes[1, 0].scatter(
                        np.where(min_eigvals < -1e-6)[0],
                        min_eigvals[min_eigvals < -1e-6],
                        color='red', s=30, zorder=5, label='Negative!'
                    )
                    axes[1, 0].legend(fontsize=8)
                axes[1, 0].set_ylabel('Min Eigenvalue of P')
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_title('P Min Eigenvalue (Numerical Stability)')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No valid eigenvalue data', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('P Min Eigenvalue')
        else:
            axes[1, 0].text(0.5, 0.5, 'Eigenvalue data not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('P Min Eigenvalue')
        
        # Per-channel innovations
        if len(innovations_list) > 0:
            innovations = np.array(innovations_list)
            for i in range(6):
                axes[1, 1].plot(innovations[:, i], alpha=0.6, label=f'Ch {i}', linewidth=1.2)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_ylabel('Innovation')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_title('Per-Channel Innovations')
        axes[1, 1].legend(fontsize=7, ncol=3, loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics text
        axes[1, 2].axis('off')
        stats = self.get_innovation_statistics()
        health = self.check_filter_health()
        
        summary_text = "DIAGNOSTIC SUMMARY\n" + "=" * 40 + "\n\n"
        
        if stats:
            summary_text += f"Max Mahalanobis: {stats.get('mahalanobis_max', 'N/A'):.3f}\n"
            if stats.get('mahalanobis_pct_above_5', 0) > 0:
                summary_text += f"  {stats['mahalanobis_pct_above_5']:.1f}% > 5 (⚠)\n"
            summary_text += f"\nTrace(P): {stats.get('P_trace_initial', 'N/A'):.4f} → {stats.get('P_trace_final', 'N/A'):.4f}\n"
            if stats.get('P_trace_is_monotonic_increasing'):
                summary_text += "  ⚠ Monotonic increasing!\n"
            if stats.get('P_trace_trending_to_zero'):
                summary_text += "  ⚠ Trending to zero!\n"
            if stats.get('P_has_negative_eigenvalues'):
                summary_text += f"\n⚠ Negative eigenvalues: {stats.get('P_min_eigenvalue_min', 'N/A'):.6f}\n"
        
        summary_text += f"\nStatus: {health.get('status', 'unknown')}\n"
        if health.get('warnings'):
            summary_text += f"Warnings: {len(health['warnings'])}\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=9, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved diagnostics plot to: {save_path}")
        
        return fig