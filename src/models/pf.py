"""Standard Particle Filter baseline for next-step IMU prediction.

Uses importance weighting + resampling with full sequence iteration.

State: [x1..x6, v1..v6] (12D constant-velocity model)
Dynamics: x_{k+1} = x_k + dt * v_k + n_x; v_{k+1} = v_k + n_v
Measurement: z = x + noise (observe position only)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseModel


class PFBaseline(BaseModel):
    """Standard Particle Filter with importance weighting and resampling."""

    def __init__(
        self,
        num_particles: int = 200,
        Q_scale: float = 1.0,
        R_scale: Optional[float] = None,
        dt: float = 0.02,
        resample_threshold: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize standard PF baseline.

        Args:
            num_particles: number of particles
            Q_scale: process noise scale for Q = Q_scale * I_12x12
            R_scale: measurement noise scale. If None, computed from training data.
            dt: time step for constant-velocity model
            resample_threshold: ESS threshold (fraction of num_particles)
            seed: random seed for reproducibility
        """
        super().__init__(name="pf_baseline")
        self.num_particles = num_particles
        self.Q_scale = Q_scale
        self.R_scale = R_scale
        self.dt = dt
        self.resample_threshold = resample_threshold
        self.seed = seed

        # Set during fit()
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self._R_diag_from_data: Optional[np.ndarray] = None
        self.rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit PF parameters from training data.

        Args:
            X: shape (N, T, 6) - training sequences
            y: shape (N, 6) - next-step targets (not used)
        """
        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6, "Expected 6 channels"

        # Initialize RNG
        self.rng = np.random.default_rng(self.seed)

        # Compute per-channel variance of the signal over time (measurement variance baseline)
        # NOTE: use variance of X itself, not differences (differences reflect dynamics)
        meas_var = np.var(X, axis=(0, 1))
        self._R_diag_from_data = meas_var

        # Initialize Q and R
        self.Q = np.eye(12) * self.Q_scale
        if self.R_scale is not None:
            self.R = np.diag(self._R_diag_from_data * self.R_scale)
        else:
            self.R = np.diag(self._R_diag_from_data)

    def predict(self, X: np.ndarray, log_diagnostics: bool = False) -> np.ndarray:
        """Run PF and predict next step.
        
        Args:
            X: shape (N, T, 6)
            log_diagnostics: if True, collect ESS statistics
        
        Returns:
            shape (N, 6)
        """
        if self.Q is None or self.R is None or self.rng is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6

        # Initialize diagnostic storage
        if log_diagnostics:
            self._all_ess = []

        predictions = np.zeros((N, 6), dtype=np.float32)

        for n in range(N):
            seq = X[n, :, :]  # (T, 6)
            x_pred = self._filter_and_predict(seq, log_diagnostics=log_diagnostics)
            predictions[n, :] = x_pred

        return predictions

    def _filter_and_predict(self, seq: np.ndarray, log_diagnostics: bool = False) -> np.ndarray:
        """Run standard PF over full sequence and predict next step.

        Args:
            seq: shape (T, 6)
            log_diagnostics: if True, collect ESS statistics

        Returns:
            shape (6,)
        """
        T, C = seq.shape
        assert C == 6

        # Initialize diagnostic storage for this sequence
        if log_diagnostics:
            self._ess = []

        # 1. Initialize particles
        particles = np.zeros((self.num_particles, 12), dtype=np.float64)
        weights = np.ones(self.num_particles) / self.num_particles

        # Position around z_0
        particles[:, :6] = seq[0, :] + self.rng.normal(0, 0.01, size=(self.num_particles, 6))

        # Velocity around z_1 - z_0
        if T > 1:
            v_init = (seq[1, :] - seq[0, :]) / self.dt
            particles[:, 6:] = v_init + self.rng.normal(0, 0.01, size=(self.num_particles, 6))
        else:
            particles[:, 6:] = 0.0

        # 2. Standard PF loop (textbook): for t = 1..T-1 -> Predict, Update, Normalize, Resample
        # Do NOT weight at t=0 (we already initialized particles around z0).
        for t in range(1, T):
            # Predict (propagate with process noise)
            particles = self._propagate(particles)

            # Update weights using measurement z_t
            z_t = seq[t, :]
            weights = self._update_weights(particles, weights, z_t)

            # Normalize weights (with safety)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(self.num_particles) / self.num_particles

            # Resample if ESS is low
            ess = 1.0 / np.sum(weights ** 2)
            threshold = self.resample_threshold * self.num_particles
            
            if log_diagnostics:
                self._ess.append(ess)
            
            if ess < threshold:
                particles, weights = self._resample(particles, weights)

        # Aggregate ESS for this sequence
        if log_diagnostics and hasattr(self, '_ess'):
            if hasattr(self, '_all_ess'):
                self._all_ess.append(self._ess.copy())
            else:
                self._all_ess = [self._ess.copy()]

        # 3. Final one-step-ahead prediction: deterministic
        # Take the weighted mean state, then propagate deterministically (no process noise)
        state_mean = (particles * weights.reshape(-1, 1)).sum(axis=0)
        # Deterministic propagation: x += dt * v
        state_mean[:6] = state_mean[:6] + self.dt * state_mean[6:]

        return state_mean[:6]

    def _propagate(self, particles: np.ndarray) -> np.ndarray:
        """Propagate particles through dynamics with process noise.

        x += dt * v + noise_x
        v += noise_v

        Args:
            particles: shape (num_particles, 12)

        Returns:
            shape (num_particles, 12)
        """
        particles = particles.copy()

        # Position: x += dt * v + noise
        particles[:, :6] += self.dt * particles[:, 6:]
        particles[:, :6] += self.rng.normal(0, np.sqrt(self.Q[0, 0]), size=(self.num_particles, 6))

        # Velocity: v += noise
        particles[:, 6:] += self.rng.normal(0, np.sqrt(self.Q[6, 6]), size=(self.num_particles, 6))

        return particles

    def _update_weights(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Update weights using measurement likelihood.

        w_i = w_i * p(z | x_i) where p(z|x) ~ N(z; x[:6], R)

        Args:
            particles: shape (num_particles, 12)
            weights: shape (num_particles,)
            z: shape (6,) - measurement

        Returns:
            shape (num_particles,) - unnormalized weights
        """
        z_hat = particles[:, :6]  # Predicted measurement (position)
        residual = z - z_hat  # (num_particles, 6)

        # Log likelihood: -0.5 * e^T R^-1 e
        R_diag = np.maximum(np.diag(self.R), 1e-8)  # Add epsilon floor to avoid division by near-zero
        log_likelihood = -0.5 * (residual ** 2 / R_diag).sum(axis=1)

        # Numerical stability
        max_ll = np.max(log_likelihood)
        likelihood = np.exp(log_likelihood - max_ll)

        # Update weights
        updated_weights = weights * likelihood

        return updated_weights

    def _resample(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multinomial resampling.

        Args:
            particles: shape (num_particles, 12)
            weights: shape (num_particles,)

        Returns:
            (resampled particles, uniform weights)
        """
        indices = self.rng.choice(self.num_particles, size=self.num_particles, p=weights)
        particles_new = particles[indices, :].copy()
        weights_new = np.ones(self.num_particles) / self.num_particles

        return particles_new, weights_new

    def get_ess_statistics(self) -> dict:
        """Return statistics about ESS from last filter run.
        
        Returns:
            Dictionary with ESS statistics
        """
        ess_list = getattr(self, '_all_ess', getattr(self, '_ess', []))
        
        if len(ess_list) == 0:
            return {
                "mean_ess": np.nan,
                "min_ess": np.nan,
                "max_ess": np.nan,
                "std_ess": np.nan,
                "ess_pct_below_threshold": np.nan,
                "num_resamples": np.nan,
            }
        
        # Flatten all ESS values from all sequences
        all_ess_values = []
        num_resamples = 0
        threshold = self.resample_threshold * self.num_particles
        
        for ess_seq in ess_list:
            all_ess_values.extend(ess_seq)
            num_resamples += sum(1 for ess in ess_seq if ess < threshold)
        
        if len(all_ess_values) == 0:
            return {
                "mean_ess": np.nan,
                "min_ess": np.nan,
                "max_ess": np.nan,
                "std_ess": np.nan,
                "ess_pct_below_threshold": np.nan,
                "num_resamples": np.nan,
            }
        
        ess_array = np.array(all_ess_values)
        
        return {
            "mean_ess": float(np.mean(ess_array)),
            "min_ess": float(np.min(ess_array)),
            "max_ess": float(np.max(ess_array)),
            "std_ess": float(np.std(ess_array)),
            "ess_pct_below_threshold": float(np.mean(ess_array < threshold) * 100),
            "num_resamples": int(num_resamples),
        }

    def plot_ess_curves(self, save_path: Optional[str] = None):
        """Plot average ESS curve across all sequences.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib figure or None
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, cannot plot ESS curves")
            return None
        
        ess_list = getattr(self, '_all_ess', getattr(self, '_ess', []))
        
        if len(ess_list) == 0:
            print("No ESS data available. Run predict() with log_diagnostics=True first.")
            return None
        
        # Find maximum sequence length
        max_len = max(len(ess_seq) for ess_seq in ess_list)
        
        # Pad sequences to same length and compute average
        ess_matrix = []
        for ess_seq in ess_list:
            padded = np.pad(ess_seq, (0, max_len - len(ess_seq)), mode='constant', constant_values=np.nan)
            ess_matrix.append(padded)
        
        ess_matrix = np.array(ess_matrix)
        
        # Compute mean and std across sequences (ignoring NaN)
        ess_mean = np.nanmean(ess_matrix, axis=0)
        ess_std = np.nanstd(ess_matrix, axis=0)
        
        # Create single plot
        fig, ax = plt.subplots(figsize=(12, 6))
        time_steps = np.arange(len(ess_mean))
        threshold = self.resample_threshold * self.num_particles
        
        # Plot average ESS curve with std band
        ax.plot(time_steps, ess_mean, 'b-', linewidth=2, label='Mean ESS', alpha=0.8)
        ax.fill_between(time_steps, ess_mean - ess_std, ess_mean + ess_std, 
                       alpha=0.3, color='blue', label='±1 Std')
        
        # Plot threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                  label=f'Resample Threshold ({threshold:.1f})', alpha=0.7)
        
        # Plot max ESS line (num_particles)
        ax.axhline(y=self.num_particles, color='g', linestyle=':', linewidth=1.5,
                  label=f'Max ESS ({self.num_particles})', alpha=0.5)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('ESS', fontsize=12)
        ax.set_title(f'Average ESS Curve (N={self.num_particles}, threshold={self.resample_threshold:.2f}, {len(ess_list)} sequences)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return fig