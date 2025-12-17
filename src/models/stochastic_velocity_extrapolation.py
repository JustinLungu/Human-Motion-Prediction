"""Particle Filter baseline for next-step IMU prediction.

Uses simplified particle ensemble averaging approach:
  - Particles represent different velocity estimates
  - No explicit measurement update step (avoids numerical instability)
  - Simple ensemble average of propagated particles
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseModel


class PFBaseline(BaseModel):
    """Simplified Particle Filter baseline using ensemble averaging.

    This is a regularized version that avoids aggressive measurement updates.
    """

    def __init__(
        self,
        num_particles: int = 100,
        Q_scale: float = 1.0,
        R_scale: Optional[float] = None,
        dt: float = 0.02,
        resample_threshold: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize PF baseline.

        Args:
            num_particles: number of particles (default 100)
            Q_scale: velocity perturbation scale
            R_scale: unused (for config compatibility)
            dt: time step for constant-velocity model
            resample_threshold: unused (for config compatibility)
            seed: optional random seed for reproducibility
        """
        super().__init__(name="pf_baseline")
        self.num_particles = num_particles
        self.Q_scale = Q_scale
        self.dt = dt
        self.seed = seed

        # Will be set during fit()
        self.rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit PF (no parameters to fit, just initialize RNG).

        Args:
            X: shape (N, T, 6) - training sequences
            y: shape (N, 6) - next-step targets (not used)
        """
        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6, "Expected 6 channels"

        # Initialize RNG
        self.rng = np.random.default_rng(self.seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run PF filter over sequences and predict next step.

        Args:
            X: shape (N, T, 6) - input sequences

        Returns:
            shape (N, 6) - predicted next-step IMU readings
        """
        if self.rng is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6, "Expected 6 channels"

        predictions = np.zeros((N, 6), dtype=np.float32)

        for n in range(N):
            seq = X[n, :, :]  # (T, 6)
            x_pred = self._filter_and_predict(seq)
            predictions[n, :] = x_pred

        return predictions

    def _filter_and_predict(self, seq: np.ndarray) -> np.ndarray:
        """Run simplified PF using ensemble averaging.

        Strategy:
        1. Estimate velocity from last two measurements
        2. Create particles with velocity + noise
        3. Propagate all particles one step
        4. Return mean position

        Args:
            seq: shape (T, 6) - single sequence

        Returns:
            shape (6,) - predicted next-step IMU readings
        """
        T, C = seq.shape
        assert C == 6

        # Initialize particles at last position
        particles = np.zeros((self.num_particles, 12), dtype=np.float64)
        particles[:, :6] = seq[-1, :]
        
        # Velocity from last two measurements (or zero if T=1)
        if T > 1:
            v_last = (seq[-1, :] - seq[-2, :]) / self.dt
        else:
            v_last = np.zeros(6)
        
        # Create ensemble: all particles use same velocity, but with noise
        for i in range(self.num_particles):
            particles[i, 6:] = v_last + self.rng.normal(0, self.Q_scale * 0.01, size=6)

        # Propagate one step
        particles_next = particles.copy()
        particles_next[:, :6] += self.dt * particles_next[:, 6:]

        # Return ensemble mean
        x_next = particles_next[:, :6].mean(axis=0)

        return x_next
