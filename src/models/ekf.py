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
        self._R_diag_from_data = meas_var

        # Initialize Q (process noise covariance, 12x12 diagonal)
        self.Q = np.eye(12) * self.Q_scale

        # Initialize R (measurement noise covariance, 6x6 diagonal)
        if self.R_scale is not None:
            self.R = np.eye(6) * self.R_scale
        else:
            # Use measurement variance directly (unscaled); tuning can adjust via R_scale
            self.R = np.diag(self._R_diag_from_data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run EKF filter over sequences and predict next step.

        Args:
            X: shape (N, T, 6) - input sequences

        Returns:
            shape (N, 6) - predicted next-step IMU readings
        """
        if self.Q is None or self.R is None:
            raise RuntimeError("Model not fitted yet")

        X = np.asarray(X, dtype=np.float64)
        N, T, C = X.shape
        assert C == 6, "Expected 6 channels"

        predictions = np.zeros((N, 6), dtype=np.float32)

        for n in range(N):
            seq = X[n, :, :]  # (T, 6)
            # Run filter and predict
            x_pred = self._filter_and_predict(seq)
            predictions[n, :] = x_pred

        return predictions

    def _filter_and_predict(self, seq: np.ndarray) -> np.ndarray:
        """Run EKF over a single sequence and return next-step prediction.

        Loop order (correct for Kalman filtering):
        1. Initialize state from z0, update with z0 (no predict step first)
        2. For t=1..T-1: predict then update with z_t
        3. Final predict step to get next-step estimate

        Args:
            seq: shape (T, 6) - single sequence

        Returns:
            shape (6,) - predicted next-step IMU readings
        """
        T, C = seq.shape
        assert C == 6

        # Initialize state: position from first frame, zero velocity
        x_hat = np.zeros(12, dtype=np.float64)
        x_hat[:6] = seq[0, :]  # initial position from z0
        x_hat[6:] = 0.0  # initial velocity

        # Initialize covariance (large uncertainty)
        P = np.eye(12, dtype=np.float64) * 1.0

        # Update with z0 first (before any predict step)
        z0 = seq[0, :]
        x_hat, P = self._update_step(x_hat, P, z0)

        # Then for t=1..T-1: predict then update
        for t in range(1, T):
            z = seq[t, :]

            # Predict step
            x_hat, P = self._predict_step(x_hat, P)

            # Update step
            x_hat, P = self._update_step(x_hat, P, z)

        # Final predict step to get next-step estimate
        x_next, _ = self._predict_step(x_hat, P)

        # Return only position (first 6 dims) as the next-step IMU estimate
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

        return x_hat_pred, P_pred

    def _update_step(
        self, x_hat: np.ndarray, P: np.ndarray, z: np.ndarray
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

        # Innovation
        z_hat = H @ x_hat
        y = z - z_hat  # residual

        # Innovation covariance
        S = H @ P @ H.T + self.R

        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        x_hat_upd = x_hat + K @ y
        P_upd = (np.eye(12) - K @ H) @ P

        return x_hat_upd, P_upd
