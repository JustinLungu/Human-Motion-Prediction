"""Persistence baseline for next-step IMU prediction.

Simple model: predict z_{t+T} = z_t (use the last timestep of the window).
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel


class PersistenceBaseline(BaseModel):
    """Predicts next-step IMU as the last step in the window.
    
    For each sample window (T, 6), output is simply window[-1, :].
    This should beat random noise but lose to learned models.
    """

    def __init__(self) -> None:
        super().__init__(name="persistence_baseline")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No training needed for persistence baseline."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the last timestep of each input window.
        
        Args:
            X: shape (N, T, 6) where T is the window length (can be any value)
        
        Returns:
            shape (N, 6) - last step of each window
        """
        if X.ndim != 3 or X.shape[2] != 6:
            raise ValueError(f"Expected shape (N, T, 6), got {X.shape}")
        
        return X[:, -1, :].astype(np.float32)  # (N, 6)
