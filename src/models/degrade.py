"""Degradation utilities (placed inside `models` so models can import locally).

These functions mirror the standalone `src/degrade.py` implementation but
live inside the `models` package for convenience.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np


def add_gaussian_noise(X: np.ndarray, sigma: Union[float, np.ndarray], seed: Optional[int] = None) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (N, T, C)")

    rng = np.random.default_rng(seed)
    N, T, C = X.shape

    sigma_arr = np.asarray(sigma)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full((C,), float(sigma_arr))
    elif sigma_arr.shape[0] != C:
        raise ValueError("sigma must be scalar or length-C array")

    noise = rng.normal(loc=0.0, scale=1.0, size=(N, T, C)) * sigma_arr.reshape(1, 1, C)
    return (X + noise).astype(X.dtype, copy=True)


def add_bias_drift(X: np.ndarray, drift_rate: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (N, T, C)")

    rng = np.random.default_rng(seed)
    N, T, C = X.shape

    step_sigma = max(abs(drift_rate) * 0.1, 1e-8)
    increments = rng.normal(loc=drift_rate / T, scale=step_sigma, size=(N, T, C))
    drift = np.cumsum(increments, axis=1)

    return (X + drift).astype(X.dtype, copy=True)


def apply_dropout(
    X: np.ndarray,
    p: float = 0.05,
    max_gap: int = 10,
    mode: str = "hold",
    seed: Optional[int] = None,
) -> np.ndarray:
    if mode not in ("hold", "zero"):
        raise ValueError("mode must be 'hold' or 'zero'")

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (N, T, C)")

    rng = np.random.default_rng(seed)
    N, T, C = X.shape
    out = X.copy().astype(X.dtype, copy=True)

    for n in range(N):
        for c in range(C):
            t = 0
            while t < T:
                if rng.random() < p:
                    gap_len = int(rng.integers(1, max_gap + 1))
                    end = min(t + gap_len, T)
                    if mode == "zero":
                        out[n, t:end, c] = 0
                    else:  # hold
                        if t == 0:
                            out[n, t:end, c] = 0
                        else:
                            out[n, t:end, c] = out[n, t - 1, c]
                    t = end
                else:
                    t += 1

    return out
