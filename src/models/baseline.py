from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseModel


class MajorityBaseline(BaseModel):
    """Simple baseline that always predicts the most frequent class.

    This is intentionally tiny: it demonstrates OOP usage and integrates
    with `UCIHARDatasetLoader` in runners/tests.
    """

    def __init__(self) -> None:
        super().__init__(name="majority_baseline")
        self._majority: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        vals, counts = np.unique(y, return_counts=True)
        self._majority = int(vals[np.argmax(counts)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._majority is None:
            raise RuntimeError("Model not fitted yet")
        n = X.shape[0]
        return np.full(n, self._majority, dtype=np.int64)
