"""Models package for experiments.

This package provides a small OOP scaffold for models that consume
the `UCIHARDatasetLoader` output. Start here when adding EKF/PF/RNN
implementations.
"""

from .base import BaseModel
from .baseline import MajorityBaseline
from .persistence import PersistenceBaseline
from .rnn import RNNBaseline
from .evaluate import compute_rmse, evaluate_model

__all__ = [
    "BaseModel",
    "MajorityBaseline",
    "PersistenceBaseline",
    "RNNBaseline",
    "compute_rmse",
    "evaluate_model",
]
