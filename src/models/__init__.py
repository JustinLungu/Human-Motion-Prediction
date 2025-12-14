"""Models package for experiments.

This package provides a small OOP scaffold for models that consume
the `UCIHARDatasetLoader` output. Start here when adding EKF/PF/RNN
implementations.
"""

from .base import BaseModel
from .baseline import MajorityBaseline

__all__ = ["BaseModel", "MajorityBaseline"]
