from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import yaml


@dataclass(frozen=True)
class SplitData:
    """
    Container for one split of the UCI HAR inertial time-series.
    """
    X: np.ndarray          # (N, 128, 6)
    y: np.ndarray          # (N,)
    subject: np.ndarray    # (N,)


class UCIHARDatasetLoader:
    """
    Loads the UCI HAR Dataset (raw inertial signals) in a clean, reusable way.

    This loader intentionally uses the time-series inertial signals:
      - body_acc_{x,y,z}
      - body_gyro_{x,y,z}

    It does not use the engineered 561-feature vectors in X_train/X_test.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self._config_path = config_path
        self._cfg = self._load_config(config_path)
        self._dataset_root = self._resolve_dataset_root()
        self._channels = self._load_channels()

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_dataset_root(self) -> str:
        """
        Load dataset root path from config.
        """
        root = self._cfg.get("paths", {}).get("dataset_dir")
        
        if not root:
            raise ValueError(
                "dataset_dir not found in config. "
                "Please check configs/config.yaml"
            )

        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Dataset root not found at: {root}. "
                f"Did you run scripts/get_data.sh?"
            )

        return root

    def _load_channels(self) -> List[Tuple[str, str]]:
        """
        Load channel definitions from config.
        Returns list of tuples [(base_name, alias), ...].
        """
        channels = self._cfg.get("dataset", {}).get("channels", [])
        return [(ch[0], ch[1]) for ch in channels]

    def get_dataset_root(self) -> str:
        return self._dataset_root

    def load_split(self, split: str) -> SplitData:
        """
        Load one of: split='train' or split='test'

        Returns:
          SplitData with X (N, 128, 6), y (N,), subject (N,)
        """
        split = split.strip().lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        inertial_dir = os.path.join(self._dataset_root, split, "Inertial Signals")

        # Load signals and stack to (N, 128, 6)
        signals = []
        for base, _alias in self._channels:
            path = os.path.join(inertial_dir, f"{base}_{split}.txt")
            arr = self._load_signal_file(path)
            signals.append(arr)

        X = np.stack(signals, axis=-1).astype(np.float32)  # (N, 128, 6)

        # Labels and subject ids
        y_path = os.path.join(self._dataset_root, split, f"y_{split}.txt")
        s_path = os.path.join(self._dataset_root, split, f"subject_{split}.txt")

        y = self._load_vector_file(y_path).astype(np.int64)
        subject = self._load_vector_file(s_path).astype(np.int64)

        self._validate_split(X=X, y=y, subject=subject, split=split)

        return SplitData(X=X, y=y, subject=subject)

    @staticmethod
    def _load_signal_file(path: str) -> np.ndarray:
        """
        Loads a (N, 128) inertial signal file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing inertial file: {path}")

        # UCI HAR uses space-separated values with variable spacing.
        arr = np.loadtxt(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        return arr

    @staticmethod
    def _load_vector_file(path: str) -> np.ndarray:
        """
        Loads a (N,) integer vector file like y_train.txt or subject_train.txt.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing vector file: {path}")
        arr = np.loadtxt(path)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D vector in {path}, got shape {arr.shape}")
        return arr

    @staticmethod
    def _validate_split(X: np.ndarray, y: np.ndarray, subject: np.ndarray, split: str) -> None:
        if X.ndim != 3 or X.shape[1] != 128 or X.shape[2] != 6:
            raise ValueError(
                f"{split}: Expected X shape (N, 128, 6), got {X.shape}"
            )

        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError(f"{split}: y length {y.shape[0]} does not match X {n}")
        if subject.shape[0] != n:
            raise ValueError(f"{split}: subject length {subject.shape[0]} does not match X {n}")

        if np.min(y) < 1 or np.max(y) > 6:
            raise ValueError(f"{split}: labels y expected in 1..6, got [{np.min(y)}, {np.max(y)}]")

    def describe(self) -> str:
        """
        Returns a short description string that can be printed in scripts.
        """
        channel_str = ", ".join([alias for _, alias in self._channels])
        return (
            f"UCIHARDatasetLoader(dataset_root='{self._dataset_root}', "
            f"channels=[{channel_str}])"
        )
