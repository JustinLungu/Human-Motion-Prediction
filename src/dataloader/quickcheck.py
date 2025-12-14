from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml

from dataset import UCIHARDatasetLoader


@dataclass(frozen=True)
class QuickcheckResult:
    output_path: str
    train_shape: tuple
    label_counts: Dict[int, int]


class QuickChecker:
    """
    Creates a simple proof-of-life plot:
      - one sample window per activity class (1..6)
      - plots 6 channels over the 128 time steps
      - saves to results/plots/sample_signals.png
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self._cfg = self._load_config(config_path)
        
        # Load paths and visualization settings from config
        paths = self._cfg.get("paths", {})
        self._output_dir = paths.get("plot_output_dir", os.path.join("results", "plots"))
        os.makedirs(self._output_dir, exist_ok=True)

        # Load activity names and channel names from config
        quickcheck_cfg = self._cfg.get("quickcheck", {})
        self._activity_names = quickcheck_cfg.get("activities", {})
        
        # Channel names derived from dataset config
        dataset_cfg = self._cfg.get("dataset", {})
        channels = dataset_cfg.get("channels", [])
        self._channel_names = [ch[1] for ch in channels]  # Use aliases
        
        # Plot configuration
        self._plot_cfg = quickcheck_cfg.get("plot", {})

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def run(self) -> QuickcheckResult:
        loader = UCIHARDatasetLoader()
        split = loader.load_split("train")

        X, y = split.X, split.y

        # Counts for quick sanity
        label_counts = {k: int(np.sum(y == k)) for k in sorted(self._activity_names.keys())}

        # Pick one sample index per class
        indices = self._pick_one_per_class(y)

        # Generate plot
        out_path = os.path.join(self._output_dir, "sample_signals.png")
        self._plot_samples(X=X, y=y, indices=indices, out_path=out_path)

        print("Quickcheck completed.")
        print(f"  Loader: {loader.describe()}")
        print(f"  Train X shape: {X.shape}")
        print(f"  Saved plot: {out_path}")
        print(f"  Label counts: {label_counts}")

        return QuickcheckResult(
            output_path=out_path,
            train_shape=X.shape,
            label_counts=label_counts,
        )

    def _pick_one_per_class(self, y: np.ndarray) -> Dict[int, int]:
        """
        Returns dict {class_id: sample_index}.
        """
        indices: Dict[int, int] = {}
        for c in sorted(self._activity_names.keys()):
            hits = np.where(y == c)[0]
            if len(hits) == 0:
                raise ValueError(f"No samples found for class {c}")
            indices[c] = int(hits[0])
        return indices

    def _plot_samples(self, X: np.ndarray, y: np.ndarray, indices: Dict[int, int], out_path: str) -> None:
        """
        6 rows (one per activity). Each row plots 6 channels.
        """
        t = np.arange(X.shape[1])  # 0..127

        # Get plot configuration
        figsize = self._plot_cfg.get("figsize", [12, 14])
        dpi = self._plot_cfg.get("dpi", 200)

        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=tuple(figsize), sharex=True)

        for row_idx, class_id in enumerate(sorted(indices.keys())):
            ax = axes[row_idx]
            i = indices[class_id]
            window = X[i]  # (128, 6)

            for ch in range(window.shape[1]):
                ax.plot(t, window[:, ch], label=self._channel_names[ch])

            ax.set_title(f"Activity {class_id}: {self._activity_names[class_id]}  (sample index {i})")
            ax.grid(True)

            # Keep legend readable: show legend only on first subplot
            if row_idx == 0:
                ax.legend(loc="upper right", ncol=3, fontsize=9)

        axes[-1].set_xlabel("Time step (0..127)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


