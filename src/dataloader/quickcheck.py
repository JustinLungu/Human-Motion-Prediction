from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, output_dir: str = os.path.join("results", "plots")) -> None:
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        self._activity_names = {
            1: "WALKING",
            2: "WALKING_UPSTAIRS",
            3: "WALKING_DOWNSTAIRS",
            4: "SITTING",
            5: "STANDING",
            6: "LAYING",
        }

        self._channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

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

        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 14), sharex=True)

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
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


