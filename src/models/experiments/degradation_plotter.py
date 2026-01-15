"""
Utility script to generate all degradation plots from an existing metrics.json.

This is useful when you already have results/metrics/metrics.json and just
want to re-create or update plots without re-running all experiments.

Plots generated:
- error_vs_noise.png
- error_vs_dropout.png
- error_vs_drift.png
- filtered_traces.png (optional, requires models and data)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# Ensure src root is on sys.path (for consistency with other scripts)
# File is at src/models/experiments/degradation_plotter.py
# parents[0] = src/models/experiments/
# parents[1] = src/models/
# parents[2] = src/
# parents[3] = project root
SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Project root is parent of src
PROJECT_ROOT = SRC_ROOT.parent

# Desired model order for all plots
MODEL_ORDER = ["persistence", "ekf", "pf", "rnn"]


def _resolve_path(path: str) -> Path:
    """Resolve a potentially relative path from the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_metrics(metrics_path: str = "results/metrics/metrics.json") -> Dict[str, Dict]:
    """Load the metrics JSON produced by experiments.run_experiment."""
    path = _resolve_path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"metrics.json not found at {path}")
    with open(path, "r") as f:
        return json.load(f)


def extract_degradation_config(
    all_results: Dict[str, Dict], condition: str, param_name: str
) -> tuple[List[float], List[str]]:
    """
    Infer parameter values and model names from metrics.json for a given condition.

    Parameters
    ----------
    all_results : Dict[str, Dict]
        The loaded metrics.json data
    condition : str
        The condition name ("noise", "dropout", or "drift")
    param_name : str
        The parameter name ("sigma", "p", or "drift_rate")

    Returns
    -------
    tuple[List[float], List[str]]
        (sorted parameter values, ordered model names)
    """
    param_set = set()
    models_set = set()
    for _key, info in all_results.items():
        if not isinstance(info, dict):
            continue
        if info.get("condition") != condition:
            continue
        param_val = info.get(param_name)
        model_name = info.get("model")
        if param_val is not None:
            param_set.add(float(param_val))
        if model_name:
            models_set.add(str(model_name))

    param_values = sorted(param_set)
    # Order models according to MODEL_ORDER, then add any others not in the list
    models = [m for m in MODEL_ORDER if m in models_set]
    models.extend([m for m in sorted(models_set) if m not in MODEL_ORDER])

    if not param_values:
        raise ValueError(
            f"No {condition} entries found in metrics.json (condition == '{condition}')."
        )
    if not models:
        raise ValueError(
            f"No model names found for {condition} condition in metrics.json."
        )

    return param_values, models


def _plot_error_vs_param(
    all_results: Dict[str, Dict],
    param_values: List[float],
    models: List[str],
    condition: str,
    param_name: str,
    xlabel: str,
    title: str,
    output_path: str,
) -> None:
    """
    Generic function to plot error vs parameter.

    This mirrors the behavior of _plot_error_vs_param in experiments.py
    but is kept local here so this script can run independently.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in models:
        rmse_values = []
        params_used = []
        for param_val in param_values:
            key = f"{model_name}_{condition}_{param_val:.4f}"
            if key in all_results:
                rmse_values.append(all_results[key]["mean_rmse"])
                params_used.append(param_val)
        if rmse_values:
            ax.plot(params_used, rmse_values, marker="o", label=model_name, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Mean RMSE", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_error_vs_noise(
    all_results: Dict[str, Dict],
    noise_sigmas: List[float],
    models: List[str],
    output_path: str,
) -> None:
    """Plot error vs noise sigma for each model."""
    _plot_error_vs_param(
        all_results,
        noise_sigmas,
        models,
        "noise",
        "sigma",
        "Noise Sigma",
        "Error vs Noise Level",
        output_path,
    )


def plot_error_vs_dropout(
    all_results: Dict[str, Dict],
    dropout_ps: List[float],
    models: List[str],
    output_path: str,
) -> None:
    """Plot error vs dropout probability for each model."""
    _plot_error_vs_param(
        all_results,
        dropout_ps,
        models,
        "dropout",
        "p",
        "Dropout Probability (p)",
        "Error vs Dropout Level",
        output_path,
    )


def plot_error_vs_drift(
    all_results: Dict[str, Dict],
    drift_rates: List[float],
    models: List[str],
    output_path: str,
) -> None:
    """Plot error vs drift rate for each model."""
    _plot_error_vs_param(
        all_results,
        drift_rates,
        models,
        "drift",
        "drift_rate",
        "Drift Rate",
        "Error vs Drift Level",
        output_path,
    )


def plot_filtered_trace(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    example_idx: int = 0,
    channel: int = 0,
    output_path: str = "results/plots/filtered_traces.png",
) -> None:
    """
    Plot filtered trace for one example sequence per model.

    Note: This function requires model objects and data, which are not available
    from metrics.json alone. It must be called with models and data passed in.
    """
    seq = X_test[example_idx, :, :]
    true_next = y_test[example_idx, channel]
    full_seq = np.concatenate([seq, y_test[example_idx : example_idx + 1, :]], axis=0)
    time_steps = np.arange(full_seq.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    model_names = MODEL_ORDER
    for idx, model_name in enumerate(model_names):
        if model_name not in models:
            continue
        ax = axes[idx]
        model = models[model_name]
        seq_input = seq.reshape(1, seq.shape[0], seq.shape[1])
        pred_next = model.predict(seq_input)[0, channel]
        ax.plot(
            time_steps[:-1],
            full_seq[:-1, channel],
            "b-",
            label="True",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            time_steps[-1],
            true_next,
            "b*",
            markersize=12,
            label="True (next)",
            alpha=0.7,
        )
        ax.plot(
            time_steps[-1],
            pred_next,
            "r^",
            markersize=12,
            label="Predicted",
            alpha=0.7,
        )
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel(f"Channel {channel}", fontsize=10)
        ax.set_title(f"{model_name.upper()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Filtered Traces - Example {example_idx}, Channel {channel}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def generate_all_plots(
    metrics_path: str = "results/metrics/metrics.json",
    plots_dir: str = "results/plots",
    generate_filtered_trace: bool = False,
    models: Optional[Dict[str, object]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> None:
    """
    Generate all degradation plots from an existing metrics.json file.

    Parameters
    ----------
    metrics_path : str
        Path to metrics.json (relative to project root or absolute).
    plots_dir : str
        Directory where plots will be saved.
    generate_filtered_trace : bool
        Whether to generate filtered_traces.png (requires models and data).
    models : Optional[Dict[str, object]]
        Model objects needed for filtered_trace plot (if generate_filtered_trace=True).
    X_test : Optional[np.ndarray]
        Test input data needed for filtered_trace plot (if generate_filtered_trace=True).
    y_test : Optional[np.ndarray]
        Test target data needed for filtered_trace plot (if generate_filtered_trace=True).
    """
    all_results = load_metrics(metrics_path)

    plots_path = _resolve_path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Generate noise plot
    try:
        noise_sigmas, models_noise = extract_degradation_config(
            all_results, "noise", "sigma"
        )
        plot_error_vs_noise(
            all_results,
            noise_sigmas,
            models_noise,
            str(plots_path / "error_vs_noise.png"),
        )
    except ValueError as e:
        print(f"Skipping noise plot: {e}")

    # Generate dropout plot
    try:
        dropout_ps, models_dropout = extract_degradation_config(
            all_results, "dropout", "p"
        )
        plot_error_vs_dropout(
            all_results,
            dropout_ps,
            models_dropout,
            str(plots_path / "error_vs_dropout.png"),
        )
    except ValueError as e:
        print(f"Skipping dropout plot: {e}")

    # Generate drift plot
    try:
        drift_rates, models_drift = extract_degradation_config(
            all_results, "drift", "drift_rate"
        )
        plot_error_vs_drift(
            all_results,
            drift_rates,
            models_drift,
            str(plots_path / "error_vs_drift.png"),
        )
    except ValueError as e:
        print(f"Skipping drift plot: {e}")

    # Generate filtered trace plot (optional, requires models and data)
    if generate_filtered_trace:
        if models is None or X_test is None or y_test is None:
            print(
                "Warning: Skipping filtered_trace plot - models and data required but not provided."
            )
        else:
            plot_filtered_trace(
                models,
                X_test,
                y_test,
                example_idx=0,
                channel=0,
                output_path=str(plots_path / "filtered_traces.png"),
            )


if __name__ == "__main__":
    # Allow overriding paths via environment variables if desired
    metrics_path = os.environ.get(
        "DEGRADATION_METRICS_PATH", "results/metrics/metrics.json"
    )
    plots_dir = os.environ.get("DEGRADATION_PLOTS_DIR", "results/plots")
    generate_all_plots(metrics_path=metrics_path, plots_dir=plots_dir)

