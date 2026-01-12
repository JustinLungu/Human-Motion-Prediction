"""
Utility script to generate ONLY the drift plot from an existing metrics.json.

This is useful when you already have results/metrics/metrics.json and just
want to re-create or update results/plots/error_vs_drift.png without
re-running all experiments.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


# Ensure src root is on sys.path (for consistency with other scripts)
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Project root is parent of src
PROJECT_ROOT = SRC_ROOT.parent


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


def extract_drift_config(all_results: Dict[str, Dict]) -> tuple[List[float], List[str]]:
    """
    Infer drift rates and model names from metrics.json.

    Looks for entries with condition == \"drift\" and reads their
    \"drift_rate\" and \"model\" fields.
    """
    drift_rates_set = set()
    models_set = set()
    for _key, info in all_results.items():
        if not isinstance(info, dict):
            continue
        if info.get("condition") != "drift":
            continue
        drift_rate = info.get("drift_rate")
        model_name = info.get("model")
        if drift_rate is not None:
            drift_rates_set.add(float(drift_rate))
        if model_name:
            models_set.add(str(model_name))

    drift_rates = sorted(drift_rates_set)
    models = sorted(models_set)

    if not drift_rates:
        raise ValueError("No drift entries found in metrics.json (condition == 'drift').")
    if not models:
        raise ValueError("No model names found for drift condition in metrics.json.")

    return drift_rates, models


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


def generate_drift_plot(
    metrics_path: str = "results/metrics/metrics.json",
    output_path: str = "results/plots/error_vs_drift.png",
) -> None:
    """
    Generate the drift plot from an existing metrics.json file.

    Parameters
    ----------
    metrics_path : str
        Path to metrics.json (relative to project root or absolute).
    output_path : str
        Path where the drift plot PNG will be saved.
    """
    all_results = load_metrics(metrics_path)
    drift_rates, models = extract_drift_config(all_results)

    out_path = _resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_error_vs_drift(all_results, drift_rates, models, str(out_path))


if __name__ == "__main__":
    # Allow overriding paths via environment variables if desired
    metrics_path = os.environ.get("DRIFT_METRICS_PATH", "results/metrics/metrics.json")
    output_path = os.environ.get("DRIFT_PLOT_PATH", "results/plots/error_vs_drift.png")
    generate_drift_plot(metrics_path=metrics_path, output_path=output_path)


