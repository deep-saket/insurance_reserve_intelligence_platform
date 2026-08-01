"""Optimization plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "premium": "Premium",
    "sum_assured": "Sum Assured",
    "interest_rate": "Interest Rate",
    "scenario_interest_rate": "Interest Rate",
    "reserve": "Reserve",
    "model_reserve": "Raw Model Reserve",
    "profit": "Profit",
    "objective": "Objective",
    "capital": "Capital",
}


def plot_metric_sweep(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot one optimization metric against one decision variable."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if frame.empty or x_column not in frame or y_column not in frame:
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame[x_column], frame[y_column], color="#004c6d", linewidth=2.0)
    axis.set_title(title)
    axis.set_xlabel(_label(x_column))
    axis.set_ylabel(_label(y_column))
    axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_convergence(
    history: pd.DataFrame,
    y_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot best-so-far convergence using objective/profit as the progress score."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if history.empty or y_column not in history:
        return

    frame = history.reset_index(drop=True).copy()

    # Use business objective/profit to decide which iteration is "best".
    if "objective" in frame:
        score = frame["objective"].astype(float).to_numpy()
    elif "profit" in frame:
        score = frame["profit"].astype(float).to_numpy()
    else:
        score = frame[y_column].astype(float).to_numpy()

    # Capital is a minimization-style objective; most others maximize.
    minimize = "capital" in str(path).lower() and y_column == "objective"

    best_indices = []
    best_idx = 0
    best_score = np.inf if minimize else -np.inf

    for idx, value in enumerate(score):
        if (minimize and value <= best_score) or (not minimize and value >= best_score):
            best_score = value
            best_idx = idx
        best_indices.append(best_idx)

    if y_column in {"objective", "profit"}:
        if minimize:
            y = np.minimum.accumulate(frame[y_column].astype(float).to_numpy())
        else:
            y = np.maximum.accumulate(frame[y_column].astype(float).to_numpy())
    else:
        y = frame.iloc[best_indices][y_column].astype(float).to_numpy()

    x = np.arange(len(y))

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, y, color="#7a3e00", linewidth=2.5)
    axis.set_title(title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel(_label(y_column))
    axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_standard_convergence_set(
    history: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> list[Path]:
    """Create objective, premium, reserve, and profit convergence charts."""

    output = Path(output_dir)
    created: list[Path] = []

    for column, title in (
        ("objective", "Optimization Convergence"),
        ("premium", "Premium Convergence"),
        ("reserve", "Reserve Convergence"),
        ("profit", "Profit Convergence"),
    ):
        path = output / f"{prefix}_{column}_convergence.png"

        plot_convergence(
            history,
            column,
            path,
            title,
        )

        if path.exists():
            created.append(path)

    return created


def _score_column(history: pd.DataFrame) -> str | None:
    """Choose the column that defines best-so-far optimizer progress."""

    if "profit" in history:
        return "profit"
    if "objective" in history:
        return "objective"
    return None


def _label(column: str) -> str:
    """Return a human-readable axis label."""

    return LABELS.get(column, column.replace("_", " ").title())