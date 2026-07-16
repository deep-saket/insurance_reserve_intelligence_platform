"""Reserve plotting utilities.

Created: 2026-05-31
Purpose: Visualize reserve trajectories for analytical review and reporting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_reserve_trajectory(frame: pd.DataFrame, output_path: str) -> None:
    """Plot one or more reserve trajectories over time.

    Args:
        frame: Data frame containing a ``time`` column and one or more reserve
            series columns.
        output_path: Destination path for the plot image.

    Business Interpretation:
        This chart shows how liabilities build up and run off over the policy
        lifecycle, which is the most intuitive reserve view for stakeholders.
    """

    fig, axis = plt.subplots(figsize=(8, 5))
    series_columns = [column for column in frame.columns if column != "time"]
    palette = ["#004c6d", "#e07a5f", "#2a9d8f", "#6c757d"]
    for index, column in enumerate(series_columns):
        linestyle = "--" if index else "-"
        axis.plot(
            frame["time"],
            frame[column],
            color=palette[index % len(palette)],
            linewidth=2.0,
            linestyle=linestyle,
            label=column.replace("_", " ").title(),
        )
    axis.set_title("Reserve Trajectory")
    axis.set_xlabel("Time")
    axis.set_ylabel("Reserve")
    axis.grid(alpha=0.25)
    if len(series_columns) > 1:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
