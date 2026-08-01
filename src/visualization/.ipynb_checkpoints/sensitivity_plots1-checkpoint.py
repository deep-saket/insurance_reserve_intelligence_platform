"""Sensitivity and elasticity visualisation for reserve models.

Produces two separate reports:
  1. sensitivity_report.png  — raw dV/dx in real £ units (different units per variable)
  2. elasticity_report.png   — elasticity E = (dV/dx)·(x/V), dimensionless and comparable

Created: 2026-06-03  Revised: 2026-06-14
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── typical mid-point values of each input variable ───────────────────────────
# Used to convert dV/dx → elasticity: E = dV/dx · (x_typical / V_mean)
# These should reflect your actual training data ranges
_X_TYPICAL = {
    "dV_dr": 0.04,
    "dV_dmu": 0.002,
    "dV_dS": 250000.0,
}

_LABELS = {
    "dV_dr":  "dV/dr\n(£ per unit r)",
    "dV_dmu": "dV/dμ\n(£ per unit μ)",
    "dV_dP":  "dV/dP\n(£ per £1 prem.)",
    "dV_dS":  "dV/dS\n(£ per £1 SA)",
}

_ELABEL = {
    "dV_dr":  "Elasticity w.r.t. r\n(% ΔV per 1% Δr)",
    "dV_dmu": "Elasticity w.r.t. μ\n(% ΔV per 1% Δμ)",
    "dV_dP":  "Elasticity w.r.t. P\n(% ΔV per 1% ΔP)",
    "dV_dS":  "Elasticity w.r.t. S\n(% ΔV per 1% ΔS)",
}

_EXPECTED_SIGN = {
    "dV_dr":  "neg",   # higher r → lower PV
    "dV_dmu": "pos",   # higher mortality → higher reserve
    "dV_dP":  "pos",   # higher premium → higher reserve
    "dV_dS":  "pos",   # higher SA → higher reserve
}


def _sign_color(mean_val: float, expected: str) -> str:
    """Green if sign matches expectation, red if inverted."""
    if abs(mean_val) < 1e-10:
        return "#888888"
    correct = (expected == "pos" and mean_val > 0) or (expected == "neg" and mean_val < 0)
    return "#2ecc71" if correct else "#e74c3c"


def plot_sensitivities(report: pd.DataFrame, output_path: str) -> None:
    """Plot raw dV/dx sensitivities with units labelled clearly.

    NOTE: bars are in DIFFERENT UNITS and cannot be compared directly.
    See plot_elasticities() for the comparable version.
    """
    cols = [c for c in _LABELS if c in report.columns]
    n    = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Reserve Sensitivities  (raw dV/dx — DIFFERENT UNITS, not directly comparable)\n"
        "See elasticity report for unit-free comparison",
        fontsize=10, fontweight="bold", color="#555555", y=1.03,
    )

    for ax, col in zip(axes, cols):
        vals     = report[col].dropna().values
        mean_val = float(vals.mean())
        p25, p75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        color    = _sign_color(mean_val, _EXPECTED_SIGN[col])
        expected = _EXPECTED_SIGN[col]

        # IQR box
        ax.bar(0, p75 - p25, bottom=p25, width=0.5, color=color, alpha=0.3, linewidth=0)
        # Mean line
        ax.hlines(mean_val, -0.28, 0.28, colors=color, linewidths=2.5, zorder=5)
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5)

        ax.text(0.32, mean_val, f" {mean_val:+.3g}", va="center", ha="left",
                fontsize=8, color=color, fontweight="bold", transform=ax.get_yaxis_transform())

        ax.set_title(_LABELS[col], fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel(
            f"expected: {'> 0' if expected == 'pos' else '< 0'}",
            fontsize=7, color="#555", style="italic"
        )
        ax.set_xlim(-0.6, 0.8)

    green_p = mpatches.Patch(color="#2ecc71", alpha=0.7, label="Sign correct ✓")
    red_p   = mpatches.Patch(color="#e74c3c", alpha=0.7, label="Sign inverted ✗")
    grey_p  = mpatches.Patch(color="#888888", alpha=0.7, label="Near zero")
    fig.legend(handles=[green_p, red_p, grey_p],
               loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  sensitivity plot → {output_path}")


def compute_and_plot_elasticities(
    report,
    output_path,
    v_mean,
    premium_ratio_mean,
) -> pd.DataFrame:

    cols = [
        "dV_dr",
        "dV_dmu",
        "dV_dP",
        "dV_dS",
    ]

    # --------------------------
    # Compute elasticities
    # --------------------------
    elas_df = pd.DataFrame()

    for col in cols:
    
        if col == "dV_dP":
    
            x_typical = premium_ratio_mean
    
        else:
    
            x_typical = _X_TYPICAL[col]
    
        elas_df[col.replace("dV_d", "E_")] = (
            report[col]
            * x_typical
            / max(abs(v_mean), 1.0)
        )
    summary_rows = []

    means = []
    stds = []
    colors = []
    labels = []

    display_names = {
        "dV_dr": "Interest\nRate",
        "dV_dmu": "Mortality",
        "dV_dP": "Premium",
        "dV_dS": "Sum\nAssured",
    }

    for col in cols:

        e_col = col.replace("dV_d", "E_")

        vals = elas_df[e_col].values

        mean_e = float(vals.mean())
        std_e = float(vals.std())

        means.append(mean_e)
        stds.append(std_e)

        labels.append(display_names[col])

        colors.append(_sign_color(mean_e, _EXPECTED_SIGN[col]))

        expected = _EXPECTED_SIGN[col]

        summary_rows.append({
            "Variable": col,
            "Elasticity_mean": mean_e,
            "Elasticity_std": std_e,
            "Sign_correct_%":
                float((vals > 0).mean()*100)
                if expected=="pos"
                else float((vals<0).mean()*100)
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\nElasticity Summary")
    print(summary_df)

    # --------------------------------------------------------
    # SINGLE COMPARISON PLOT
    # --------------------------------------------------------
    order = np.argsort(np.abs(means))[::-1]

    means = np.array(means)[order]
    stds = np.array(stds)[order]
    colors = np.array(colors)[order]
    labels = np.array(labels)[order]
    fig, ax = plt.subplots(figsize=(8,6))

    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        means,
        color=colors,
        alpha=0.75,
        width=0.6,
        edgecolor="black",
        linewidth=1,
    )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="black",
        capsize=6,
        linewidth=2,
    )

    ax.axhline(0, color="black", linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylabel("Elasticity", fontsize=12)

    ax.set_title(
        "Reserve Elasticities\n(Dimensionless - Directly Comparable)",
        fontsize=14,
        fontweight="bold",
    )

    for i, value in enumerate(means):

        offset = 0.03 * max(abs(np.array(means)))

        if value >= 0:
            ypos = value + offset
            va = "bottom"
        else:
            ypos = value - offset
            va = "top"

        ax.text(
            i,
            ypos,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
        )

    green_patch = mpatches.Patch(
        color="#2ecc71",
        label="Expected sign"
    )

    red_patch = mpatches.Patch(
        color="#e74c3c",
        label="Unexpected sign"
    )

    ax.legend(handles=[green_patch, red_patch])

    plt.tight_layout()

    plt.savefig(output_path, dpi=200)

    plt.close()

    print(f"Elasticity plot -> {output_path}")

    return summary_df