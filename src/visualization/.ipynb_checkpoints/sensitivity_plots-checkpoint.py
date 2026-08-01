"""Sensitivity and elasticity visualisation for reserve models.

Produces two separate reports:
  1. sensitivity_report.png  — raw dV/dx in real £ units (different units per variable)
  2. elasticity_report.png   — elasticity E = (dV/dx)·(x/V), dimensionless and comparable

Revised: 2026-06-15
- Premium x_typical now uses premium_ratio (e.g. 0.0032) not raw £
  because the separated sensitivity pipeline perturbs premium_ratio directly
  with all other features held fixed.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── reference values matching build_reference_feature_vector() in evaluate.py ─
# These are the x_typical used to compute elasticity: E = (dV/dx) · (x/V)
# Premium uses the ratio (0.0032) not raw £ — consistent with how the
# separated pipeline perturbs it.
_X_TYPICAL = {
    "dV_dr":  0.04,         # interest rate 4%
    "dV_dmu": 0.0015,       # mortality intensity 0.0015/yr
    "dV_dP":  0.0032,       # premium ratio 0.32% of SA  ← ratio, not raw £
    "dV_dS":  500_000.0,    # sum assured £500k
}

_LABELS = {
    "dV_dr":  "dV/dr\n(£ per unit r)",
    "dV_dmu": "dV/dμ\n(£ per unit μ)",
    "dV_dP":  "dV/dP\n(£ per unit P_ratio)",
    "dV_dS":  "dV/dS\n(£ per £1 SA)",
}

_ELABEL = {
    "dV_dr":  "Elasticity w.r.t. r\n(% ΔV per 1% Δr)",
    "dV_dmu": "Elasticity w.r.t. μ\n(% ΔV per 1% Δμ)",
    "dV_dP":  "Elasticity w.r.t. P\n(% ΔV per 1% ΔP_ratio)",
    "dV_dS":  "Elasticity w.r.t. S\n(% ΔV per 1% ΔS)",
}

_EXPECTED_SIGN = {
    "dV_dr":  "neg",
    "dV_dmu": "pos",
    "dV_dP":  "pos",
    "dV_dS":  "pos",
}

_DISPLAY_NAME = {
    "dV_dr":  "Interest\nRate",
    "dV_dmu": "Mortality",
    "dV_dP":  "Premium\nRatio",
    "dV_dS":  "Sum\nAssured",
}


def _sign_color(mean_val: float, expected: str) -> str:
    if abs(mean_val) < 1e-10:
        return "#888888"
    correct = (expected == "pos" and mean_val > 0) or (expected == "neg" and mean_val < 0)
    return "#2ecc71" if correct else "#e74c3c"


def plot_sensitivities(report: pd.DataFrame, output_path: str) -> None:
    """Raw dV/dx — four subplots, different units, sign-checked."""
    cols = [c for c in _LABELS if c in report.columns]
    n    = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Reserve Sensitivities  (raw dV/dx — DIFFERENT UNITS, not directly comparable)\n"
        "Each bar: ONE feature perturbed, all others held fixed at reference values",
        fontsize=10, fontweight="bold", color="#555555", y=1.03,
    )

    for ax, col in zip(axes, cols):
    
        value = float(report[col].iloc[0])
    
        color = _sign_color(
            value,
            _EXPECTED_SIGN[col],
        )
    
        expected = _EXPECTED_SIGN[col]
    
        ax.bar(
            0,
            value,
            width=0.45,
            color=color,
            alpha=0.75,
        )
    
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    
        ax.text(
            0,
            value,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    
        ax.set_title(
            _LABELS[col],
            fontsize=10,
            fontweight="bold",
        )
    
        ax.set_xticks([])
    
        ax.set_xlabel(
            f"Expected {'+' if expected=='pos' else '-'}",
            fontsize=8,
        )

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
    report: pd.DataFrame,
    output_path: str,
    reserve_reference: float,
    premium_ratio_typical: float,
) -> pd.DataFrame:
    """All four elasticities on one comparable plot.

    Elasticity = (dV/dx) × (x_typical / V_mean)
    Interpretation: 1% change in x → Elasticity% change in reserve

    Premium x_typical = premium_ratio_mean (the ratio, not raw £)
    This is consistent with how the separated pipeline perturbs premium.
    """
    cols = [c for c in _X_TYPICAL if c in report.columns]

    # ── per-policy elasticities ────────────────────────────────────────────
    elas_data = {}
    for col in cols:
        x_typ = (
            premium_ratio_typical
            if col == "dV_dP"
            else _X_TYPICAL[col]
        )
        elas_data[col] = report[col].values * x_typ / max(abs(reserve_reference), 1.0)

    # ── summary ───────────────────────────────────────────────────────────
    summary_rows = []
    means, stds, colors, labels = [], [], [], []

    for col in cols:
        e_vals   = elas_data[col]
        mean_e   = float(e_vals.mean())
        std_e    = float(e_vals.std())
        expected = _EXPECTED_SIGN[col]
        sign_pct = float((e_vals > 0).mean() * 100) if expected == "pos" \
                   else float((e_vals < 0).mean() * 100)

        means.append(mean_e)
        stds.append(std_e)
        colors.append(_sign_color(mean_e, expected))
        labels.append(_DISPLAY_NAME[col])

        x_typ = premium_ratio_typical if col == "dV_dP" else _X_TYPICAL[col]
        summary_rows.append({
            "Variable":        col,
            "x_typical":       x_typ,
            "Elasticity_mean": mean_e,
            "Elasticity_std":  std_e,
            "Sign_correct_%":  sign_pct,
            "Interpretation":  f"1% ↑ in {_DISPLAY_NAME[col].replace(chr(10),' ')} → {mean_e*100:+.3f}% ΔV",
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\nElasticity Summary  (true partial derivatives, all directly comparable):")
    for _, row in summary_df.iterrows():
        marker = "✓" if row["Sign_correct_%"] > 70 else "✗"
        print(f"  {marker}  {row['Variable']:10s}  {row['Interpretation']}")

    # ── sort by absolute magnitude for readability ─────────────────────────
    order  = np.argsort(np.abs(means))[::-1]
    means  = np.array(means)[order]
    stds   = np.array(stds)[order]
    colors = np.array(colors)[order]
    labels = np.array(labels)[order]

    # ── single combined plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, color=colors, alpha=0.75, width=0.6,
                  edgecolor="black", linewidth=1)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="black", capsize=6, linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Elasticity  (ΔV% / Δx%)", fontsize=12)
    ax.set_title(
        "Reserve Elasticities — True Partial Derivatives\n"
        "(dimensionless, all directly comparable)\n"
        "Each bar: 1% change in ONE variable, all others held fixed",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.25, axis="y")

    # Value labels
    y_range = max(abs(means.max()), abs(means.min())) if len(means) else 1
    for i, (val, std) in enumerate(zip(means, stds)):
        offset = y_range * 0.04
        va = "bottom" if val >= 0 else "top"
        ypos = val + (offset if val >= 0 else -offset) + (std if val >= 0 else -std)
        ax.text(i, ypos, f"{val*100:+.2f}%",
                ha="center", va=va, fontsize=11, fontweight="bold")

    # x_typical footnote
    x_notes = []
    for col in [cols[i] for i in order]:
        x_typ = premium_ratio_typical if col == "dV_dP" else _X_TYPICAL[col]
        x_notes.append(f"x({_DISPLAY_NAME[col].replace(chr(10),' ')}) = {x_typ:g}")
    fig.text(0.5, -0.04, "  |  ".join(x_notes),
             ha="center", fontsize=8, color="#888", style="italic")

    green_p = mpatches.Patch(color="#2ecc71", alpha=0.8, label="Sign correct ✓")
    red_p   = mpatches.Patch(color="#e74c3c", alpha=0.8, label="Sign inverted ✗")
    ax.legend(handles=[green_p, red_p], fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  elasticity plot  → {output_path}")

    return summary_df