from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.visualization.sensitivity_plots import (
    compute_and_plot_elasticities,
    plot_sensitivities,
)


@dataclass(slots=True)
class EvaluationResult:
    mse: float
    mae: float
    rmse: float
    r2: float


_PERTURBATIONS = {
    "scenario_interest_rate": 0.01,
    "mortality": 0.0005,
    "premium_ratio": 0.0003,
    "sum_assured": 5_000.0,
}


class ReserveEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def _denorm(self, z, target_mean, target_std, sum_assured):
        """Convert model z-output to reserve currency units."""

        return (z * target_std + target_mean) * sum_assured

    def _predict_V(self, features, sum_assured, target_mean, target_std):
        """Run model on normalized features and denormalize to reserve currency."""

        with torch.no_grad():
            z = self.model(features)
        return self._denorm(z, target_mean, target_std, sum_assured)

    def _normalise_delta(
        self,
        feat_name: str,
        delta_real: float,
        interest_std: float,
        premium_std: float,
    ) -> float:
        """Convert a real-unit perturbation to normalized-space perturbation."""

        if feat_name == "scenario_interest_rate":
            return delta_real / interest_std
        if feat_name == "premium_ratio":
            return delta_real / premium_std
        return delta_real / FEATURE_SCALES[feat_name]

    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        """Compute regression metrics in reserve currency units."""

        self.model.eval()
        y_true, y_pred = [], []

        for batch in dataloader:
            features = batch["features"].to(self.device)
            target_mean = batch["target_mean"].to(self.device)
            target_std = batch["target_std"].to(self.device)
            sum_assured = batch["sum_assured_scale"].to(self.device)
            target_z = batch["target"].to(self.device)

            with torch.no_grad():
                pred_z = self.model(features)

            y_pred.append(
                self._denorm(pred_z, target_mean, target_std, sum_assured)
                .cpu()
                .numpy()
            )
            y_true.append(
                self._denorm(target_z, target_mean, target_std, sum_assured)
                .cpu()
                .numpy()
            )

        y_true_array = np.vstack(y_true)
        y_pred_array = np.vstack(y_pred)

        mse = float(np.mean((y_true_array - y_pred_array) ** 2))
        mae = float(np.mean(np.abs(y_true_array - y_pred_array)))
        rmse = float(np.sqrt(mse))
        variance = float(np.var(y_true_array))
        r2 = 1.0 - mse / variance if variance > 0 else 0.0

        return EvaluationResult(mse=mse, mae=mae, rmse=rmse, r2=r2)

    def compute_sensitivities(
        self,
        ref_features: torch.Tensor,
        ref_raw_features: torch.Tensor,
        target_mean: float,
        target_std: float,
        interest_mean: float,
        interest_std: float,
        premium_mean: float,
        premium_std: float,
    ) -> tuple[pd.DataFrame, float]:
        """Compute finite-difference sensitivities in real units."""

        del interest_mean, premium_mean

        self.model.eval()

        features = ref_features.to(self.device)
        raw_features = ref_raw_features.to(self.device)
        sum_assured = raw_features[:, FEATURE_INDEX["sum_assured"]].clamp(min=1.0).unsqueeze(1)

        base_reserve = self._predict_V(
            features,
            sum_assured,
            target_mean,
            target_std,
        )
        mean_base_reserve = float(base_reserve.mean().item())

        print(f"\nReference portfolio: N={features.shape[0]}")
        print(f"Mean base reserve: {mean_base_reserve:,.2f}")

        results: dict[str, np.ndarray] = {}

        for feat_name, delta_real in _PERTURBATIONS.items():
            feature_index = FEATURE_INDEX[feat_name]
            delta_norm = self._normalise_delta(
                feat_name,
                delta_real,
                interest_std,
                premium_std,
            )

            features_high = features.clone()
            features_low = features.clone()

            features_high[:, feature_index] += delta_norm
            features_low[:, feature_index] -= delta_norm

            if feat_name == "sum_assured":
                sum_assured_high = (sum_assured + delta_real).clamp(min=1.0)
                sum_assured_low = (sum_assured - delta_real).clamp(min=1.0)
            else:
                sum_assured_high = sum_assured
                sum_assured_low = sum_assured

            reserve_high = self._predict_V(
                features_high,
                sum_assured_high,
                target_mean,
                target_std,
            )
            reserve_low = self._predict_V(
                features_low,
                sum_assured_low,
                target_mean,
                target_std,
            )

            derivative = (
                (reserve_high - reserve_low) / (2.0 * delta_real)
            ).squeeze(1).cpu().numpy()

            results[feat_name] = derivative

            correct = (
                (feat_name == "scenario_interest_rate" and np.mean(derivative) < 0)
                or (feat_name != "scenario_interest_rate" and np.mean(derivative) > 0)
            )
            sign = "ok" if correct else "check"

            print(
                f"  {sign} dV/d({feat_name:22s}) "
                f"delta_real={delta_real:+.6f} "
                f"delta_norm={delta_norm:+.6f} "
                f"mean={np.mean(derivative):+.4f} "
                f"pos={(derivative > 0).mean() * 100:.0f}%"
            )

        dr_real = _PERTURBATIONS["scenario_interest_rate"]
        dr_norm = self._normalise_delta(
            "scenario_interest_rate",
            dr_real,
            interest_std,
            premium_std,
        )

        features_rate_high = features.clone()
        features_rate_low = features.clone()
        features_rate_high[:, FEATURE_INDEX["scenario_interest_rate"]] += dr_norm
        features_rate_low[:, FEATURE_INDEX["scenario_interest_rate"]] -= dr_norm

        reserve_rate_high = self._predict_V(
            features_rate_high,
            sum_assured,
            target_mean,
            target_std,
        )
        reserve_rate_low = self._predict_V(
            features_rate_low,
            sum_assured,
            target_mean,
            target_std,
        )

        second_rate_derivative = (
            (reserve_rate_high - 2.0 * base_reserve + reserve_rate_low)
            / (dr_real ** 2)
        ).squeeze(1).cpu().numpy()

        print("\n" + "=" * 60)
        print("SENSITIVITY SUMMARY")
        print("=" * 60)

        col_map = {
            "scenario_interest_rate": ("dV_dr", "neg"),
            "mortality": ("dV_dmu", "pos"),
            "premium_ratio": ("dV_dP", "pos"),
            "sum_assured": ("dV_dS", "pos"),
        }

        for feat_name, (column_name, expected_sign) in col_map.items():
            values = results[feat_name]
            correct_sign_pct = (
                (values < 0).mean() * 100
                if expected_sign == "neg"
                else (values > 0).mean() * 100
            )
            print(
                f"  {column_name:10s} mean={values.mean():+.4f} "
                f"correct_sign={correct_sign_pct:.0f}%"
            )

        print("=" * 60)

        sensitivity_df = pd.DataFrame(
            {
                "dV_dr": results["scenario_interest_rate"],
                "dV_dmu": results["mortality"],
                "dV_dP": results["premium_ratio"],
                "dV_dS": results["sum_assured"],
                "d2V_dr2": second_rate_derivative,
            }
        )

        return sensitivity_df, mean_base_reserve

    def generate_sensitivity_report(
        self,
        ref_features: torch.Tensor,
        output_path: str,
        ref_raw_features: torch.Tensor | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        interest_mean: float = 0.0,
        interest_std: float = 1.0,
        premium_mean: float = 0.0,
        premium_std: float = 1.0,
    ) -> pd.DataFrame:
        """Run sensitivity analysis and write CSV plus plots."""

        if ref_raw_features is None:
            raise ValueError(
                "ref_raw_features is required. Pass the raw unnormalized feature tensor."
            )

        sensitivity_df, mean_base_reserve = self.compute_sensitivities(
            ref_features=ref_features,
            ref_raw_features=ref_raw_features,
            target_mean=target_mean,
            target_std=target_std,
            interest_mean=interest_mean,
            interest_std=interest_std,
            premium_mean=premium_mean,
            premium_std=premium_std,
        )

        sensitivity_df.to_csv(output_path, index=False)
        plot_sensitivities(
            sensitivity_df,
            output_path.replace(".csv", ".png"),
        )

        premium_ratio_typical = float(
            ref_raw_features[:, FEATURE_INDEX["premium_ratio"]].mean().item()
        )
        elasticity_path = str(output_path).replace(
            "sensitivity_report",
            "elasticity_report",
        )
        elasticity_png = elasticity_path.replace(".csv", ".png")

        elasticity_summary = compute_and_plot_elasticities(
            sensitivity_df,
            elasticity_png,
            mean_base_reserve,
            premium_ratio_typical,
        )
        elasticity_summary.to_csv(elasticity_path, index=False)

        print(f"Elasticity CSV written to {elasticity_path}")

        return sensitivity_df