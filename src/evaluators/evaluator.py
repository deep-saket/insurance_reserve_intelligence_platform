"""Model evaluation and sensitivity analysis."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.visualization.sensitivity_plots import plot_sensitivities


@dataclass(slots=True)
class EvaluationResult:
    mse: float
    mae: float
    rmse: float
    r2: float


class ReserveEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        y_true, y_pred = [], []
        for batch in dataloader:
            features = batch["features"].to(self.device)
            t_mean   = batch["target_mean"].to(self.device)
            t_std    = batch["target_std"].to(self.device)
            S        = batch["sum_assured_scale"].to(self.device)
            target_z = batch["target"].to(self.device)
            with torch.no_grad():
                pred_z = self.model(features)
            # Denormalise: V = (z*σ + μ) * S
            y_pred.append(((pred_z   * t_std + t_mean) * S).cpu().numpy())
            y_true.append(((target_z * t_std + t_mean) * S).cpu().numpy())

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        mse  = float(np.mean((y_true - y_pred) ** 2))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        var  = float(np.var(y_true))
        r2   = 1.0 - mse / var if var > 0 else 0.0
        return EvaluationResult(mse=mse, mae=mae, rmse=rmse, r2=r2)

    def compute_sensitivities(self, features, raw_features, target_mean, target_std):
        self.model.eval()
        inputs = features.to(self.device).clone().detach().requires_grad_(True)
        S = raw_features[:, FEATURE_INDEX["sum_assured"]].to(self.device).clamp(min=1.0)

        pred_z = self.model(inputs)

        grad_z = torch.autograd.grad(
            outputs=pred_z, inputs=inputs,
            grad_outputs=torch.ones_like(pred_z),
            create_graph=True,
        )[0]

        r_idx = FEATURE_INDEX["scenario_interest_rate"]
        grad2_z = torch.autograd.grad(
            outputs=grad_z[:, r_idx:r_idx+1], inputs=inputs,
            grad_outputs=torch.ones_like(grad_z[:, r_idx:r_idx+1]),
            create_graph=False,
        )[0][:, r_idx].detach()

        S = S.detach()

        def to_real(g, scale_x):
            return (g.detach() * target_std * S / scale_x).cpu().numpy()

        def to_real2(g, scale_x):
            return (g.detach() * target_std * S / (scale_x**2)).cpu().numpy()

        return pd.DataFrame({
            "dV_dr":   to_real(
                grad_z[:, FEATURE_INDEX["scenario_interest_rate"]],
                FEATURE_SCALES["scenario_interest_rate"],
            ),
            "dV_dmu":  to_real(grad_z[:, FEATURE_INDEX["mortality"]],     FEATURE_SCALES["mortality"]),
            "dV_dP":   to_real(grad_z[:, FEATURE_INDEX["premium"]],       FEATURE_SCALES["premium"]),
            "dV_dS":   to_real(grad_z[:, FEATURE_INDEX["sum_assured"]],   FEATURE_SCALES["sum_assured"]),
            "d2V_dr2": to_real2(grad2_z, FEATURE_SCALES["scenario_interest_rate"]),
        })

    def generate_sensitivity_report(self, features, output_path,
                                    raw_features=None, target_mean=0.0, target_std=1.0):
        if raw_features is None:
            raw_features = features.clone()
            for name, idx in FEATURE_INDEX.items():
                raw_features[:, idx] *= FEATURE_SCALES[name]
        report = self.compute_sensitivities(features, raw_features, target_mean, target_std)
        report.to_csv(output_path, index=False)
        plot_sensitivities(report, output_path.replace(".csv", ".png"))
        return report
