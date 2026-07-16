"""Loss registry for config-driven experiments.

Created: 2026-06-03
Purpose: Map YAML loss names to concrete PINN and KINN loss classes.
"""

from __future__ import annotations

from src.losses.age_monotonicity_loss import AgeMonotonicityLoss
from src.losses.boundary_loss import BoundaryLoss
from src.losses.data_loss import DataLoss
from src.losses.interest_rate_monotonicity_loss import InterestRateMonotonicityLoss
from src.losses.interest_rate_scenario_loss import InterestRateScenarioLoss
from src.losses.interest_rate_peak_loss import InterestRatePeakLoss
from src.losses.l2_regularization_loss import L2RegularizationLoss
from src.losses.mortality_monotonicity_loss import MortalityMonotonicityLoss
from src.losses.pde_loss import PDEResidualLoss
from src.losses.portfolio_consistency_loss import PortfolioConsistencyLoss
from src.losses.reserve_ceiling_loss import ReserveCeilingLoss
from src.losses.smoothness_loss import SmoothnessLoss
from src.losses.solvency_loss import SolvencyLoss
from src.losses.sum_assured_monotonicity_loss import SumAssuredMonotonicityLoss

LOSS_REGISTRY = {
    "data_loss": DataLoss,
    "pde_loss": PDEResidualLoss,
    "boundary_loss": BoundaryLoss,
    "mortality_monotonicity_loss": MortalityMonotonicityLoss,
    "age_monotonicity_loss": AgeMonotonicityLoss,
    "interest_rate_monotonicity_loss": InterestRateMonotonicityLoss,
    "interest_rate_scenario_loss": InterestRateScenarioLoss,
    "interest_rate_peak_loss": InterestRatePeakLoss,
    "solvency_loss": SolvencyLoss,
    "reserve_ceiling_loss": ReserveCeilingLoss,
    "smoothness_loss": SmoothnessLoss,
    "portfolio_consistency_loss": PortfolioConsistencyLoss,
    "l2_regularization_loss": L2RegularizationLoss,
    "sum_assured_monotonicity_loss": SumAssuredMonotonicityLoss,
}
