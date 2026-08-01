"""Configuration management utilities.

Created: 2026-05-31
Purpose: Define typed experiment configuration objects and YAML loading helpers.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
import re
from typing import Any, Dict, Type, TypeVar, cast, get_type_hints

import yaml


T = TypeVar("T")


def _default_loss_terms() -> dict[str, "LossTermConfig"]:
    """Create the default experiment-friendly loss menu.

    Returns:
        dict[str, LossTermConfig]: Default named loss-term configuration.

    Business Interpretation:
        These defaults define a practical starting point for reserve modeling
        while still allowing analysts to move between PINN, KINN, hybrid, and
        supervised experiments by editing only YAML.
    """

    return {
        "data_loss": LossTermConfig(enabled=True, weight=1.0),
        "pde_loss": LossTermConfig(enabled=True, weight=1.0),
        "boundary_loss": LossTermConfig(enabled=True, weight=10.0),
        "mortality_monotonicity_loss": LossTermConfig(enabled=False, weight=0.2),
        "age_monotonicity_loss": LossTermConfig(enabled=False, weight=0.2),
        "interest_rate_monotonicity_loss": LossTermConfig(enabled=False, weight=0.2),
        "interest_rate_scenario_loss": LossTermConfig(enabled=False, weight=1.0),
        "interest_rate_peak_loss": LossTermConfig(enabled=False, weight=1.0),
        "sum_assured_monotonicity_loss": LossTermConfig(enabled=True, weight=1.0),
        "solvency_loss": LossTermConfig(enabled=True, weight=1.0),
        "reserve_ceiling_loss": LossTermConfig(enabled=True, weight=1.0),
        "smoothness_loss": LossTermConfig(enabled=False, weight=0.05),
        "portfolio_consistency_loss": LossTermConfig(enabled=False, weight=0.5),
        "l2_regularization_loss": LossTermConfig(enabled=True, weight=1e-6),
    }


def _coerce_dataclass(cls: Type[T], payload: Dict[str, Any]) -> T:
    """Recursively coerce a dictionary into a dataclass instance.

    Args:
        cls: Target dataclass type.
        payload: Raw dictionary payload.

    Returns:
        T: Instantiated dataclass of type ``cls``.

    Business Interpretation:
        This lets research and production settings be controlled from YAML rather
        than hard-coded assumptions.
    """

    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for item in fields(cls):
        if item.name in payload:
            value = payload[item.name]
        elif item.default is not MISSING:
            value = item.default
        elif item.default_factory is not MISSING:
            value = item.default_factory()
        else:
            value = None

        hint = type_hints.get(item.name)
        if hint and is_dataclass(hint) and isinstance(value, dict):
            kwargs[item.name] = _coerce_dataclass(cast(Type[Any], hint), value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


@dataclass(slots=True)
class PathConfig:
    """File-system configuration.

    Attributes:
        artifacts_dir: Root directory for generated outputs.
        checkpoints_dir: Directory for model checkpoints.
        logs_dir: Directory for logs and CSV metrics.
        reports_dir: Directory for analytical reports.
        plots_dir: Directory for exported plots.
        tensorboard_dir: Directory for TensorBoard outputs.
        data_dir: Directory for input data assets.
        run_dir: Run-scoped artifact directory.
    """

    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "artifacts/checkpoints"
    logs_dir: str = "artifacts/logs"
    reports_dir: str = "artifacts/reports"
    plots_dir: str = "artifacts/plots"
    tensorboard_dir: str = "artifacts/tensorboard"
    data_dir: str = "data"
    run_dir: str = "artifacts/default"


@dataclass(slots=True)
class DataConfig:
    """Synthetic and tabular data generation configuration.

    Scientific Context:
        This configuration defines the synthetic portfolio sampling space and the
        granularity of reserve trajectories.

    Business Interpretation:
        These settings describe the training portfolio universe and how broad or
        narrow the modeled insurance book should be.
    """

    train_size: int = 512
    validation_size: int = 128
    test_size: int = 128
    time_steps: int = 40
    batch_size: int = 64
    num_workers: int = 0
    age_min: int = 25
    age_max: int = 70
    term_min: int = 5
    term_max: int = 30
    interest_rate_min: float = 0.01
    interest_rate_max: float = 0.08
    sum_assured_min: float = 50_000.0
    sum_assured_max: float = 1_000_000.0
    mortality_scale: float = 0.0005
    mortality_shape: float = 1.08
    mortality_reference_age: int = 25
    premium_loading: float = 1.10
    max_expiry_age: int = 80
    sum_assured_rounding: float = 50_000.0
    sum_assured_age_decay: float = 0.02
    random_seed: int = 42


@dataclass(slots=True)
class SolverConfig:
    """Actuarial solver configuration.

    Business Interpretation:
        These settings govern how the benchmark actuarial reserve engine is run.
    """

    method: str = "solve_ivp"
    integration_step: float = 0.25
    rtol: float = 1e-6
    atol: float = 1e-8


@dataclass(slots=True)
class ModelConfig:
    """PINN model configuration.

    Business Interpretation:
        These settings control the complexity of the reserve surrogate used for
        inference, sensitivities, and digital twin simulation.
    """

    input_dim: int = 7
    hidden_dim: int = 128
    num_layers: int = 4
    activation: str = "tanh"
    dropout: float = 0.1
    skip_connections: bool = False


@dataclass(slots=True)
class LossTermConfig:
    """Configuration for one named loss term.

    Attributes:
        enabled: Whether the loss is active for the experiment.
        weight: Scalar coefficient applied to the raw loss value.

    Business Interpretation:
        This is the experimenter's switch-and-weight control for a single model
        governance rule, such as physics compliance, monotonicity, or solvency.
    """

    enabled: bool = False
    weight: float | None = None


@dataclass(slots=True)
class LossConfig:
    """Named loss configuration for PINN, KINN, and hybrid experiments.

    Attributes:
        terms: Mapping from loss name to its configuration.

    Business Interpretation:
        This structure lets researchers turn individual actuarial and
        knowledge-informed controls on or off without rewriting Python code.
    """

    terms: dict[str, LossTermConfig] = field(default_factory=_default_loss_terms)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "LossConfig":
        """Build a loss configuration from raw YAML content.

        Args:
            payload: Raw ``losses`` mapping from YAML.

        Returns:
            LossConfig: Parsed named loss configuration.
        """

        default_terms = _default_loss_terms()
        if payload is None:
            return cls(terms=default_terms)

        merged_terms = dict(default_terms)
        for name, value in payload.items():
            if not isinstance(value, dict):
                raise ValueError(
                    "Each configured loss must be a mapping with at least "
                    f"'enabled' and 'weight'. Invalid entry for '{name}': {value!r}"
                )
            merged_terms[name] = _coerce_dataclass(LossTermConfig, value)
        return cls(terms=merged_terms)


@dataclass(slots=True)
class LossSettingsConfig:
    """Settings shared by all configured loss modules.

    Attributes:
        reduction: Reduction used by scalar losses.
        use_adaptive_weights: Whether adaptive reweighting is enabled.

    Business Interpretation:
        These settings control how the platform aggregates penalties across a
        batch and whether weighting is fixed or dynamically adjusted.
    """

    reduction: str = "mean"
    use_adaptive_weights: bool = False


@dataclass(slots=True)
class TrainerConfig:
    """Training loop configuration.

    Business Interpretation:
        These settings control learning stability, checkpoint cadence, and
        operational training behavior.
    """

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    checkpoint_every: int = 5
    resume_from: str | None = None
    device: str = "auto"
    run_name: str = "default"
    tensorboard_enabled: bool = True
    mixed_precision: bool = True


@dataclass(slots=True)
class StressScenarioConfig:
    """Default shock amplitudes used by the stress tester.

    Business Interpretation:
        These are the platform's default adverse assumptions used for scenario and
        capital-sensitivity analysis.
    """

    mortality_shock: float = 0.15
    interest_rate_shock: float = -0.01
    inflation_shock: float = 0.05
    longevity_shock: float = -0.10
    lapse_shock: float = 0.10


@dataclass(slots=True)
class OptimizationConfig:
    """Optimization defaults.

    Business Interpretation:
        These settings govern decision-search workflows such as premium or target
        reserve optimization.
    """

    learning_rate: float = 0.05
    steps: int = 150
    reserve_tolerance: float = 1e-4
    solvency_threshold: float = 0.0


@dataclass(slots=True)
class DigitalTwinConfig:
    """Digital twin simulation configuration.

    Business Interpretation:
        These settings define the time horizon and standard scenario catalogue for
        the liability digital twin.
    """

    forecast_horizon: int = 30
    scenario_steps: int = 12
    regime_names: list[str] = field(
        default_factory=lambda: ["base", "soft_recession", "inflationary", "mortality_crisis"]
    )


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level platform configuration.

    Attributes:
        project_name: Human-readable project identifier.
        experiment_name: Named experiment configuration.
        seed: Global random seed.
        paths: File-system paths.
        data: Data-generation settings.
        solver: Actuarial solver settings.
        model: Neural model settings.
        losses: Named loss-term settings.
        loss_settings: Shared reduction and weighting settings.
        trainer: Training-loop settings.
        stress: Stress-scenario settings.
        optimization: Optimization settings.
        digital_twin: Digital twin settings.

    Business Interpretation:
        This object is the contract that ties together actuarial assumptions,
        machine-learning settings, and operational outputs for one experiment.
    """

    project_name: str = "src"
    experiment_name: str = "actuary_twin_pinn"
    seed: int = 42
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    loss_settings: LossSettingsConfig = field(default_factory=LossSettingsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    stress: StressScenarioConfig = field(default_factory=StressScenarioConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    digital_twin: DigitalTwinConfig = field(default_factory=DigitalTwinConfig)


class ConfigLoader:
    """Load YAML configuration into typed dataclasses.

    Business Interpretation:
        This provides a controlled, auditable path from analyst-edited YAML into
        the executable research environment.
    """

    @staticmethod
    def load(path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file.

        Args:
            path: YAML configuration file path.

        Returns:
            ExperimentConfig: Typed configuration object.
        """

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        payload = dict(payload)
        losses_payload = payload.pop("losses", None)
        loss_settings_payload = payload.pop("loss_settings", None)
        config = _coerce_dataclass(ExperimentConfig, payload)
        config.losses = LossConfig.from_mapping(losses_payload)
        if loss_settings_payload is not None:
            config.loss_settings = _coerce_dataclass(LossSettingsConfig, loss_settings_payload)
        return config


def ensure_directories(config: ExperimentConfig) -> None:
    """Create artifact directories required by the workflow.

    Args:
        config: Experiment configuration containing path definitions.

    Business Interpretation:
        This ensures reporting, checkpointing, and governance artifacts have a
        stable place to be written during model development.
    """

    run_name = _sanitize_run_name(config.trainer.run_name or config.experiment_name)
    run_dir = Path(config.paths.artifacts_dir) / run_name
    config.paths.run_dir = str(run_dir)
    config.paths.checkpoints_dir = str(run_dir / "checkpoints")
    config.paths.logs_dir = str(run_dir / "logs")
    config.paths.reports_dir = str(run_dir / "reports")
    config.paths.plots_dir = str(run_dir / "plots")
    config.paths.tensorboard_dir = str(run_dir / "tensorboard")

    for item in fields(config.paths):
        Path(getattr(config.paths, item.name)).mkdir(parents=True, exist_ok=True)


def _sanitize_run_name(name: str) -> str:
    """Convert a free-form run name into a filesystem-safe directory name.

    Args:
        name: User-supplied or configured run name.

    Returns:
        str: Sanitized run name safe for artifact directory creation.
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "default"
