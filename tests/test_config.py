"""Configuration tests.

Created: 2026-05-31
Purpose: Validate typed configuration loading behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigLoader


def test_config_loader_reads_yaml() -> None:
    """Verify that the YAML configuration loads into the typed config object."""
    config = ConfigLoader.load(Path("configs/config.yaml"))
    assert config.project_name == "src"
    assert config.model.input_dim == 7
    assert config.losses.terms["data_loss"].enabled is True
    assert config.loss_settings.reduction == "mean"


def test_total_loss_raises_for_unknown_loss_name(tmp_path: Path) -> None:
    """Verify that unknown configured loss names fail with a clear error."""

    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(
        """
project_name: src
losses:
  imaginary_loss:
    enabled: true
    weight: 1.0
""".strip(),
        encoding="utf-8",
    )
    config = ConfigLoader.load(config_path)

    from src.losses.total_loss import TotalLoss

    with pytest.raises(ValueError, match="Available loss names"):
        TotalLoss(config.losses, settings=config.loss_settings)


def test_total_loss_raises_for_missing_weight(tmp_path: Path) -> None:
    """Verify that enabled losses without weights fail clearly."""

    config_path = tmp_path / "missing_weight.yaml"
    config_path.write_text(
        """
project_name: src
losses:
  data_loss:
    enabled: true
    weight:
""".strip(),
        encoding="utf-8",
    )
    config = ConfigLoader.load(config_path)

    from src.losses.total_loss import TotalLoss

    with pytest.raises(ValueError, match="Missing weights"):
        TotalLoss(config.losses, settings=config.loss_settings)
