"""Command-line entrypoint for reserve optimization workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.optimization.optimization_runner import OptimizationRunner
from src.utils.config import ConfigLoader


def main() -> None:
    """Run optimization workflows from the command line."""

    parser = argparse.ArgumentParser(description="Run reserve optimization workflows.")
    parser.add_argument(
        "--mode",
        choices=["all", "pricing", "capital", "portfolio", "product", "scenario"],
        default="all",
        help="Optimization workflow to run.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint path.")
    parser.add_argument("--output-dir", default=None, help="Directory for optimization artifacts.")
    args = parser.parse_args()

    config = ConfigLoader.load(Path(args.config))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("artifacts") / config.trainer.run_name / "optimization"

    runner = OptimizationRunner.from_config_path(
        config_path=Path(args.config),
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
    )
    results = runner.run(mode=args.mode)
    runner.print_before_after(results)
    print(f"Optimization artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
