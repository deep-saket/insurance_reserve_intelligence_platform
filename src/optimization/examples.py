"""Optimization examples for business demonstrations.

Created: 2026-07-02
Purpose: Demonstrate pricing, capital, product design, and portfolio use cases.
"""

from __future__ import annotations

from src.optimization.optimization_runner import OptimizationRunner


def run_demo(output_dir: str = "artifacts/optimization") -> None:
    """Run all optimization examples and print a manager-friendly summary."""

    runner = OptimizationRunner.from_config_path(output_dir=output_dir)
    results = runner.run(mode="all")
    runner.print_before_after(results)


if __name__ == "__main__":
    run_demo()
