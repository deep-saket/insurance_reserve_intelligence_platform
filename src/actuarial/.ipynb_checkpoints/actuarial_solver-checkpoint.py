"""Classical actuarial reserve solvers.

Created: 2026-05-31
Purpose: Define classical numerical solvers for term-life reserve trajectories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from src.actuarial.policy import Policy

# Global reserve floor applied to every trajectory produced by this module.
# Set to 0.0 for industrial use (non-negativity requirement).
# Set to float("-inf") to disable (pure actuarial mode, allows V(0) < 0).
RESERVE_FLOOR = float("-inf")


@dataclass(slots=True)
class ReserveTrajectory:
    """Numerical reserve solution across a policy term.

    Attributes:
        times: Time grid over the policy term.
        reserves: Reserve values aligned to ``times``.

    Scientific Context:
        The reserve trajectory is a discrete approximation to the continuous reserve
        function ``V(t)`` governed by the Thiele differential equation.

    Business Interpretation:
        This object is the liability path of a policy through time. Actuaries and
        finance teams can read it as "how much money should be held at each point
        in the contract to remain adequately reserved."
    """

    times: np.ndarray
    reserves: np.ndarray


class BaseActuarialSolver(ABC):
    """Abstract base class for classical reserve solvers.

    Scientific Context:
        Classical solvers generate benchmark reserve trajectories by directly
        integrating the governing liability equation rather than approximating it
        with a learned surrogate.

    Business Interpretation:
        This is the trusted actuarial engine used to produce ground-truth reserve
        paths for validation, benchmarking, and model governance.
    """

    @abstractmethod
    def solve(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        """Solve the reserve equation for a policy.

        Args:
            policy: Policy to value.
            num_steps: Number of evaluation steps across the term.

        Returns:
            ReserveTrajectory: Numerical reserve path for the policy.
        """


class ThieleSolver(BaseActuarialSolver):
    """Numerically solve the term-life Thiele reserve equation.

    Scientific Context:
        The solver integrates ``dV/dt = rV + P - Î¼(S - V)`` backward from the
        terminal condition ``V(T)=0``. This is the standard continuous-time
        reserve formulation for a one-state term-life contract under mortality and
        interest assumptions.

    Business Interpretation:
        This class answers the question: "Given product assumptions, what reserve
        profile should the insurer hold over the life of the policy?"
    """

    def __init__(
        self,
        method: str = "solve_ivp",
        integration_step: float = 0.25,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> None:
        """Initialize the actuarial solver.

        Args:
            method: Numerical method name. Supported values are ``solve_ivp`` and ``rk4``.
            integration_step: Maximum integration step size.
            rtol: Relative tolerance for ``solve_ivp``.
            atol: Absolute tolerance for ``solve_ivp``.
        """
        self.method = method
        self.integration_step = integration_step
        self.rtol = rtol
        self.atol = atol

    @staticmethod
    def _rhs(time_point: float, reserve: np.ndarray, policy: Policy) -> np.ndarray:
        """Evaluate the reserve differential equation right-hand side.

        Args:
            time_point: Current elapsed policy time.
            reserve: Current reserve state vector.
            policy: Policy assumptions used in the equation.

        Returns:
            np.ndarray: Single-element derivative vector.

        Scientific Context:
            The derivative combines investment growth, premium inflow, and expected
            mortality outgo net of reserve release.

        Business Interpretation:
            This is the instantaneous liability drift of the policy at a single
            point in time.
        """
        reserve_value = float(reserve[0])
        mortality = policy.mortality_profile.intensity_at(time_point)
        derivative = (
            policy.scenario_interest_rate * reserve_value
            + policy.premium
            - mortality * (policy.sum_assured - reserve_value)
        )
        return np.asarray([derivative], dtype=float)

    def _solve_rk4(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        """Solve the reserve equation using a hand-coded RK4 routine.

        Args:
            policy: Policy to value.
            num_steps: Number of points in the returned trajectory.

        Returns:
            ReserveTrajectory: Backward-solved reserve trajectory.

        Scientific Context:
            Fourth-order Runge-Kutta is a fixed-step explicit integration scheme
            that provides a transparent numerical baseline.

        Business Interpretation:
            This gives a stable reserve path using a simple actuarial numerical
            method that is easy to audit.
        """
        grid = np.linspace(float(policy.term), 0.0, num_steps, dtype=float)
        reserves = np.zeros_like(grid)
        reserves[0] = 0.0

        for index in range(1, len(grid)):
            t_prev = grid[index - 1]
            y_prev = np.asarray([reserves[index - 1]], dtype=float)
            step = grid[index] - grid[index - 1]
            k1 = self._rhs(t_prev, y_prev, policy)
            k2 = self._rhs(t_prev + step / 2.0, y_prev + step * k1 / 2.0, policy)
            k3 = self._rhs(t_prev + step / 2.0, y_prev + step * k2 / 2.0, policy)
            k4 = self._rhs(grid[index], y_prev + step * k3, policy)
            reserves[index] = float(
                y_prev[0] + (step / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            )

        times    = grid[::-1].copy()
        raw_reserves = reserves[::-1]

        if np.isfinite(RESERVE_FLOOR):
            reserves = np.maximum(raw_reserves, RESERVE_FLOOR)
        else:
            reserves = raw_reserves.copy()
        return ReserveTrajectory(times=times, reserves=reserves)

    def _solve_ivp(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        """Solve the reserve equation with SciPy's adaptive integrator.

        Args:
            policy: Policy to value.
            num_steps: Number of points in the returned trajectory.

        Returns:
            ReserveTrajectory: Backward-solved reserve trajectory.

        Raises:
            RuntimeError: If SciPy fails to integrate the reserve equation.

        Scientific Context:
            ``solve_ivp`` adaptively controls local integration error, which is
            useful when reserve dynamics become steep under stressed assumptions.

        Business Interpretation:
            This is the higher-robustness production-style solver used when the
            liability profile needs more numerical stability.
        """
        evaluation_times = np.linspace(float(policy.term), 0.0, num_steps, dtype=float)
        solution = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, policy),
            t_span=(float(policy.term), 0.0),
            y0=np.asarray([0.0], dtype=float),
            t_eval=evaluation_times,
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.integration_step,
        )
        if not solution.success:
            raise RuntimeError(f"Actuarial solver failed: {solution.message}")

        times    = solution.t[::-1].copy()
        if np.isfinite(RESERVE_FLOOR):
            reserves = np.maximum(solution.y[0][::-1], RESERVE_FLOOR)
        else:
            reserves = solution.y[0][::-1].copy()
        return ReserveTrajectory(times=times, reserves=reserves)

    def solve(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        """Solve the reserve trajectory using the configured numerical routine.

        Args:
            policy: Policy to value.
            num_steps: Number of points in the returned trajectory.

        Returns:
            ReserveTrajectory: Reserve path computed by the configured solver,
                with values floored at RESERVE_FLOOR (default 0.0).

        Business Interpretation:
            This method is the main actuarial valuation entry point for generating
            reserve curves used in training, validation, and scenario analysis.
            The floor ensures no negative reserves enter any downstream component â€”
            training data, evaluation, digital twin, stress testing, or optimisation.
        """
        if self.method.lower() == "rk4":
            return self._solve_rk4(policy=policy, num_steps=num_steps)
        return self._solve_ivp(policy=policy, num_steps=num_steps)
