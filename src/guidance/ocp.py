# mypy: disable-error-code="call-arg"
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import numpy.typing as npt

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import SphericalGravityModel
from src.guidance.trajectory import ReferenceTrajectory


@dataclass(frozen=True)
class VehicleLimits:
    thrust_min: float  # Minimum throttle thrust (N)
    thrust_max: float  # Maximum throttle thrust (N)
    isp: float  # Specific impulse (s)
    g0: float = 9.80665  # Standard gravity for Isp (m/s^2)
    max_gimbal_angle_deg: float = 20.0  # Max thrust tilt angle from vertical
    glideslope_angle_deg: float = (
        20.0  # Min glideslope cone from horizontal (pad centered)
    )
    min_mass: float = 1_000.0  # Dry mass lower bound (kg)


@dataclass
class OCPSolution:
    status: str
    optimal: bool
    solve_time_sec: float
    positions: npt.NDArray[np.float64]  # (N+1, 3)
    velocities: npt.NDArray[np.float64]  # (N+1, 3)
    masses: npt.NDArray[np.float64]  # (N+1,)
    thrusts: npt.NDArray[np.float64]  # (N, 3)
    thrust_magnitudes: npt.NDArray[np.float64]  # (N,)

    @property
    def commanded_thrust(self) -> npt.NDArray[np.float64]:
        return self.thrusts[0]

    @property
    def commanded_magnitude(self) -> float:
        return float(self.thrust_magnitudes[0])


class LandingOCP:
    """
    Second-Order Cone Program (SOCP) formulation for minimum-fuel precision landing.
    Uses Lossless Convexification to handle non-convex thrust bounds.
    """

    def __init__(
        self,
        vehicle_limits: VehicleLimits,
        gravity_model: SphericalGravityModel,
        aero_model: AerodynamicModel,
        solver: str = "CLARABEL",
        verbose: bool = False,
    ) -> None:
        self.limits = vehicle_limits
        self.gravity_model = gravity_model
        self.aero_model = aero_model
        self.solver = solver
        self.verbose = verbose

    def solve(
        self,
        initial_state: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], float
        ],
        target_state: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ref_traj: ReferenceTrajectory,
        enable_glideslope: bool = True,
        trust_region_radius: float | None = None,
    ) -> OCPSolution:
        """
        Solves the convex subproblem linearized along ref_traj.

        initial_state: (r_0, v_0, m_0)
        target_state: (r_target, v_target)
        ref_traj: current reference trajectory
        """
        r_init, v_init, m_init = initial_state
        r_target, v_target = target_state

        N = ref_traj.n_nodes
        dt = ref_traj.dt
        alpha = 1.0 / (self.limits.isp * self.limits.g0)  # Mass depletion rate

        # Linearize gravity and drag along reference nodes
        lin_nodes = ref_traj.compute_node_linearizations(
            self.gravity_model, self.aero_model
        )

        # Decision variables
        r = cp.Variable((N + 1, 3), name="position")
        v = cp.Variable((N + 1, 3), name="velocity")
        m = cp.Variable(N + 1, name="mass")
        T = cp.Variable((N, 3), name="thrust")
        Gamma = cp.Variable(N, name="slack_thrust")

        constraints: list[cp.Constraint] = []

        # 1. Boundary conditions
        constraints += [
            r[0] == r_init,
            v[0] == v_init,
            m[0] == m_init,
            r[N] == r_target,
            v[N] == v_target,
        ]

        # 2. Pointing angle & glideslope limits
        cos_theta = np.cos(np.deg2rad(self.limits.max_gimbal_angle_deg))
        tan_gamma = np.tan(np.deg2rad(self.limits.glideslope_angle_deg))

        # 3. Step-by-step dynamics & convex operational constraints
        for k in range(N):
            m_ref_k = ref_traj.masses[k]
            lin_k = lin_nodes[k]

            # Dynamics: acceleration at step k
            # a_net = A_pos * r_k + A_vel * v_k + a_const + (1 / m_ref) * T_k
            a_grav_aero = (
                lin_k.A_pos @ r[k] + lin_k.A_vel @ v[k] + lin_k.a_const
            )
            a_thrust = T[k] / m_ref_k
            a_net = a_grav_aero + a_thrust

            # Trapezoidal integration for position, forward Euler for velocity
            constraints.append(r[k + 1] == r[k] + 0.5 * dt * (v[k] + v[k + 1]))
            constraints.append(v[k + 1] == v[k] + dt * a_net)
            constraints.append(m[k + 1] == m[k] - dt * alpha * Gamma[k])

            # Second-Order Cone constraint: ||T_k||_2 <= Gamma_k
            constraints.append(cp.norm(T[k], 2) <= Gamma[k])

            # Thrust magnitude bounds
            constraints.append(Gamma[k] >= self.limits.thrust_min)
            constraints.append(Gamma[k] <= self.limits.thrust_max)

            # Pointing limit (thrust directed upwards in Z)
            constraints.append(T[k, 2] >= Gamma[k] * cos_theta)

            # Mass lower bound
            constraints.append(m[k + 1] >= self.limits.min_mass)

            # Glideslope cone constraint (r_z >= tan(gamma) * ||r_xy||)
            if enable_glideslope:
                constraints.append(cp.norm(r[k, :2], 2) <= r[k, 2] / tan_gamma)

            # Trust-region constraint (limits deviation from reference)
            if trust_region_radius is not None and trust_region_radius > 0.0:
                constraints.append(
                    cp.norm(r[k] - ref_traj.positions[k], 2)
                    <= trust_region_radius
                )

        # Objective: Maximize terminal mass (minimize fuel) + smooth control variation
        fuel_cost = cp.sum(Gamma) * dt * alpha
        thrust_smoothness = (
            1e-4 * cp.sum([cp.norm(T[k + 1] - T[k], 2) for k in range(N - 1)])
            if N > 1
            else 0.0
        )
        objective = cp.Minimize(fuel_cost + thrust_smoothness)

        problem = cp.Problem(objective, constraints)

        # Solve SOCP
        solver_instance = getattr(cp, self.solver, cp.CLARABEL)
        problem.solve(solver=solver_instance, verbose=self.verbose)

        is_optimal = problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

        if not is_optimal or r.value is None:
            return OCPSolution(
                status=problem.status,
                optimal=False,
                solve_time_sec=float(problem.solver_stats.solve_time or 0.0),
                positions=ref_traj.positions,
                velocities=ref_traj.velocities,
                masses=ref_traj.masses,
                thrusts=ref_traj.thrusts,
                thrust_magnitudes=np.zeros(N),
            )

        return OCPSolution(
            status=problem.status,
            optimal=True,
            solve_time_sec=float(problem.solver_stats.solve_time or 0.0),
            positions=np.array(r.value, dtype=np.float64),
            velocities=np.array(v.value, dtype=np.float64),
            masses=np.array(m.value, dtype=np.float64),
            thrusts=np.array(T.value, dtype=np.float64),
            thrust_magnitudes=np.array(Gamma.value, dtype=np.float64),
        )
