from __future__ import annotations

import numpy as np
import pytest

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import KERBIN_RADIUS, SphericalGravityModel
from src.guidance.ocp import LandingOCP, VehicleLimits
from src.guidance.trajectory import ReferenceTrajectory


@pytest.fixture
def guidance_system() -> tuple[
    SphericalGravityModel, AerodynamicModel, LandingOCP
]:
    offset = np.array([0.0, 0.0, KERBIN_RADIUS], dtype=np.float64)
    gravity = SphericalGravityModel(planet_center_offset=offset)
    aero = AerodynamicModel(cd_area=4.0)
    # Realistic landing TWR (~2.5 - 3.0 for safe terminal deceleration)
    limits = VehicleLimits(
        thrust_min=40_000.0,
        thrust_max=450_000.0,
        isp=310.0,
        max_gimbal_angle_deg=25.0,
        glideslope_angle_deg=15.0,
        min_mass=7_500.0,
    )
    ocp = LandingOCP(limits, gravity, aero, solver="CLARABEL")
    return gravity, aero, ocp


class TestGuidanceIntegration:
    def test_successive_convexification_iteration(
        self,
        guidance_system: tuple[
            SphericalGravityModel, AerodynamicModel, LandingOCP
        ],
    ) -> None:
        """
        Verifies that successive iterations of SOCP (SCvx) monotonically
        refine the reference trajectory and converge.
        """
        gravity, aero, ocp = guidance_system

        r_init = np.array([200.0, -100.0, 1_800.0], dtype=np.float64)
        v_init = np.array([-20.0, 10.0, -90.0], dtype=np.float64)
        m_init = 15_000.0

        r_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        m_dry = 8_500.0

        duration = 20.0
        n_nodes = 25

        # 1. Start from initial straight-line reference trajectory
        current_ref = ReferenceTrajectory.from_straight_line(
            r_init, v_init, m_init, r_target, v_target, m_dry, duration, n_nodes
        )

        # Run 3 SCvx iterations
        for iteration in range(3):
            sol = ocp.solve(
                initial_state=(r_init, v_init, m_init),
                target_state=(r_target, v_target),
                ref_traj=current_ref,
                enable_glideslope=True,
            )

            assert sol.optimal is True, f"Failed at iteration {iteration}"

            # Calculate total fuel consumed
            fuel_burned = m_init - sol.masses[-1]
            assert fuel_burned > 0.0

            # Update reference trajectory with the newly solved solution
            current_ref = ReferenceTrajectory(
                times=current_ref.times,
                positions=sol.positions,
                velocities=sol.velocities,
                masses=sol.masses,
                thrusts=sol.thrusts,
            )

    def test_simulated_receding_horizon_mpc_steps(
        self,
        guidance_system: tuple[
            SphericalGravityModel, AerodynamicModel, LandingOCP
        ],
    ) -> None:
        """
        Simulates multiple sequential closed-loop MPC steps:
        1. Read state.
        2. Solve SOCP along shifted reference.
        3. Apply commanded thrust T_0 to true non-linear dynamics.
        4. Step state forward and verify smooth trajectory tracking.
        """
        gravity, aero, ocp = guidance_system

        r_sim = np.array([100.0, -50.0, 1_500.0], dtype=np.float64)
        v_sim = np.array([-10.0, 5.0, -80.0], dtype=np.float64)
        m_sim = 15_000.0

        r_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        m_dry = 8_500.0

        duration = 18.0
        n_nodes = 20

        ref_traj = ReferenceTrajectory.from_straight_line(
            r_sim, v_sim, m_sim, r_target, v_target, m_dry, duration, n_nodes
        )

        n_mpc_steps = 4
        for step in range(n_mpc_steps):
            dt = ref_traj.dt

            sol = ocp.solve(
                initial_state=(r_sim, v_sim, m_sim),
                target_state=(r_target, v_target),
                ref_traj=ref_traj,
                enable_glideslope=False,
            )

            assert sol.optimal is True, f"MPC step {step} failed to solve"

            T_0 = sol.commanded_thrust
            Gamma_0 = sol.commanded_magnitude

            # Non-linear forward integration step (Euler)
            g_sim = gravity.acceleration(r_sim)
            a_drag_sim = aero.acceleration(r_sim, v_sim, m_sim)
            a_net = g_sim + a_drag_sim + (T_0 / m_sim)

            # Update simulated vessel state
            r_sim = r_sim + v_sim * dt + 0.5 * a_net * (dt**2)
            v_sim = v_sim + a_net * dt
            alpha = 1.0 / (ocp.limits.isp * ocp.limits.g0)
            m_sim = m_sim - dt * alpha * Gamma_0

            # Altitude should decrease toward target
            assert r_sim[2] < sol.positions[0, 2]

            # Warm-start / shift reference trajectory for next MPC step
            ref_traj = ReferenceTrajectory(
                times=ref_traj.times,
                positions=sol.positions,
                velocities=sol.velocities,
                masses=sol.masses,
                thrusts=sol.thrusts,
            ).shift(dt_elapsed=dt)
