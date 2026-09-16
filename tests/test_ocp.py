from __future__ import annotations

import numpy as np
import pytest

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import KERBIN_RADIUS, SphericalGravityModel
from src.guidance.ocp import LandingOCP, VehicleLimits
from src.guidance.trajectory import ReferenceTrajectory


@pytest.fixture
def gravity_model() -> SphericalGravityModel:
    offset = np.array([0.0, 0.0, KERBIN_RADIUS], dtype=np.float64)
    return SphericalGravityModel(planet_center_offset=offset)


@pytest.fixture
def aero_model() -> AerodynamicModel:
    return AerodynamicModel(cd_area=4.5)


@pytest.fixture
def vehicle_limits() -> VehicleLimits:
    return VehicleLimits(
        thrust_min=20_000.0,  # 20 kN min throttle
        thrust_max=250_000.0,  # 250 kN max throttle
        isp=300.0,
        max_gimbal_angle_deg=25.0,
        glideslope_angle_deg=20.0,
        min_mass=8_000.0,
    )


class TestLandingOCP:
    def test_ocp_solves_feasible_landing(
        self,
        gravity_model: SphericalGravityModel,
        aero_model: AerodynamicModel,
        vehicle_limits: VehicleLimits,
    ) -> None:
        """Verifies SOCP solves to optimality for an incoming booster scenario."""
        ocp = LandingOCP(
            vehicle_limits=vehicle_limits,
            gravity_model=gravity_model,
            aero_model=aero_model,
            solver="CLARABEL",
        )

        r_init = np.array([200.0, -100.0, 1_500.0], dtype=np.float64)
        v_init = np.array([-20.0, 10.0, -120.0], dtype=np.float64)
        m_init = 18_000.0

        r_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        m_dry = 10_000.0

        duration = 18.0
        n_nodes = 20

        ref_traj = ReferenceTrajectory.from_straight_line(
            r_init=r_init,
            v_init=v_init,
            m_init=m_init,
            r_target=r_target,
            v_target=v_target,
            m_dry=m_dry,
            duration=duration,
            n_nodes=n_nodes,
        )

        sol = ocp.solve(
            initial_state=(r_init, v_init, m_init),
            target_state=(r_target, v_target),
            ref_traj=ref_traj,
            enable_glideslope=True,
        )

        assert sol.optimal is True
        assert sol.positions.shape == (n_nodes + 1, 3)
        assert sol.velocities.shape == (n_nodes + 1, 3)
        assert sol.thrusts.shape == (n_nodes, 3)
        assert sol.thrust_magnitudes.shape == (n_nodes,)

        # Boundary condition satisfaction
        assert np.allclose(sol.positions[0], r_init, atol=1e-3)
        assert np.allclose(sol.velocities[0], v_init, atol=1e-3)
        assert np.allclose(sol.positions[-1], r_target, atol=1e-2)
        assert np.allclose(sol.velocities[-1], v_target, atol=1e-2)

        # Final mass must be <= initial mass and >= min_mass
        assert sol.masses[-1] <= m_init
        assert sol.masses[-1] >= vehicle_limits.min_mass

    def test_lossless_convexification_slack_tightness(
        self,
        gravity_model: SphericalGravityModel,
        aero_model: AerodynamicModel,
        vehicle_limits: VehicleLimits,
    ) -> None:
        """Verifies ||T_k||_2 == Gamma_k when thrust is actively decelerating."""
        ocp = LandingOCP(
            vehicle_limits=vehicle_limits,
            gravity_model=gravity_model,
            aero_model=aero_model,
        )

        r_init = np.array([50.0, 0.0, 1_000.0], dtype=np.float64)
        v_init = np.array([0.0, 0.0, -90.0], dtype=np.float64)
        m_init = 16_000.0

        ref_traj = ReferenceTrajectory.from_straight_line(
            r_init=r_init,
            v_init=v_init,
            m_init=m_init,
            r_target=np.zeros(3),
            v_target=np.array([0.0, 0.0, -1.0]),
            m_dry=10_000.0,
            duration=14.0,
            n_nodes=20,
        )

        sol = ocp.solve(
            initial_state=(r_init, v_init, m_init),
            target_state=(np.zeros(3), np.array([0.0, 0.0, -1.0])),
            ref_traj=ref_traj,
            enable_glideslope=False,
        )

        assert sol.optimal is True

        for k in range(ref_traj.n_nodes):
            t_norm = np.linalg.norm(sol.thrusts[k])
            gamma = sol.thrust_magnitudes[k]
            # Second order cone holds: ||T_k|| <= Gamma_k
            assert t_norm <= gamma + 1e-4
            # When burning fuel actively, Gamma should be tight with ||T_k||
            if gamma > vehicle_limits.thrust_min + 1.0:
                assert np.isclose(t_norm, gamma, rtol=1e-3, atol=1.0)

    def test_physical_and_cone_constraints(
        self,
        gravity_model: SphericalGravityModel,
        aero_model: AerodynamicModel,
        vehicle_limits: VehicleLimits,
    ) -> None:
        """Verifies thrust bounds, gimbal cone, and glideslope cone."""
        ocp = LandingOCP(
            vehicle_limits=vehicle_limits,
            gravity_model=gravity_model,
            aero_model=aero_model,
        )

        r_init = np.array([150.0, 100.0, 1_200.0], dtype=np.float64)
        v_init = np.array([-15.0, -10.0, -100.0], dtype=np.float64)
        m_init = 15_000.0

        ref_traj = ReferenceTrajectory.from_straight_line(
            r_init=r_init,
            v_init=v_init,
            m_init=m_init,
            r_target=np.zeros(3),
            v_target=np.array([0.0, 0.0, -0.5]),
            m_dry=10_000.0,
            duration=16.0,
            n_nodes=20,
        )

        sol = ocp.solve(
            initial_state=(r_init, v_init, m_init),
            target_state=(np.zeros(3), np.array([0.0, 0.0, -0.5])),
            ref_traj=ref_traj,
            enable_glideslope=True,
        )

        assert sol.optimal is True

        cos_gimbal = np.cos(np.deg2rad(vehicle_limits.max_gimbal_angle_deg))
        tan_gs = np.tan(np.deg2rad(vehicle_limits.glideslope_angle_deg))

        for k in range(ref_traj.n_nodes):
            gamma = sol.thrust_magnitudes[k]
            # Thrust magnitude bounds
            assert gamma >= vehicle_limits.thrust_min - 1e-3
            assert gamma <= vehicle_limits.thrust_max + 1e-3

            # Pointing gimbal limit (thrust Z >= Gamma * cos(theta))
            assert sol.thrusts[k, 2] >= gamma * cos_gimbal - 1e-3

            # Glideslope cone: ||r_xy|| <= r_z / tan(gamma)
            r_xy = np.linalg.norm(sol.positions[k, :2])
            r_z = sol.positions[k, 2]
            assert r_xy <= (r_z / tan_gs) + 1e-2
