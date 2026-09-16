from __future__ import annotations

import numpy as np
import pytest

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import KERBIN_RADIUS, SphericalGravityModel
from src.guidance.trajectory import LinearizedNodeDynamics, ReferenceTrajectory


@pytest.fixture
def gravity_model() -> SphericalGravityModel:
    offset = np.array([0.0, 0.0, KERBIN_RADIUS], dtype=np.float64)
    return SphericalGravityModel(planet_center_offset=offset)


@pytest.fixture
def aero_model() -> AerodynamicModel:
    return AerodynamicModel(cd_area=5.0)


class TestReferenceTrajectory:
    def test_from_straight_line_initialization(self) -> None:
        r_init = np.array([1_000.0, -500.0, 5_000.0], dtype=np.float64)
        v_init = np.array([-50.0, 20.0, -150.0], dtype=np.float64)
        m_init = 25_000.0

        r_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v_target = np.array([0.0, 0.0, -2.0], dtype=np.float64)
        m_dry = 10_000.0

        duration = 40.0
        n_nodes = 20

        traj = ReferenceTrajectory.from_straight_line(
            r_init=r_init,
            v_init=v_init,
            m_init=m_init,
            r_target=r_target,
            v_target=v_target,
            m_dry=m_dry,
            duration=duration,
            n_nodes=n_nodes,
        )

        assert traj.n_nodes == n_nodes
        assert np.isclose(traj.total_duration, duration)
        assert np.isclose(traj.dt, duration / n_nodes)

        # Shapes
        assert traj.times.shape == (n_nodes + 1,)
        assert traj.positions.shape == (n_nodes + 1, 3)
        assert traj.velocities.shape == (n_nodes + 1, 3)
        assert traj.masses.shape == (n_nodes + 1,)
        assert traj.thrusts.shape == (n_nodes, 3)

        # Initial state checks
        assert np.allclose(traj.positions[0], r_init)
        assert np.allclose(traj.velocities[0], v_init)
        assert np.isclose(traj.masses[0], m_init)

        # Final state checks
        assert np.allclose(traj.positions[-1], r_target)
        assert np.allclose(traj.velocities[-1], v_target)
        assert np.isclose(traj.masses[-1], m_dry)

    def test_invalid_parameters_raise(self) -> None:
        r = np.zeros(3)
        v = np.zeros(3)

        with pytest.raises(AssertionError):
            ReferenceTrajectory.from_straight_line(
                r, v, 1000.0, r, v, 500.0, duration=10.0, n_nodes=0
            )

        with pytest.raises(AssertionError):
            ReferenceTrajectory.from_straight_line(
                r, v, 1000.0, r, v, 500.0, duration=-5.0, n_nodes=10
            )

    def test_compute_node_linearizations(
        self,
        gravity_model: SphericalGravityModel,
        aero_model: AerodynamicModel,
    ) -> None:
        r_init = np.array([2_000.0, 500.0, 10_000.0], dtype=np.float64)
        v_init = np.array([-100.0, 0.0, -250.0], dtype=np.float64)
        m_init = 20_000.0

        r_target = np.zeros(3, dtype=np.float64)
        v_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        m_dry = 9_000.0

        traj = ReferenceTrajectory.from_straight_line(
            r_init,
            v_init,
            m_init,
            r_target,
            v_target,
            m_dry,
            duration=30.0,
            n_nodes=15,
        )

        lin_nodes = traj.compute_node_linearizations(gravity_model, aero_model)
        assert len(lin_nodes) == traj.n_nodes + 1

        for k, lin in enumerate(lin_nodes):
            assert isinstance(lin, LinearizedNodeDynamics)
            assert lin.A_pos.shape == (3, 3)
            assert lin.A_vel.shape == (3, 3)
            assert lin.a_const.shape == (3,)

            # Verify Taylor expansion affine identity at expansion point:
            # A_pos @ r_k + A_vel @ v_k + a_const == a_total(r_k, v_k, m_k)
            r_k = traj.positions[k]
            v_k = traj.velocities[k]
            m_k = traj.masses[k]

            a_expected = gravity_model.acceleration(
                r_k
            ) + aero_model.acceleration(r_k, v_k, m_k)
            a_reconstructed = lin.A_pos @ r_k + lin.A_vel @ v_k + lin.a_const

            assert np.allclose(a_reconstructed, a_expected, atol=1e-10)

    def test_trajectory_shift(self) -> None:
        r_init = np.array([1_000.0, 0.0, 4_000.0], dtype=np.float64)
        v_init = np.array([0.0, 0.0, -100.0], dtype=np.float64)

        traj = ReferenceTrajectory.from_straight_line(
            r_init,
            v_init,
            15_000.0,
            np.zeros(3),
            np.zeros(3),
            8_000.0,
            duration=20.0,
            n_nodes=10,
        )
        # Populate thrusts with synthetic values
        for i in range(traj.n_nodes):
            traj.thrusts[i] = np.array([0.0, 0.0, 50_000.0 + i * 100.0])

        shifted = traj.shift()

        assert shifted.n_nodes == traj.n_nodes
        # Node 0 of shifted should match Node 1 of original
        assert np.allclose(shifted.positions[0], traj.positions[1])
        assert np.allclose(shifted.velocities[0], traj.velocities[1])
        assert np.isclose(shifted.masses[0], traj.masses[1])
        assert np.allclose(shifted.thrusts[0], traj.thrusts[1])

        # Terminal node of shifted should be repeated from previous terminal
        assert np.allclose(shifted.positions[-1], traj.positions[-1])
        assert np.allclose(shifted.velocities[-1], traj.velocities[-1])
        assert np.isclose(shifted.masses[-1], traj.masses[-1])

        # Times should start at 0 and have same dt step
        assert np.isclose(shifted.times[0], 0.0)
        assert np.isclose(shifted.dt, traj.dt)
