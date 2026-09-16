from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from src.guidance.models.aerodynamics import (
    KERBIN_ATMOSPHERE_DEPTH,
    KERBIN_SCALE_HEIGHT,
    KERBIN_SEA_LEVEL_DENSITY,
    AerodynamicModel,
)


def numerical_jacobian_velocity(
    model: AerodynamicModel,
    pos: npt.NDArray[np.float64],
    vel: npt.NDArray[np.float64],
    mass: float,
    delta: float = 1e-4,
) -> npt.NDArray[np.float64]:
    """Computes central finite-difference Jacobian da_drag / dv."""
    j_num = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        vel_plus = vel.copy()
        vel_minus = vel.copy()
        vel_plus[i] += delta
        vel_minus[i] -= delta
        a_plus = model.acceleration(pos, vel_plus, mass)
        a_minus = model.acceleration(pos, vel_minus, mass)
        j_num[:, i] = (a_plus - a_minus) / (2.0 * delta)
    return j_num


def numerical_jacobian_position(
    model: AerodynamicModel,
    pos: npt.NDArray[np.float64],
    vel: npt.NDArray[np.float64],
    mass: float,
    delta: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Computes central finite-difference Jacobian da_drag / dr."""
    j_num = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        pos_plus = pos.copy()
        pos_minus = pos.copy()
        pos_plus[i] += delta
        pos_minus[i] -= delta
        a_plus = model.acceleration(pos_plus, vel, mass)
        a_minus = model.acceleration(pos_minus, vel, mass)
        j_num[:, i] = (a_plus - a_minus) / (2.0 * delta)
    return j_num


@pytest.fixture
def default_aero_model() -> AerodynamicModel:
    # Typical booster parameters: Cd * A = 0.5 * 10.0 m^2 = 5.0 m^2
    return AerodynamicModel(cd_area=5.0)


class TestAerodynamicModel:
    def test_density_profile(
        self, default_aero_model: AerodynamicModel
    ) -> None:
        model = default_aero_model

        # Sea level
        assert np.isclose(model.density(0.0), KERBIN_SEA_LEVEL_DENSITY)

        # Clamped negative altitude
        assert np.isclose(model.density(-500.0), KERBIN_SEA_LEVEL_DENSITY)

        # Scale height
        assert np.isclose(
            model.density(KERBIN_SCALE_HEIGHT),
            KERBIN_SEA_LEVEL_DENSITY / np.e,
            rtol=1e-6,
        )

        # Space (above atmosphere)
        assert model.density(KERBIN_ATMOSPHERE_DEPTH) == 0.0
        assert model.density(KERBIN_ATMOSPHERE_DEPTH + 10_000.0) == 0.0

    def test_density_gradient_vs_finite_difference(
        self, default_aero_model: AerodynamicModel
    ) -> None:
        model = default_aero_model
        h = 15_000.0
        delta = 0.1

        grad_analytical = model.density_gradient(h)
        d_rho_num = (model.density(h + delta) - model.density(h - delta)) / (
            2.0 * delta
        )

        # Only vertical gradient should be non-zero (assuming up_vector = [0, 0, 1])
        assert np.isclose(grad_analytical[0], 0.0)
        assert np.isclose(grad_analytical[1], 0.0)
        assert np.isclose(grad_analytical[2], d_rho_num, rtol=1e-5)

    def test_drag_acceleration_direction_and_scaling(
        self, default_aero_model: AerodynamicModel
    ) -> None:
        model = default_aero_model
        pos = np.array([0.0, 0.0, 5_000.0], dtype=np.float64)
        vel = np.array([-150.0, 80.0, -320.0], dtype=np.float64)
        mass = 12_000.0

        acc = model.acceleration(pos, vel, mass)

        # Drag must strictly oppose velocity
        unit_v = vel / np.linalg.norm(vel)
        unit_a = acc / np.linalg.norm(acc)
        assert np.isclose(np.dot(unit_v, unit_a), -1.0, atol=1e-10)

        # Doubling speed should quadruple drag acceleration
        acc_double = model.acceleration(pos, vel * 2.0, mass)
        assert np.isclose(
            np.linalg.norm(acc_double), 4.0 * np.linalg.norm(acc), rtol=1e-6
        )

    def test_zero_and_edge_conditions(
        self, default_aero_model: AerodynamicModel
    ) -> None:
        model = default_aero_model
        pos = np.array([0.0, 0.0, 5_000.0], dtype=np.float64)
        vel = np.array([0.0, 0.0, -200.0], dtype=np.float64)

        # Zero velocity -> zero acceleration
        assert np.all(
            model.acceleration(pos, np.zeros(3), mass=10_000.0) == 0.0
        )

        # Zero or negative mass -> zero acceleration (guards against div by zero)
        assert np.all(model.acceleration(pos, vel, mass=0.0) == 0.0)
        assert np.all(model.acceleration(pos, vel, mass=-5.0) == 0.0)

        # Space -> zero acceleration
        pos_space = np.array([0.0, 0.0, KERBIN_ATMOSPHERE_DEPTH + 1_000.0])
        assert np.all(model.acceleration(pos_space, vel, mass=10_000.0) == 0.0)
        assert np.all(
            model.jacobian_position(pos_space, vel, mass=10_000.0) == 0.0
        )

    @pytest.mark.parametrize(
        "vel",
        [
            np.array([0.0, 0.0, -250.0]),
            np.array([120.0, -80.0, -350.0]),
            np.array([-500.0, 300.0, 0.0]),
            np.array([15.0, 20.0, 45.0]),
        ],
    )
    def test_jacobian_velocity_vs_finite_difference(
        self, default_aero_model: AerodynamicModel, vel: npt.NDArray[np.float64]
    ) -> None:
        model = default_aero_model
        pos = np.array([500.0, -200.0, 12_000.0], dtype=np.float64)
        mass = 15_000.0

        j_analytical = model.jacobian_velocity(pos, vel, mass)
        j_numerical = numerical_jacobian_velocity(
            model, pos, vel, mass, delta=1e-3
        )

        # Velocity Jacobian should be symmetric
        assert np.allclose(j_analytical, j_analytical.T, atol=1e-10)
        # Analytical vs numerical match
        assert np.allclose(j_analytical, j_numerical, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize(
        "altitude",
        [1_000.0, 10_000.0, 25_000.0, 50_000.0],
    )
    def test_jacobian_position_vs_finite_difference(
        self, default_aero_model: AerodynamicModel, altitude: float
    ) -> None:
        model = default_aero_model
        pos = np.array([1_200.0, -800.0, altitude], dtype=np.float64)
        vel = np.array([-90.0, 45.0, -280.0], dtype=np.float64)
        mass = 14_000.0

        j_analytical = model.jacobian_position(pos, vel, mass)
        j_numerical = numerical_jacobian_position(
            model, pos, vel, mass, delta=1.0
        )

        assert np.allclose(j_analytical, j_numerical, rtol=1e-4, atol=1e-6)
