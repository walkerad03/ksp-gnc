from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from src.guidance.models.gravity import (
    KERBIN_MU,
    KERBIN_RADIUS,
    SphericalGravityModel,
)


def numerical_jacobian_gravity(
    model: SphericalGravityModel,
    pos: npt.NDArray[np.float64],
    delta: float = 1e-3,
) -> npt.NDArray[np.float64]:
    """Computes central finite-difference Jacobian dg/dr."""
    j_num = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        pos_plus = pos.copy()
        pos_minus = pos.copy()
        pos_plus[i] += delta
        pos_minus[i] -= delta
        g_plus = model.acceleration(pos_plus)
        g_minus = model.acceleration(pos_minus)
        j_num[:, i] = (g_plus - g_minus) / (2.0 * delta)
    return j_num


class TestSphericalGravityModel:
    def test_surface_acceleration_magnitude(self) -> None:
        """At Kerbin radius, acceleration should be approximately 9.81 m/s^2."""
        model = SphericalGravityModel()
        pos = np.array([0.0, 0.0, KERBIN_RADIUS], dtype=np.float64)
        acc = model.acceleration(pos)

        expected_g0 = KERBIN_MU / (KERBIN_RADIUS**2)
        assert np.isclose(expected_g0, 9.81, atol=0.01)

        # Vector points toward planet center
        assert np.isclose(acc[0], 0.0)
        assert np.isclose(acc[1], 0.0)
        assert np.isclose(acc[2], -expected_g0, rtol=1e-5)

    def test_zero_position(self) -> None:
        """Handles singularity at origin without raising an exception."""
        model = SphericalGravityModel()
        pos = np.zeros(3, dtype=np.float64)
        assert np.all(model.acceleration(pos) == 0.0)
        assert np.all(model.jacobian(pos) == 0.0)

    @pytest.mark.parametrize(
        "position",
        [
            np.array([0.0, 0.0, KERBIN_RADIUS + 5_000.0]),
            np.array([KERBIN_RADIUS + 25_000.0, 0.0, 0.0]),
            np.array([450_000.0, -350_000.0, 250_000.0]),
            np.array([-KERBIN_RADIUS, 12_000.0, -8_000.0]),
        ],
    )
    def test_jacobian_analytical_vs_numerical(
        self, position: npt.NDArray[np.float64]
    ) -> None:
        """Analytical Jacobian dg/dr matches central finite differences."""
        model = SphericalGravityModel()
        j_analytical = model.jacobian(position)
        j_numerical = numerical_jacobian_gravity(model, position, delta=1.0)

        # Gravity Jacobian is symmetric (curl(g) = 0)
        assert np.allclose(j_analytical, j_analytical.T, atol=1e-12)
        # Verify analytical against finite difference
        assert np.allclose(j_analytical, j_numerical, rtol=1e-5, atol=1e-8)

    def test_with_planet_center_offset(self) -> None:
        """Local frame where origin is on Kerbin's surface (e.g. landing pad)."""
        offset = np.array([0.0, 0.0, KERBIN_RADIUS], dtype=np.float64)
        model = SphericalGravityModel(planet_center_offset=offset)

        # At local origin (pad surface)
        local_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        acc = model.acceleration(local_pos)
        assert np.isclose(acc[2], -9.81, atol=0.01)

        # 10 km above pad, with horizontal offset
        local_high = np.array([1_500.0, -2_000.0, 10_000.0], dtype=np.float64)
        j_analytical = model.jacobian(local_high)
        j_numerical = numerical_jacobian_gravity(model, local_high, delta=1.0)

        assert np.allclose(j_analytical, j_numerical, rtol=1e-5, atol=1e-8)
