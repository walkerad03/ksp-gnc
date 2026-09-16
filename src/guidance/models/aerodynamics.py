from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

KERBIN_ATMOSPHERE_DEPTH = 70_000.0  # m
KERBIN_SCALE_HEIGHT = 5600.0  # m
KERBIN_SEA_LEVEL_DENSITY = 1.225  # kg/m^3


@dataclass(frozen=True)
class AerodynamicModel:
    cd_area: float
    scale_height: float = KERBIN_SCALE_HEIGHT
    rho_0: float = KERBIN_SEA_LEVEL_DENSITY
    max_altitude: float = KERBIN_ATMOSPHERE_DEPTH
    up_vector: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    def density(self, altitude: float) -> float:
        if altitude < 0:
            altitude = 0.0
        elif altitude >= self.max_altitude:
            return 0.0
        return float(self.rho_0 * np.exp(-altitude / self.scale_height))

    def density_gradient(self, altitude: float) -> npt.NDArray[np.float64]:
        if altitude < 0 or altitude >= self.max_altitude:
            return np.zeros(3, dtype=np.float64)
        rho = self.density(altitude)
        return (-rho / self.scale_height) * self.up_vector

    def acceleration(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        mass: float,
    ) -> npt.NDArray[np.float64]:
        if mass <= 0.0:
            return np.zeros(3, dtype=np.float64)

        altitude = float(np.dot(position, self.up_vector))
        rho = self.density(altitude)
        if rho <= 0.0:
            return np.zeros(3, dtype=np.float64)

        v_mag = np.linalg.norm(velocity)
        if v_mag == 0.0:
            return np.zeros(3, dtype=np.float64)

        coeff = 0.5 * rho * self.cd_area / mass
        return -coeff * v_mag * velocity

    def jacobian_velocity(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        mass: float,
        eps: float = 1e-6,
    ) -> npt.NDArray[np.float64]:
        """
        Analytical Jacobian da_drag / dv (3x3 matrix).
        """
        if mass <= 0.0:
            return np.zeros((3, 3), dtype=np.float64)

        altitude = float(np.dot(position, self.up_vector))
        rho = self.density(altitude)
        if rho <= 0.0:
            return np.zeros((3, 3), dtype=np.float64)

        v_mag = float(np.linalg.norm(velocity))
        coeff = 0.5 * rho * self.cd_area / mass
        eye3 = np.eye(3, dtype=np.float64)

        if v_mag < eps:
            return -coeff * eps * eye3

        outer_v = np.outer(velocity, velocity)
        return -coeff * (v_mag * eye3 + outer_v / v_mag)

    def jacobian_position(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        mass: float,
    ) -> npt.NDArray[np.float64]:
        altitude = float(np.dot(position, self.up_vector))
        if altitude < 0 or altitude >= self.max_altitude:
            return np.zeros((3, 3), dtype=np.float64)

        a_drag = self.acceleration(position, velocity, mass)
        # da_drag / dr = (1 / rho) * a_drag * (d_rho / dr)^T = -(1 / scale_height) * a_drag * up_vector^T
        return -(1.0 / self.scale_height) * np.outer(a_drag, self.up_vector)
