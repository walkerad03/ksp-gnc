from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

KERBIN_MU = 3.5316000e12  # m^3/s^2
KERBIN_RADIUS = 600_000.0  # m


@dataclass(frozen=True)
class SphericalGravityModel:
    mu: float = KERBIN_MU
    planet_center_offset: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def acceleration(
        self, position: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r = position + self.planet_center_offset
        norm_r = np.linalg.norm(r)
        if norm_r == 0.0:
            return np.zeros(3, dtype=np.float64)
        return -self.mu / (norm_r**3) * r

    def jacobian(
        self, position: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r = position + self.planet_center_offset
        norm_r = np.linalg.norm(r)
        if norm_r == 0.0:
            return np.zeros((3, 3), dtype=np.float64)

        eye3 = np.eye(3, dtype=np.float64)
        outer_r = np.outer(r, r)
        return (
            -self.mu / (norm_r**3) * eye3
            + (3.0 * self.mu / (norm_r**5)) * outer_r
        )
