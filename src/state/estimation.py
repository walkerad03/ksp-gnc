from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class StateEstimation:
    _position: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0, 0, 0], dtype=np.float64)
    )
    _velocity: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0, 0, 0], dtype=np.float64)
    )
    _mass: float = 0.0

    def __repr__(self) -> str:
        return (
            f"StateEstimation(\n"
            f"    Position: {self._position}\n"
            f"    Velocity: {self._velocity}\n"
            f"    Mass:     {self._mass}\n"
            f")"
        )

    @property
    def position(self) -> npt.NDArray[np.float64]:
        return self._position

    @position.setter
    def position(
        self, position: tuple[float, float, float] | npt.NDArray[np.float64]
    ) -> None:
        assert len(position) == 3
        if isinstance(position, tuple):
            self._position = np.asarray(position, dtype=np.float64)
        else:
            self._position = position

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        return self._velocity

    @velocity.setter
    def velocity(
        self, velocity: tuple[float, float, float] | npt.NDArray[np.float64]
    ) -> None:
        assert len(velocity) == 3
        if isinstance(velocity, tuple):
            self._velocity = np.asarray(velocity, dtype=np.float64)
        else:
            self._velocity = velocity

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, mass: float) -> None:
        assert mass >= 0, "Mass must be non-negative."
        self._mass = mass


if __name__ == "__main__":
    state = StateEstimation()
    state.position = (1, 1, 1)
    state.velocity = (0.5, 0.5, 0.5)
    state.mass = 5

    print(state.mass)
