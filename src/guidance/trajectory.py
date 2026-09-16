from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import SphericalGravityModel


@dataclass
class LinearizedNodeDynamics:
    A_pos: npt.NDArray[np.float64]  # 3x3 (d(g + a_drag)/dr)
    A_vel: npt.NDArray[np.float64]  # 3x3 (d(a_drag)/dv)
    a_const: npt.NDArray[np.float64]  # 3x1 constant acceleration offset


@dataclass
class ReferenceTrajectory:
    """Discretized trajectory containing positions, velocities, masses, and time stamps."""

    times: npt.NDArray[np.float64]  # Shape: (N+1,)
    positions: npt.NDArray[np.float64]  # Shape: (N+1, 3)
    velocities: npt.NDArray[np.float64]  # Shape: (N+1, 3)
    masses: npt.NDArray[np.float64]  # Shape: (N+1,)
    thrusts: npt.NDArray[np.float64]  # Shape: (N, 3)

    @property
    def n_nodes(self) -> int:
        return len(self.times) - 1

    @property
    def total_duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def dt(self) -> float:
        if self.n_nodes <= 0:
            return 0.0
        return self.total_duration / self.n_nodes

    @classmethod
    def from_straight_line(
        cls,
        r_init: npt.NDArray[np.float64],
        v_init: npt.NDArray[np.float64],
        m_init: float,
        r_target: npt.NDArray[np.float64],
        v_target: npt.NDArray[np.float64],
        m_dry: float,
        duration: float,
        n_nodes: int = 30,
    ) -> ReferenceTrajectory:
        """
        Creates an initial non-physical straight-line reference trajectory
        linearly connecting initial state to target state over n_nodes intervals.
        """
        assert n_nodes > 0, "n_nodes must be greater than 0"
        assert duration > 0.0, "duration must be positive"

        times = np.linspace(0.0, duration, n_nodes + 1, dtype=np.float64)
        alphas = np.linspace(0.0, 1.0, n_nodes + 1, dtype=np.float64)[
            :, np.newaxis
        ]

        positions = (1.0 - alphas) * r_init + alphas * r_target
        velocities = (1.0 - alphas) * v_init + alphas * v_target
        masses = np.linspace(m_init, m_dry, n_nodes + 1, dtype=np.float64)
        thrusts = np.zeros((n_nodes, 3), dtype=np.float64)

        return cls(
            times=times,
            positions=positions,
            velocities=velocities,
            masses=masses,
            thrusts=thrusts,
        )

    def compute_node_linearizations(
        self,
        gravity_model: SphericalGravityModel,
        aero_model: AerodynamicModel,
    ) -> list[LinearizedNodeDynamics]:
        """
        Evaluates analytical gravity & aerodynamic Jacobians at each of the N+1 nodes.
        Returns a list of LinearizedNodeDynamics objects.
        """
        linearizations: list[LinearizedNodeDynamics] = []

        for k in range(self.n_nodes + 1):
            r_k = self.positions[k]
            v_k = self.velocities[k]
            m_k = self.masses[k]

            g_k = gravity_model.acceleration(r_k)
            a_drag_k = aero_model.acceleration(r_k, v_k, m_k)

            # Jacobians
            J_grav_r = gravity_model.jacobian(r_k)
            J_drag_r = aero_model.jacobian_position(r_k, v_k, m_k)
            J_drag_v = aero_model.jacobian_velocity(r_k, v_k, m_k)

            A_pos = J_grav_r + J_drag_r
            A_vel = J_drag_v

            # Constant acceleration offset: a_total - A_pos * r_ref - A_vel * v_ref
            a_total = g_k + a_drag_k
            a_const = a_total - (A_pos @ r_k) - (A_vel @ v_k)

            linearizations.append(
                LinearizedNodeDynamics(
                    A_pos=A_pos,
                    A_vel=A_vel,
                    a_const=a_const,
                )
            )

        return linearizations

    def shift(self, dt_elapsed: float | None = None) -> ReferenceTrajectory:
        """
        Shifts the trajectory forward by one node for receding-horizon MPC.
        The first node is dropped, and the last node is repeated with updated time.
        """
        dt = self.dt if dt_elapsed is None else dt_elapsed

        new_times = self.times.copy()
        new_times[:-1] = self.times[1:] - dt
        new_times[-1] = new_times[-2] + dt

        new_positions = np.empty_like(self.positions)
        new_positions[:-1] = self.positions[1:]
        new_positions[-1] = self.positions[-1]

        new_velocities = np.empty_like(self.velocities)
        new_velocities[:-1] = self.velocities[1:]
        new_velocities[-1] = self.velocities[-1]

        new_masses = np.empty_like(self.masses)
        new_masses[:-1] = self.masses[1:]
        new_masses[-1] = self.masses[-1]

        new_thrusts = np.empty_like(self.thrusts)
        new_thrusts[:-1] = self.thrusts[1:]
        new_thrusts[-1] = self.thrusts[-1]

        return ReferenceTrajectory(
            times=new_times,
            positions=new_positions,
            velocities=new_velocities,
            masses=new_masses,
            thrusts=new_thrusts,
        )
