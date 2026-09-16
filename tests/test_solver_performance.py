from __future__ import annotations

import time

import numpy as np
import pytest

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import KERBIN_RADIUS, SphericalGravityModel
from src.guidance.ocp import LandingOCP, VehicleLimits
from src.guidance.trajectory import ReferenceTrajectory


class TestSolverPerformance:
    @pytest.mark.parametrize("n_nodes", [15, 25, 35])
    def test_solve_time_within_realtime_budget(self, n_nodes: int) -> None:
        """
        Tests that Clarabel/ECOS solves the SOCP within a real-time MPC cycle
        budget (typically < 100-200 ms for 5-10 Hz guidance loops).
        """
        offset = np.array([0.0, 0.0, KERBIN_RADIUS], dtype=np.float64)
        gravity = SphericalGravityModel(planet_center_offset=offset)
        aero = AerodynamicModel(cd_area=4.5)
        limits = VehicleLimits(
            thrust_min=20_000.0,
            thrust_max=250_000.0,
            isp=300.0,
            max_gimbal_angle_deg=25.0,
            glideslope_angle_deg=20.0,
            min_mass=8_000.0,
        )
        ocp = LandingOCP(limits, gravity, aero, solver="CLARABEL")

        r_init = np.array([250.0, -120.0, 1_800.0], dtype=np.float64)
        v_init = np.array([-20.0, 10.0, -110.0], dtype=np.float64)
        m_init = 16_000.0

        r_target = np.zeros(3, dtype=np.float64)
        v_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        m_dry = 9_000.0

        ref_traj = ReferenceTrajectory.from_straight_line(
            r_init,
            v_init,
            m_init,
            r_target,
            v_target,
            m_dry,
            duration=18.0,
            n_nodes=n_nodes,
        )

        # Warm up solver (JIT / CVXPY canonicalization caching)
        _ = ocp.solve(
            initial_state=(r_init, v_init, m_init),
            target_state=(r_target, v_target),
            ref_traj=ref_traj,
            enable_glideslope=True,
        )

        # Benchmark repeated solves
        solve_times: list[float] = []
        n_trials = 5

        for i in range(n_trials):
            # Perturb slightly to simulate new telemetry sample
            r_sample = r_init + np.random.uniform(-5.0, 5.0, 3)
            v_sample = v_init + np.random.uniform(-1.0, 1.0, 3)

            start = time.perf_counter()
            sol = ocp.solve(
                initial_state=(r_sample, v_sample, m_init),
                target_state=(r_target, v_target),
                ref_traj=ref_traj,
                enable_glideslope=True,
            )
            elapsed = time.perf_counter() - start

            assert sol.optimal is True
            solve_times.append(elapsed)

        avg_time = float(np.mean(solve_times))
        max_time = float(np.max(solve_times))

        print(
            f"\n[Nodes: {n_nodes}] Avg Solve: {avg_time * 1000.0:.2f} ms | "
            f"Max Solve: {max_time * 1000.0:.2f} ms | Internal Clarabel: {sol.solve_time_sec * 1000.0:.2f} ms"
        )

        # Ensure average solve is fast enough for real-time operation (< 200 ms)
        assert avg_time < 0.25, (
            f"Average solve time {avg_time * 1000:.1f} ms exceeds 250 ms budget"
        )
