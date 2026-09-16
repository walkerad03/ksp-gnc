from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.kerbal.actuation import ActuationBridge, ActuationCommand
from src.kerbal.coordinates import TopocentricFrame


class MockAutoPilot:
    def __init__(self) -> None:
        self.reference_frame: Any = None
        self.target_direction: tuple[float, float, float] | None = None
        self.target_roll: float = 0.0
        self.engaged: bool = False

    def engage(self) -> None:
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False


class MockControl:
    def __init__(self) -> None:
        self.throttle: float = float(0.0)
        self.sas: bool = True
        self.rcs: bool = False


class MockBody:
    def __init__(self) -> None:
        self.reference_frame: str = "Body.reference_frame"


class MockOrbit:
    def __init__(self) -> None:
        self.body: MockBody = MockBody()


class MockVesselActuation:
    def __init__(self, available_thrust: float = 450_000.0) -> None:
        self.available_thrust: float = available_thrust
        self.auto_pilot: MockAutoPilot = MockAutoPilot()
        self.control: MockControl = MockControl()
        self.orbit: MockOrbit = MockOrbit()


class TestActuationBridgeUnit:
    @pytest.fixture
    def setup_bridge(
        self,
    ) -> tuple[ActuationBridge, MockVesselActuation, TopocentricFrame]:
        frame = TopocentricFrame.from_lat_lon(
            0.0, 0.0, altitude=0.0, body_radius=600_000.0
        )
        vessel = MockVesselActuation(available_thrust=400_000.0)
        bridge = ActuationBridge(
            vessel=vessel,
            frame=frame,
            min_throttle_cutoff=0.05,
            max_tilt_angle_deg=25.0,
        )
        return bridge, vessel, frame

    def test_throttle_mapping_and_deadband(
        self,
        setup_bridge: tuple[
            ActuationBridge, MockVesselActuation, TopocentricFrame
        ],
    ) -> None:
        bridge, _, _ = setup_bridge

        # 1. Zero or negative thrust -> cutoff (0.0 throttle)
        cmd_zero = bridge.compute_command(
            [0.0, 0.0, 0.0], Gamma_0=0.0, available_thrust=400_000.0
        )
        assert cmd_zero.throttle == 0.0
        assert cmd_zero.is_engine_active is False

        # 2. Below minimum cutoff (4% < 5%) -> cutoff
        cmd_low = bridge.compute_command(
            [0.0, 0.0, 16_000.0], Gamma_0=16_000.0, available_thrust=400_000.0
        )
        assert cmd_low.throttle == 0.0
        assert cmd_low.is_engine_active is False

        # 3. Exactly at / above cutoff (50%)
        cmd_mid = bridge.compute_command(
            [0.0, 0.0, 200_000.0], Gamma_0=200_000.0, available_thrust=400_000.0
        )
        assert np.isclose(cmd_mid.throttle, 0.5)
        assert cmd_mid.is_engine_active is True

        # 4. Saturation at 100% (overshoot requested)
        cmd_sat = bridge.compute_command(
            [0.0, 0.0, 500_000.0], Gamma_0=500_000.0, available_thrust=400_000.0
        )
        assert cmd_sat.throttle == 1.0
        assert cmd_sat.is_engine_active is True

        # 5. Zero available thrust edge case
        cmd_no_avail = bridge.compute_command(
            [0.0, 0.0, 100_000.0], Gamma_0=100_000.0, available_thrust=0.0
        )
        assert cmd_no_avail.throttle == 0.0
        assert cmd_no_avail.is_engine_active is False

    def test_tilt_angle_and_safety_clamping(
        self,
        setup_bridge: tuple[
            ActuationBridge, MockVesselActuation, TopocentricFrame
        ],
    ) -> None:
        bridge, _, frame = setup_bridge

        # Pure vertical (+Z in ENU) -> 0 deg tilt
        cmd_vert = bridge.compute_command(
            [0.0, 0.0, 100_000.0], Gamma_0=100_000.0, available_thrust=400_000.0
        )
        assert np.isclose(cmd_vert.tilt_angle_deg, 0.0, atol=1e-3)
        # In equator lon 0, local Up is [1, 0, 0] in ECEF
        np.testing.assert_allclose(
            cmd_vert.direction_ecef, [1.0, 0.0, 0.0], atol=1e-6
        )

        # 15 deg East tilt (within 25 deg limit)
        angle_rad = np.deg2rad(15.0)
        T_tilt = (
            np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)]) * 100_000.0
        )
        cmd_tilt = bridge.compute_command(
            T_tilt, Gamma_0=100_000.0, available_thrust=400_000.0
        )
        assert np.isclose(cmd_tilt.tilt_angle_deg, 15.0, atol=1e-3)

        # 45 deg excessive tilt -> must be clamped to max_tilt_angle_deg (25 deg)
        angle_excessive = np.deg2rad(45.0)
        T_excess = (
            np.array([np.sin(angle_excessive), 0.0, np.cos(angle_excessive)])
            * 100_000.0
        )
        cmd_clamped = bridge.compute_command(
            T_excess, Gamma_0=100_000.0, available_thrust=400_000.0
        )
        assert np.isclose(cmd_clamped.tilt_angle_deg, 25.0, atol=1e-3)

        # Direction vector must remain a valid unit vector
        dir_norm = float(np.linalg.norm(np.array(cmd_clamped.direction_ecef)))
        assert np.isclose(dir_norm, 1.0, atol=1e-9)

    def test_apply_and_disengage(
        self,
        setup_bridge: tuple[
            ActuationBridge, MockVesselActuation, TopocentricFrame
        ],
    ) -> None:
        bridge, vessel, _ = setup_bridge

        cmd = ActuationCommand(
            throttle=0.75,
            direction_ecef=(1.0, 0.0, 0.0),
            thrust_magnitude=300_000.0,
            tilt_angle_deg=0.0,
            is_engine_active=True,
        )

        assert bool(vessel.control.sas) is True
        assert bool(vessel.auto_pilot.engaged) is False

        # Apply command
        bridge.apply(cmd)

        assert bool(vessel.control.sas) is False  # SAS turned off
        assert np.isclose(
            float(vessel.control.throttle), 0.75
        )  # Throttle applied
        assert bool(vessel.auto_pilot.engaged) is True  # Autopilot engaged
        assert vessel.auto_pilot.target_direction == (1.0, 0.0, 0.0)
        assert vessel.auto_pilot.reference_frame == "Body.reference_frame"

        # Disengage
        bridge.disengage()
        assert np.isclose(float(vessel.control.throttle), 0.0)
        assert bool(vessel.auto_pilot.engaged) is False


class TestActuationPadIntegration:
    def test_launchpad_static_actuator_sequence(self) -> None:
        """
        Simulates testing actuator mapping on a static vehicle resting
        on the launchpad (e.g. pre-flight gimbal checks and throttle commands).
        """
        # Kerbal Space Center launchpad
        target_lat = -0.0972
        target_lon = -74.5577
        target_alt = 68.0
        frame = TopocentricFrame.from_lat_lon(
            target_lat, target_lon, target_alt
        )

        vessel = MockVesselActuation(available_thrust=450_000.0)
        bridge = ActuationBridge(
            vessel=vessel, frame=frame, min_throttle_cutoff=0.05
        )

        # Synthetic test sequence:
        # Step 1: Vertical hold (no tilt)
        cmd_vert = bridge.compute_command(
            T_0_enu=[0.0, 0.0, 200_000.0],
            Gamma_0=200_000.0,
            available_thrust=vessel.available_thrust,
        )
        bridge.apply(cmd_vert)
        assert np.isclose(vessel.control.throttle, 200_000.0 / 450_000.0)
        assert np.isclose(cmd_vert.tilt_angle_deg, 0.0, atol=1e-3)

        # Local up vector in ECEF (column 2 of R_enu_to_ecef)
        u_up_ecef = frame.R_enu_to_ecef[:, 2]
        np.testing.assert_allclose(
            cmd_vert.direction_ecef, u_up_ecef, atol=1e-6
        )

        # Step 2: Gimbal 10 deg East (positive X in ENU)
        rad_10 = np.deg2rad(10.0)
        T_east = [
            float(np.sin(rad_10) * 200_000.0),
            0.0,
            float(np.cos(rad_10) * 200_000.0),
        ]
        cmd_east = bridge.compute_command(
            T_0_enu=T_east,
            Gamma_0=200_000.0,
            available_thrust=vessel.available_thrust,
        )
        bridge.apply(cmd_east)
        assert np.isclose(cmd_east.tilt_angle_deg, 10.0, atol=1e-3)

        # Step 3: Gimbal 10 deg North (positive Y in ENU)
        T_north = [
            0.0,
            float(np.sin(rad_10) * 200_000.0),
            float(np.cos(rad_10) * 200_000.0),
        ]
        cmd_north = bridge.compute_command(
            T_0_enu=T_north,
            Gamma_0=200_000.0,
            available_thrust=vessel.available_thrust,
        )
        bridge.apply(cmd_north)
        assert np.isclose(cmd_north.tilt_angle_deg, 10.0, atol=1e-3)

        # Step 4: Disengage / shutdown
        bridge.disengage()
        assert np.isclose(float(vessel.control.throttle), 0.0)
        assert bool(vessel.auto_pilot.engaged) is False
