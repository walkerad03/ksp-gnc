from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.kerbal.coordinates import TopocentricFrame
from src.kerbal.telemetry import TelemetryBridge
from src.state.estimation import StateEstimation


class MockStream:
    """Simulates a kRPC Stream object."""

    def __init__(self, getter: Any, *args: Any) -> None:
        self.getter: Any = getter
        self.args: tuple[Any, ...] = args
        self.removed: bool = False

    def __call__(self) -> Any:
        if callable(self.getter):
            return self.getter(*self.args)
        return self.getter

    def remove(self) -> None:
        self.removed = True


class MockBody:
    def __init__(self) -> None:
        self.reference_frame: str = "Body.reference_frame"


class MockOrbit:
    def __init__(self) -> None:
        self.body: MockBody = MockBody()


class MockVesselTelemetry:
    def __init__(
        self,
        position: tuple[float, float, float] = (600_000.0, 0.0, 0.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 16_000.0,
        available_thrust: float = 400_000.0,
    ) -> None:
        self._pos: tuple[float, float, float] = position
        self._vel: tuple[float, float, float] = velocity
        self.mass: float = mass
        self.available_thrust: float = available_thrust
        self.orbit: MockOrbit = MockOrbit()

    def position(self, ref_frame: Any) -> tuple[float, float, float]:
        return self._pos

    def velocity(self, ref_frame: Any) -> tuple[float, float, float]:
        return self._vel

    def set_pos(self, pos: tuple[float, float, float]) -> None:
        self._pos = pos

    def set_vel(self, vel: tuple[float, float, float]) -> None:
        self._vel = vel


class MockClient:
    def __init__(self) -> None:
        self.streams: list[MockStream] = []

    def add_stream(self, target: Any, *args: Any) -> MockStream:
        stream = MockStream(target, *args)
        self.streams.append(stream)
        return stream


class TestTelemetryBridge:
    @pytest.fixture
    def setup_bridge(
        self,
    ) -> tuple[TelemetryBridge, MockVesselTelemetry, TopocentricFrame]:
        conn = MockClient()
        # Vessel positioned 500m above equator at prime meridian
        vessel = MockVesselTelemetry(
            position=(600_500.0, 0.0, 0.0),
            velocity=(0.0, 10.0, -50.0),
            mass=15_000.0,
            available_thrust=450_000.0,
        )
        frame = TopocentricFrame.from_lat_lon(
            0.0, 0.0, altitude=0.0, body_radius=600_000.0
        )
        bridge = TelemetryBridge(
            conn=conn, vessel=vessel, frame=frame, use_streams=True
        )
        return bridge, vessel, frame

    def test_stream_setup_and_data_retrieval(
        self,
        setup_bridge: tuple[
            TelemetryBridge, MockVesselTelemetry, TopocentricFrame
        ],
    ) -> None:
        bridge, vessel, _ = setup_bridge

        assert bridge._streams_active is True
        assert bridge.available_thrust == 450_000.0

        r_ecef, v_ecef, mass, avail = bridge.get_raw_ecef_state()
        assert r_ecef == (600_500.0, 0.0, 0.0)
        assert v_ecef == (0.0, 10.0, -50.0)
        assert mass == 15_000.0
        assert avail == 450_000.0

    def test_state_estimation_update(
        self,
        setup_bridge: tuple[
            TelemetryBridge, MockVesselTelemetry, TopocentricFrame
        ],
    ) -> None:
        bridge, vessel, _ = setup_bridge

        state = bridge.update()
        assert isinstance(state, StateEstimation)
        assert np.isclose(state.mass, 15_000.0)

        # At (lat 0, lon 0), +X_ecef is Up in ENU (+Z_enu)
        # Position was (600_500, 0, 0) relative to pad at (600_000, 0, 0)
        # So ENU altitude Z must be +500m, X and Y must be 0
        np.testing.assert_allclose(state.position, [0.0, 0.0, 500.0], atol=1e-6)

        # In-place update of existing StateEstimation object
        existing = StateEstimation()
        vessel.set_pos((600_200.0, 0.0, 0.0))
        vessel.set_vel((0.0, 0.0, -30.0))
        vessel.mass = 14_500.0

        ret = bridge.update(existing)
        assert ret is existing
        assert np.isclose(existing.mass, 14_500.0)
        np.testing.assert_allclose(
            existing.position, [0.0, 0.0, 200.0], atol=1e-6
        )

    def test_fallback_without_streams(self) -> None:
        plain_conn: Any = object()
        vessel = MockVesselTelemetry(
            position=(600_300.0, 0.0, 0.0),
            velocity=(-5.0, 0.0, -40.0),
            mass=14_000.0,
            available_thrust=300_000.0,
        )
        frame = TopocentricFrame.from_lat_lon(
            0.0, 0.0, altitude=0.0, body_radius=600_000.0
        )
        bridge = TelemetryBridge(
            conn=plain_conn, vessel=vessel, frame=frame, use_streams=False
        )

        assert bridge._streams_active is False
        assert bridge.available_thrust == 300_000.0

        state = bridge.update()
        assert np.isclose(state.mass, 14_000.0)
        np.testing.assert_allclose(state.position, [0.0, 0.0, 300.0], atol=1e-6)

    def test_clean_close(
        self,
        setup_bridge: tuple[
            TelemetryBridge, MockVesselTelemetry, TopocentricFrame
        ],
    ) -> None:
        bridge, _, _ = setup_bridge
        assert bool(bridge._streams_active) is True

        bridge.close()
        assert bool(bridge._streams_active) is False
        # Multiple closes should not raise
        bridge.close()
