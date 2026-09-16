from __future__ import annotations

from typing import Any, cast

from src.kerbal.coordinates import TopocentricFrame
from src.state.estimation import StateEstimation


class TelemetryBridge:
    def __init__(
        self,
        conn: Any,
        vessel: Any,
        frame: TopocentricFrame,
        use_streams: bool = True,
    ) -> None:
        self.conn: Any = conn
        self.vessel: Any = vessel
        self.frame: TopocentricFrame = frame
        self.use_streams: bool = use_streams

        self._streams_active: bool = False
        self._stream_pos: Any | None = None
        self._stream_vel: Any | None = None
        self._stream_mass: Any | None = None
        self._stream_avail_thrust: Any | None = None

        if self.use_streams and hasattr(conn, "add_stream"):
            self._setup_streams()

    def _setup_streams(self) -> None:
        ref_frame = self.vessel.orbit.body.reference_frame

        self._stream_pos = self.conn.add_stream(self.vessel.position, ref_frame)
        self._stream_vel = self.conn.add_stream(self.vessel.velocity, ref_frame)
        self._stream_mass = self.conn.add_stream(getattr, self.vessel, "mass")
        self._stream_avail_thrust = self.conn.add_stream(
            getattr, self.vessel, "available_thrust"
        )
        self._streams_active = True

    def close(self) -> None:
        if not self._streams_active:
            return

        for stream in [
            self._stream_pos,
            self._stream_vel,
            self._stream_mass,
            self._stream_avail_thrust,
        ]:
            if stream is not None and hasattr(stream, "remove"):
                try:
                    stream.remove()
                except Exception:
                    pass

        self._streams_active = False

    def get_raw_ecef_state(
        self,
    ) -> tuple[
        tuple[float, float, float], tuple[float, float, float], float, float
    ]:
        """Reads raw state from kRPC streams (or direct RPC fallback)."""
        ref_frame = self.vessel.orbit.body.reference_frame

        if self._streams_active and self._stream_pos and self._stream_vel:
            r_ecef = self._stream_pos()
            v_ecef = self._stream_vel()
            mass = (
                self._stream_mass()
                if self._stream_mass
                else float(self.vessel.mass)
            )
            avail_thrust = (
                self._stream_avail_thrust()
                if self._stream_avail_thrust
                else float(self.vessel.available_thrust)
            )
        else:
            r_ecef = self.vessel.position(ref_frame)
            v_ecef = self.vessel.velocity(ref_frame)
            mass = float(self.vessel.mass)
            avail_thrust = float(self.vessel.available_thrust)

        return (
            cast(tuple[float, float, float], r_ecef),
            cast(tuple[float, float, float], v_ecef),
            cast(float, mass),
            cast(float, avail_thrust),
        )

    def update(
        self, state_estimate: StateEstimation | None = None
    ) -> StateEstimation:
        """
        Polls telemetry, transforms coordinates from ECEF to local landing pad ENU,
        and populates the provided (or a newly instantiated) StateEstimation.
        """
        r_ecef, v_ecef, mass, _ = self.get_raw_ecef_state()

        r_enu = self.frame.ecef_to_enu_position(r_ecef)
        v_enu = self.frame.ecef_to_enu_velocity(v_ecef)

        if state_estimate is None:
            state_estimate = StateEstimation()

        state_estimate.position = r_enu
        state_estimate.velocity = v_enu
        state_estimate.mass = mass

        return state_estimate

    @property
    def available_thrust(self) -> float:
        """Current maximum available thrust in Newtons."""
        if self._streams_active and self._stream_avail_thrust:
            return cast(float, self._stream_avail_thrust())
        return float(self.vessel.available_thrust)
