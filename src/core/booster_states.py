from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast, override

from krpc.services.spacecenter import SASMode, Vessel
from krpc.stream import Stream

from .fsm import State


@dataclass
class BoosterContext:
    active_vessel: Vessel
    _altitude_stream: Stream
    _velocity_stream: Stream
    _drag_stream: Stream

    landed_altitude: float = 0.0

    @property
    def altitude(self) -> float:
        return cast(float, self._altitude_stream())

    @property
    def velocity(self) -> tuple[float, float, float]:
        return cast(tuple[float, float, float], self._velocity_stream())

    @property
    def drag(self) -> tuple[float, float, float]:
        return cast(tuple[float, float, float], self._drag_stream())


class LaunchState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Launching")
        _context.landed_altitude = _context.altitude

        control = _context.active_vessel.control
        control.sas = True
        control.sas_mode = SASMode.radial
        control.throttle = 1.0
        control.activate_next_stage()
        control.legs = False

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        altitude: float = _context.altitude

        if altitude > 500.0:
            return CoastState

        return None


class CoastState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Coasting")
        control = _context.active_vessel.control
        control.throttle = 0.0

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        velocity = _context.velocity

        if velocity[2] > 0.0:
            return LandingState

        return None


class LandingState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Landing")
        control = _context.active_vessel.control
        control.sas_mode = SASMode.retrograde
        control.brakes = True

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        vessel = _context.active_vessel
        control = vessel.control

        alt = _context.altitude
        vel = _context.velocity
        drag_vec = _context.drag

        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        drag_force = math.sqrt(
            drag_vec[0] ** 2 + drag_vec[1] ** 2 + drag_vec[2] ** 2
        )

        g = vessel.orbit.body.surface_gravity
        mass = vessel.mass
        available_thrust = vessel.available_thrust

        if available_thrust == 0 or mass == 0:
            control.throttle = 1.0
            return None

        max_accel = available_thrust / mass
        drag_accel = drag_force / mass
        net_accel = max_accel + drag_accel - g

        if net_accel <= 0.0:
            control.throttle = 1.0
            return None

        dist = alt - _context.landed_altitude

        if dist < 5.0 and speed < 1.0:
            return LandedState

        time_to_landing = dist / speed if speed > 0 else 999.0
        if time_to_landing <= 10.0 and not control.legs:
            control.legs = True

        vacuum_net_accel = max_accel - g
        suicide_burn_dist = (
            (speed**2) / (2.0 * vacuum_net_accel)
            if vacuum_net_accel > 0
            else float("inf")
        )

        if dist <= suicide_burn_dist * 1.1:
            target_accel = (speed**2) / (2.0 * dist) if dist > 0 else max_accel
            required_accel = target_accel + g - drag_accel
            throttle_setting = required_accel / max_accel
            control.throttle = max(0.0, min(1.0, throttle_setting))
        else:
            control.throttle = 0.0

        return None


class LandedState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Landed")
        _context.active_vessel.control.throttle = 0.0
        _context.active_vessel.control.sas = False
        _context.active_vessel.control.brakes = False

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        return None
