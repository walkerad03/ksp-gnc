from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import cast, override

from krpc.client import Client
from krpc.services.spacecenter import SASMode, Vessel
from krpc.stream import Stream

from .fsm import State


def v_add(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


def v_sub(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def v_scale(
    v: tuple[float, float, float], scalar: float
) -> tuple[float, float, float]:
    return (v[0] * scalar, v[1] * scalar, v[2] * scalar)


def v_mag(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def v_normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    m = v_mag(v)
    return v_scale(v, 1.0 / m) if m > 0 else (0.0, 0.0, 0.0)


def v_dot(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def v_cross(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    )


def solve_lambert(
    r1: tuple[float, float, float],
    r2: tuple[float, float, float],
    tof: float,
    mu: float,
) -> tuple[float, float, float]:
    """Robust bisection Lambert solver using Lagrange coefficients."""
    R1 = v_mag(r1)
    R2 = v_mag(r2)

    # Clamp cos_nu to prevent floating-point domain errors
    cos_nu = max(-1.0, min(1.0, v_dot(r1, r2) / (R1 * R2)))

    # Assume short way trajectory
    c = math.sqrt(R1**2 + R2**2 - 2 * R1 * R2 * cos_nu)
    s = (R1 + R2 + c) / 2.0

    a_min = s / 2.0
    a_low = a_min
    a_high = a_min * 100.0
    a = (a_low + a_high) / 2.0

    for _ in range(50):
        alpha = 2.0 * math.asin(math.sqrt(s / (2.0 * a)))
        beta = 2.0 * math.asin(math.sqrt((s - c) / (2.0 * a)))

        t_calc = math.sqrt(a**3 / mu) * (
            alpha - beta - (math.sin(alpha) - math.sin(beta))
        )

        if t_calc > tof:
            a_high = a
        else:
            a_low = a
        a = (a_low + a_high) / 2.0

    # Calculate exact Lagrange f and g coefficients
    delta_e = alpha - beta
    f = 1.0 - (a / R1) * (1.0 - math.cos(delta_e))
    g = tof - math.sqrt(a**3 / mu) * (delta_e - math.sin(delta_e))

    v1 = v_scale(v_sub(r2, v_scale(r1, f)), 1.0 / g)
    return v1


@dataclass
class BoosterContext:
    active_vessel: Vessel
    _altitude_stream: Stream
    _velocity_stream: Stream
    _drag_stream: Stream
    _position_stream: Stream
    conn: Client

    target_lat: float = -0.0972
    target_lon: float = -74.5577
    target_alt: float = 68.0
    landed_altitude: float = 0.0

    _logs: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    debug_info: dict[str, float] = field(default_factory=dict)

    def log(self, message: str) -> None:
        self._logs.append(message)

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    @property
    def altitude(self) -> float:
        return cast(float, self._altitude_stream())

    @property
    def velocity(self) -> tuple[float, float, float]:
        return cast(tuple[float, float, float], self._velocity_stream())

    @property
    def drag(self) -> tuple[float, float, float]:
        return cast(tuple[float, float, float], self._drag_stream())

    @property
    def position(self) -> tuple[float, float, float]:
        return cast(tuple[float, float, float], self._position_stream())


class LaunchState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Launching - Executing Gravity Turn")
        _context.landed_altitude = _context.altitude

        self.vessel = _context.active_vessel
        self.control = self.vessel.control
        self.body = self.vessel.orbit.body
        self.ref_frame = self.body.reference_frame

        # Engage autopilot in the rotating body frame
        self.control.sas = False
        self.vessel.auto_pilot.engaged = True
        self.vessel.auto_pilot.reference_frame = self.ref_frame

        self.control.throttle = 1
        self.control.activate_next_stage()
        self.control.legs = False

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        altitude: float = _context.altitude

        # Transition condition
        if altitude > 30000.0:
            return BoostbackState

        # Gravity Turn Pitch Profile
        turn_start_alt = 50.0
        turn_end_alt = 30000.0
        start_pitch = math.radians(90.0)
        end_pitch = math.radians(45.0)

        if altitude < turn_start_alt:
            pitch = start_pitch
        elif altitude > turn_end_alt:
            pitch = end_pitch
        else:
            progress = (altitude - turn_start_alt) / (
                turn_end_alt - turn_start_alt
            )
            pitch = start_pitch - progress * (start_pitch - end_pitch)

        # Calculate local Up and East vectors
        r_body = _context.position
        up_vec = v_normalize(r_body)

        north_pole = (0.0, 1.0, 0.0)
        east_vec = v_normalize(v_cross(up_vec, north_pole))

        # Blend the Up and East vectors based on the calculated pitch
        up_component = v_scale(up_vec, math.sin(pitch))
        east_component = v_scale(east_vec, math.cos(pitch))

        u_desired = v_normalize(v_add(up_component, east_component))

        self.vessel.auto_pilot.target_direction = u_desired

        return None

    @override
    def on_exit(self, _context: BoosterContext) -> None:
        self.vessel.auto_pilot.engaged = False
        self.control.activate_next_stage()


class BoostbackState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Initiating Lambert Boostback Sequence")
        self.vessel = _context.active_vessel
        self.control = self.vessel.control
        self.body = self.vessel.orbit.body
        self.inertial_frame = self.body.non_rotating_reference_frame
        self.ecef_frame = self.body.reference_frame

        # Disable native SAS so it does not fight the kRPC autopilot
        self.control.sas = False
        self.vessel.auto_pilot.engaged = True
        self.burn_calculated = False
        self.vessel.auto_pilot.reference_frame = self.inertial_frame

        # Activate RCS to ensure steering authority in vacuum
        self.control.rcs = True
        self.control.throttle = 1.0

        self.burn_complete = False
        self.u_smoothed = None  # Holds the frozen/smoothed steering vector

        # Target Constants
        self.target_ecef = self.body.surface_position(
            _context.target_lat, _context.target_lon, self.ecef_frame
        )
        self.mu = self.body.gravitational_parameter
        self.omega_vec = self.body.angular_velocity(self.inertial_frame)

        # Estimate a stable Initial Time of Flight based on the chord distance
        r_current = self.vessel.position(self.inertial_frame)
        target_eci_start = _context.conn.space_center.transform_position(
            self.target_ecef, self.ecef_frame, self.inertial_frame
        )
        chord_dist = v_mag(v_sub(target_eci_start, r_current))
        self.time_of_flight = (chord_dist / 800.0) + 120.0

        self.last_time = _context.conn.space_center.ut

        # Visuals
        drawing = _context.conn.drawing
        origin = (0.0, 0.0, 0.0)
        self.dv_line = drawing.add_line(origin, origin, self.inertial_frame)
        self.dv_line.color = (1.0, 0.0, 0.0)
        self.dv_line.thickness = 2.0
        self.vel_line = drawing.add_line(origin, origin, self.inertial_frame)
        self.vel_line.color = (0.0, 1.0, 0.0)
        self.vel_line.thickness = 2.0

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        r_body = _context.position
        v_body = _context.velocity
        up_vec = v_normalize(r_body)
        vert_vel = v_dot(v_body, up_vec)

        if self.burn_complete:
            if _context.altitude < 100000.0 and vert_vel < 0.0:
                return AeroGuidanceState
            return None

        ut = _context.conn.space_center.ut

        # 1. ONE-TIME INITIALIZATION: Exact Ballistic Time of Flight
        if not self.burn_calculated:
            g = self.body.surface_gravity
            h_diff = _context.altitude - _context.target_alt
            # Quadratic formula to find exact arrival time assuming a ballistic arc
            discriminant = max(0.0, vert_vel**2 + 2 * g * h_diff)
            tof = (vert_vel + math.sqrt(discriminant)) / g
            self.t_arrival = ut + tof
            self.burn_calculated = True

        r_current_inertial = self.vessel.position(self.inertial_frame)
        v_current_inertial = self.vessel.velocity(self.inertial_frame)

        # 2. CLOSED-LOOP EXECUTION
        if self.burn_calculated and not self.burn_complete:
            time_to_go = self.t_arrival - ut

            if time_to_go > 1.0:
                # 3. Target ECI Prediction via Rodriguez Rotation
                target_eci_now = _context.conn.space_center.transform_position(
                    self.target_ecef, self.ecef_frame, self.inertial_frame
                )

                # Predict rotation using proper right-handed formula
                theta_vector = v_scale(self.omega_vec, time_to_go)
                theta = v_mag(theta_vector)

                if theta < 1e-9:
                    r2_inertial = target_eci_now
                else:
                    k = v_scale(theta_vector, 1.0 / theta)
                    term1 = v_scale(target_eci_now, math.cos(theta))
                    term2 = v_scale(v_cross(k, target_eci_now), math.sin(theta))
                    term3 = v_scale(
                        k, v_dot(k, target_eci_now) * (1.0 - math.cos(theta))
                    )
                    r2_inertial = v_add(v_add(term1, term2), term3)

                # 4. Lambert Solver & Delta-V
                target_v = solve_lambert(
                    r_current_inertial, r2_inertial, time_to_go, self.mu
                )

                dv_vector = v_sub(target_v, v_current_inertial)
                dv_mag = v_mag(dv_vector)

                # 5. Vector Freezing
                if dv_mag > 2.0:
                    u_desired = v_normalize(dv_vector)
                    self.u_smoothed = u_desired
                else:
                    u_desired = (
                        self.u_smoothed
                        if self.u_smoothed
                        else v_normalize(dv_vector)
                    )

                self.vessel.auto_pilot.target_roll = 0.0
                self.vessel.auto_pilot.target_direction = u_desired

                # Update Debug Lines
                self.dv_line.start = r_current_inertial
                self.dv_line.end = v_add(
                    r_current_inertial, v_scale(u_desired, 50.0)
                )
                self.vel_line.start = r_current_inertial
                self.vel_line.end = v_add(
                    r_current_inertial,
                    v_scale(v_normalize(v_current_inertial), 50.0),
                )

                ap_error = self.vessel.auto_pilot.error
                _context.debug_info["AP Error"] = ap_error
                _context.debug_info["dV Mag"] = dv_mag
                _context.debug_info["T-Go"] = time_to_go

                # 6. Throttle Profiling
                throttle_threshold = 50.0
                if ap_error < 10.0:
                    if dv_mag > throttle_threshold:
                        self.control.throttle = 1.0
                    else:
                        throttle = max(0.20, dv_mag / throttle_threshold)
                        self.control.throttle = throttle

                    if dv_mag <= 0.5:
                        self.control.throttle = 0.0
                        self.vessel.auto_pilot.engaged = False
                        self.burn_complete = True
                        _context.log("Boostback burn complete. Coasting.")
                        self.dv_line.visible = False
                        self.vel_line.visible = False
                else:
                    self.control.throttle = (
                        0.02 if not self.control.rcs else 0.0
                    )
            else:
                self.control.throttle = 0.0
                self.burn_complete = True
                self.dv_line.visible = False
                self.vel_line.visible = False

        return None

    @override
    def on_exit(self, _context: BoosterContext) -> None:
        self.dv_line.remove()
        self.vel_line.remove()
        _context.debug_info.clear()


class AeroGuidanceState(State[BoosterContext]):
    @override
    def on_enter(self, _context: BoosterContext) -> None:
        print("Atmospheric Entry: Closed-Loop Guidance")
        self.vessel = _context.active_vessel
        self.control = self.vessel.control
        self.body = self.vessel.orbit.body
        self.ref_frame = self.body.reference_frame

        self.control.brakes = True
        self.vessel.auto_pilot.engaged = True
        self.vessel.auto_pilot.reference_frame = self.ref_frame

        # Lock the roll axis to prevent pitch/yaw aerodynamic coupling
        self.vessel.auto_pilot.target_roll = 0.0

        self.N = 3.0  # Navigation constant for PN

        # FAR Aerodynamic Limits
        self.max_aoa_deg = 15.0  # Absolute maximum commanded Angle of Attack
        self.guidance_gain = (
            0.02  # Sensitivity multiplier for the guidance command
        )

    @override
    def tick(
        self, _context: BoosterContext
    ) -> type[State[BoosterContext]] | None:
        if _context.altitude < 10000.0:
            return LandingState

        r_current = _context.position
        v_current = _context.velocity

        r_target = self.body.surface_position(
            _context.target_lat, _context.target_lon, self.ref_frame
        )

        dist = v_mag(v_sub(r_target, r_current))
        speed = v_mag(v_current)
        t_go = dist / speed if speed > 0 else 999.0

        g_mag = self.body.surface_gravity
        g_vec = v_scale(v_normalize(r_current), -g_mag)

        kinematic_pos = v_add(r_current, v_scale(v_current, t_go))
        gravity_drop = v_scale(g_vec, 0.5 * t_go**2)
        r_predicted = v_add(kinematic_pos, gravity_drop)

        zem = v_sub(r_target, r_predicted)

        # Raw Proportional Navigation Command
        a_cmd = v_scale(zem, self.N / (t_go**2))

        retrograde = v_normalize(v_scale(v_current, -1.0))

        # Project a_cmd onto the plane orthogonal to the retrograde vector
        # This ensures guidance only steers cross-track, not along the velocity axis
        a_cmd_proj_mag = v_dot(a_cmd, retrograde)
        a_cmd_ortho = v_sub(a_cmd, v_scale(retrograde, a_cmd_proj_mag))

        # Scale the orthogonal acceleration into a physical steering deflection
        steer_offset = v_scale(a_cmd_ortho, self.guidance_gain)

        # Clamp the steering deflection to obey the maximum Angle of Attack limit
        offset_mag = v_mag(steer_offset)
        max_tan = math.tan(math.radians(self.max_aoa_deg))

        if offset_mag > max_tan:
            steer_offset = v_scale(steer_offset, max_tan / offset_mag)

        # The commanded AoA (in degrees) for the TUI display
        actual_commanded_aoa = math.degrees(math.atan(v_mag(steer_offset)))
        _context.debug_info["Cmd AoA"] = actual_commanded_aoa

        steer_dir = v_normalize(v_add(retrograde, steer_offset))
        self.vessel.auto_pilot.target_direction = steer_dir

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

        # Hardcoded buffer for engine spool-up (e.g., 1.5 seconds)
        spool_up_time = 1.5
        suicide_burn_dist += speed * spool_up_time

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
