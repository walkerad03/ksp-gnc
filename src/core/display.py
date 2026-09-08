import curses
import math
import time
from typing import Any


def _render_tui(stdscr: Any, fsm: Any, context: Any) -> None:
    # Initialize curses settings
    curses.curs_set(0)
    stdscr.nodelay(True)

    # Define color pairs if terminal supports them
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    while True:
        # Allow graceful exit
        if stdscr.getch() == ord("q"):
            break

        # Advance the state machine
        fsm.tick()

        # Clear screen for next frame
        stdscr.erase()

        # Extract vessel & telemetry
        vessel = context.active_vessel
        body = vessel.orbit.body
        ref_frame = body.reference_frame

        state_name = type(fsm.current_state).__name__
        alt = context.altitude
        vel = context.velocity
        pos = context.position

        # 1. Kinematics
        pos_mag = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        up_vec = (
            (pos[0] / pos_mag, pos[1] / pos_mag, pos[2] / pos_mag)
            if pos_mag > 0
            else (0.0, 1.0, 0.0)
        )
        vert_vel = vel[0] * up_vec[0] + vel[1] * up_vec[1] + vel[2] * up_vec[2]
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

        # 2. Propulsion
        thrust = vessel.control.throttle
        mass = vessel.mass
        avail_thrust = vessel.available_thrust
        g_mag = body.surface_gravity
        twr = (avail_thrust / mass) / g_mag if (mass > 0 and g_mag > 0) else 0.0

        # 3. Ballistic Impact & Targeting Calculation
        tgt_lat = context.target_lat
        tgt_lon = context.target_lon
        tgt_alt = context.target_alt

        h_diff = max(0.0, alt - tgt_alt)
        discriminant = max(0.0, vert_vel**2 + 2.0 * g_mag * h_diff)
        tof = (vert_vel + math.sqrt(discriminant)) / g_mag if g_mag > 0 else 0.0

        # Predicted impact vector in body rotating frame
        g_vec = (-up_vec[0] * g_mag, -up_vec[1] * g_mag, -up_vec[2] * g_mag)
        r_pred = (
            pos[0] + vel[0] * tof + 0.5 * g_vec[0] * (tof**2),
            pos[1] + vel[1] * tof + 0.5 * g_vec[1] * (tof**2),
            pos[2] + vel[2] * tof + 0.5 * g_vec[2] * (tof**2),
        )

        # Convert to Geographic Coordinates
        pred_lat = body.latitude_at_position(r_pred, ref_frame)
        pred_lon = body.longitude_at_position(r_pred, ref_frame)

        # Calculate Total Error (Degrees & Ground Meters)
        r_target = body.surface_position(tgt_lat, tgt_lon, ref_frame)
        r_pred_mag = math.sqrt(r_pred[0] ** 2 + r_pred[1] ** 2 + r_pred[2] ** 2)
        r_tgt_mag = math.sqrt(
            r_target[0] ** 2 + r_target[1] ** 2 + r_target[2] ** 2
        )

        dot_norm = (
            (
                r_pred[0] * r_target[0]
                + r_pred[1] * r_target[1]
                + r_pred[2] * r_target[2]
            )
            / (r_pred_mag * r_tgt_mag)
            if (r_pred_mag > 0 and r_tgt_mag > 0)
            else 1.0
        )
        cos_angle = max(-1.0, min(1.0, dot_norm))
        error_deg = math.degrees(math.acos(cos_angle))

        planet_radius = body.equatorial_radius + tgt_alt
        error_meters = math.radians(error_deg) * planet_radius

        # --- Render UI ---
        height, width = stdscr.getmaxyx()

        # Header
        stdscr.addstr(
            1, 2, "GNC TELEMETRY DISPLAY", curses.A_BOLD | curses.color_pair(2)
        )
        stdscr.addstr(1, max(2, width - 20), "Press 'q' to quit")
        stdscr.hline(2, 2, curses.ACS_HLINE, max(0, width - 4))

        # Primary Flight Data
        stdscr.addstr(
            4,
            2,
            f"Active State : {state_name}",
            curses.A_BOLD | curses.color_pair(1),
        )
        stdscr.addstr(6, 2, f"Altitude     : {alt:8.2f} m")
        stdscr.addstr(7, 2, f"Vert Vel     : {vert_vel:8.2f} m/s")
        stdscr.addstr(8, 2, f"Speed        : {speed:8.2f} m/s")

        # Propulsion & Control
        stdscr.addstr(6, 40, f"Throttle     : {thrust * 100:6.1f} %")
        stdscr.addstr(7, 40, f"Max TWR      : {twr:6.2f}")

        # Targeting & Guidance Section
        stdscr.hline(10, 2, curses.ACS_HLINE, max(0, width - 4))
        stdscr.addstr(11, 2, "TARGETING & NAVIGATION", curses.color_pair(2))

        stdscr.addstr(
            13, 2, f"Target Coords: Lat {tgt_lat:8.4f}° | Lon {tgt_lon:8.4f}°"
        )
        stdscr.addstr(
            14, 2, f"Pred Landing : Lat {pred_lat:8.4f}° | Lon {pred_lon:8.4f}°"
        )

        # Color code error: Green < 100m, Yellow < 1000m, Red >= 1000m
        err_color = (
            curses.color_pair(1)
            if error_meters < 100.0
            else (
                curses.color_pair(4)
                if error_meters < 1000.0
                else curses.color_pair(3)
            )
        )
        stdscr.addstr(
            15,
            2,
            f"Target Error : {error_deg:8.4f}° ({error_meters:8.2f} m)",
            err_color,
        )

        # Debug Metrics
        stdscr.hline(17, 2, curses.ACS_HLINE, max(0, width - 4))
        stdscr.addstr(18, 2, "DEBUG METRICS", curses.color_pair(2))
        row = 20
        for k, v in context.debug_info.items():
            stdscr.addstr(row, 2, f"{k:<12}: {v:8.2f}")
            row += 1

        # System Logs
        stdscr.hline(row + 1, 2, curses.ACS_HLINE, max(0, width - 4))
        stdscr.addstr(row + 2, 2, "SYSTEM LOGS", curses.color_pair(2))
        for idx, log_msg in enumerate(context.logs):
            if "error" in log_msg.lower():
                stdscr.addstr(
                    row + 4 + idx, 2, f"> {log_msg}", curses.color_pair(3)
                )
            else:
                stdscr.addstr(row + 4 + idx, 2, f"> {log_msg}")

        stdscr.refresh()
        time.sleep(0.05)


def run_flight_loop(fsm: Any, context: Any) -> None:
    """Wrapper to initialize and cleanup the curses environment safely."""
    curses.wrapper(_render_tui, fsm, context)
