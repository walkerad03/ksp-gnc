"""
Static Launchpad Actuator Verification Script.

Run this script while a rocket is sitting on the launchpad in KSP.
It will verify:
1. Connecting to kRPC and obtaining active vessel telemetry.
2. Constructing the local TopocentricFrame at the pad.
3. Cycling gimbal steering through cardinal directions (Vertical -> East -> North -> West -> South -> Vertical).
4. (Optional) Brief throttle pulse test if enabled.
5. Safe disengagement and return of vessel controls to default.

Usage:
    uv run python scripts/test_actuator_on_pad.py
"""

from __future__ import annotations

import time

import numpy as np

from src.kerbal import TopocentricFrame, get_active_vessel
from src.kerbal.actuation import ActuationBridge
from src.kerbal.telemetry import TelemetryBridge


def main() -> None:
    print("==================================================")
    print("  KSP-GNC Launchpad Actuator Verification Test    ")
    print("==================================================")

    try:
        conn, vessel = get_active_vessel("Launchpad-Actuator-Test")
    except Exception as e:
        print(f"\n[ERROR] Could not connect to kRPC: {e}")
        print(
            "Please make sure Kerbal Space Program is running with kRPC server started."
        )
        return

    body = vessel.orbit.body

    # 1. Query vessel coordinates on the pad
    vessel_lat = vessel.flight().latitude
    vessel_lon = vessel.flight().longitude
    vessel_alt = vessel.flight().surface_altitude
    print(f"\nVessel: {vessel.name}")
    print(
        f"Location: Lat {vessel_lat:.4f}°, Lon {vessel_lon:.4f}°, Alt {vessel_alt:.1f} m"
    )

    # 2. Build local Topocentric Frame
    frame = TopocentricFrame.from_krpc(body, vessel_lat, vessel_lon, vessel_alt)
    print("Topocentric Frame initialized at pad center.")

    # 3. Initialize Bridges
    telemetry = TelemetryBridge(conn, vessel, frame)
    actuation = ActuationBridge(
        vessel, frame, min_throttle_cutoff=0.05, max_tilt_angle_deg=20.0
    )

    state = telemetry.update()
    print("Telemetry Status:")
    print(f"  Vessel Mass:     {state.mass:.1f} kg")
    print(f"  Available Thrust: {telemetry.available_thrust / 1000.0:.1f} kN")
    print(f"  Pad Offset ENU:  {state.position}")

    avail_thrust = telemetry.available_thrust
    if avail_thrust <= 0.0:
        print(
            "[WARNING] Available thrust is 0 kN. Engines may be staged off or deactivated."
        )
        avail_thrust = (
            100_000.0  # Fallback thrust magnitude for attitude testing
        )

    test_thrust_mag = avail_thrust * 0.40  # 40% equivalent thrust vector

    print("\n--- Starting Gimbal Orientation Sequence (Zero Throttle) ---")
    print("Watch your rocket's engine gimbal and attitude on screen.\n")

    steps = [
        ("1. Hold Vertical Up", np.array([0.0, 0.0, 1.0])),
        (
            "2. Tilt 12° East",
            np.array([np.sin(np.deg2rad(12.0)), 0.0, np.cos(np.deg2rad(12.0))]),
        ),
        (
            "3. Tilt 12° North",
            np.array([0.0, np.sin(np.deg2rad(12.0)), np.cos(np.deg2rad(12.0))]),
        ),
        (
            "4. Tilt 12° West",
            np.array(
                [-np.sin(np.deg2rad(12.0)), 0.0, np.cos(np.deg2rad(12.0))]
            ),
        ),
        (
            "5. Tilt 12° South",
            np.array(
                [0.0, -np.sin(np.deg2rad(12.0)), np.cos(np.deg2rad(12.0))]
            ),
        ),
        ("6. Return to Vertical", np.array([0.0, 0.0, 1.0])),
    ]

    try:
        for step_name, unit_vec_enu in steps:
            T_enu = unit_vec_enu * test_thrust_mag
            cmd = actuation.compute_command(
                T_0_enu=T_enu,
                Gamma_0=0.0,  # Keep throttle at 0% for pure steering check
                available_thrust=avail_thrust,
            )
            actuation.apply(cmd)
            print(
                f"  -> {step_name:<25} | Tilt: {cmd.tilt_angle_deg:4.1f}° | Direction: {cmd.direction_ecef}"
            )
            time.sleep(2.5)

        print("\n[SUCCESS] Gimbal verification sequence completed.")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping test early...")
    finally:
        print("\nCleaning up...")
        actuation.disengage()
        telemetry.close()
        print("Autopilot disengaged. Throttle reset to 0. Streams closed.")


if __name__ == "__main__":
    main()
