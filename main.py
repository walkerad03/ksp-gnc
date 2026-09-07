import time

from src.core.booster_states import BoosterContext, LandedState, LaunchState
from src.core.fsm import StateMachine
from src.kerbal import get_active_vessel


def main():
    program_name = "Booster Landing"
    conn, vessel = get_active_vessel(program_name)
    ref_frame = vessel.orbit.body.reference_frame

    flight_info = vessel.flight(ref_frame)
    alt_stream = conn.add_stream(getattr, flight_info, "surface_altitude")
    vel_stream = conn.add_stream(getattr, flight_info, "velocity")
    drag_stream = conn.add_stream(getattr, flight_info, "drag")

    ctx = BoosterContext(
        active_vessel=vessel,
        _altitude_stream=alt_stream,
        _velocity_stream=vel_stream,
        _drag_stream=drag_stream,
    )
    fsm = StateMachine(LaunchState, ctx)

    while fsm.current_state != LandedState:
        fsm.tick()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
