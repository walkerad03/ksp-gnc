from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.kerbal.coordinates import TopocentricFrame


@dataclass(frozen=True)
class ActuationCommand:
    throttle: float
    direction_ecef: tuple[float, float, float]
    thrust_magnitude: float
    tilt_angle_deg: float
    is_engine_active: bool


class ActuationBridge:
    def __init__(
        self,
        vessel: Any,
        frame: TopocentricFrame,
        min_throttle_cutoff: float = 0.05,
        max_tilt_angle_deg: float = 25.0,
    ) -> None:
        """
        vessel: kRPC Vessel instance
        frame: Topocentric landing pad frame
        min_throttle_cutoff: Minimum throttle threshold below which engine is shut off (coast)
        max_tilt_angle_deg: Hard safety ceiling on thrust pointing deviation from vertical
        """
        self.vessel: Any = vessel
        self.frame: TopocentricFrame = frame
        self.min_throttle_cutoff: float = min_throttle_cutoff
        self.max_tilt_angle_deg: float = max_tilt_angle_deg

    def compute_command(
        self,
        T_0_enu: npt.NDArray[np.float64] | Sequence[float],
        Gamma_0: float,
        available_thrust: float,
    ) -> ActuationCommand:
        T_vec = np.asarray(T_0_enu, dtype=np.float64)
        thrust_mag = max(0.0, float(Gamma_0))

        if available_thrust <= 0.0 or thrust_mag <= 0.0:
            throttle = 0.0
            is_active = False
        else:
            raw_throttle = thrust_mag / available_thrust
            if raw_throttle < self.min_throttle_cutoff:
                throttle = 0.0
                is_active = False
            else:
                throttle = float(np.clip(raw_throttle, 0.0, 1.0))
                is_active = True

        norm_T = float(np.linalg.norm(T_vec))
        if norm_T > 1e-3:
            unit_T = T_vec / norm_T
            cos_tilt = float(np.clip(unit_T[2], -1.0, 1.0))
            tilt_deg = float(np.degrees(np.arccos(cos_tilt)))

            if tilt_deg > self.max_tilt_angle_deg:
                max_rad = np.deg2rad(self.max_tilt_angle_deg)
                horiz_vec = unit_T[:2]
                norm_horiz = float(np.linalg.norm(horiz_vec))
                if norm_horiz > 1e-6:
                    horiz_clamped = (horiz_vec / norm_horiz) * np.sin(max_rad)
                    unit_T = np.array(
                        [horiz_clamped[0], horiz_clamped[1], np.cos(max_rad)],
                        dtype=np.float64,
                    )
                tilt_deg = self.max_tilt_angle_deg
        else:
            unit_T = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            tilt_deg = 0.0

        dir_ecef = self.frame.enu_to_ecef_direction(unit_T)
        direction_tuple = (
            float(dir_ecef[0]),
            float(dir_ecef[1]),
            float(dir_ecef[2]),
        )

        return ActuationCommand(
            throttle=throttle,
            direction_ecef=direction_tuple,
            thrust_magnitude=thrust_mag,
            tilt_angle_deg=tilt_deg,
            is_engine_active=is_active,
        )

    def apply(self, command: ActuationCommand) -> None:
        """Applies computed throttle and attitude commands to the kRPC vessel."""
        ref_frame = self.vessel.orbit.body.reference_frame

        self.vessel.control.sas = False

        self.vessel.auto_pilot.reference_frame = ref_frame
        self.vessel.auto_pilot.target_direction = command.direction_ecef
        self.vessel.auto_pilot.target_roll = float("nan")
        if not self.vessel.auto_pilot.engaged:
            self.vessel.auto_pilot.engaged = True

        self.vessel.control.throttle = command.throttle

    def disengage(self) -> None:
        """Safely zeroes throttle and disengages autopilot."""
        self.vessel.control.throttle = 0.0
        if self.vessel.auto_pilot.engaged:
            self.vessel.auto_pilot.engaged = False
