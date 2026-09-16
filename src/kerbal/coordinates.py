from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TopocentricFrame:
    """
    Local topocentric frame centered at target.
    +X = East, +Y = North, +Z = Up
    """

    pad_position_ecef: npt.NDArray[
        np.float64
    ]  # (3,) position of pad in body rotating frame
    R_ecef_to_enu: npt.NDArray[
        np.float64
    ]  # (3, 3) rotation matrix from ECEF to ENU

    @property
    def R_enu_to_ecef(self) -> npt.NDArray[np.float64]:
        """Rotation matrix from ENU to ECEF (transpose of orthogonal matrix)."""
        return self.R_ecef_to_enu.T

    @classmethod
    def from_pad_and_north(
        cls,
        pad_ecef: npt.NDArray[np.float64] | tuple[float, float, float],
        north_ecef: npt.NDArray[np.float64] | tuple[float, float, float],
    ) -> TopocentricFrame:
        """
        Constructs the local ENU frame from the pad position and a reference
        vector/point indicating local North.
        """
        r_pad = np.asarray(pad_ecef, dtype=np.float64)
        r_north_ref = np.asarray(north_ecef, dtype=np.float64)

        norm_pad = float(np.linalg.norm(r_pad))
        assert norm_pad > 0.0, "Pad position cannot be at planet center"
        u_up = r_pad / norm_pad

        delta_north = r_north_ref - r_pad
        v_north = delta_north - float(np.dot(delta_north, u_up)) * u_up
        norm_north = float(np.linalg.norm(v_north))
        assert norm_north > 0.0, (
            "North reference cannot be collinear with pad vertical"
        )
        u_north = v_north / norm_north

        u_east = np.cross(u_north, u_up)
        u_east = u_east / np.linalg.norm(u_east)

        u_north = np.cross(u_up, u_east)
        u_north = u_north / np.linalg.norm(u_north)

        R = np.vstack([u_east, u_north, u_up])

        return cls(pad_position_ecef=r_pad, R_ecef_to_enu=R)

    @classmethod
    def from_lat_lon(
        cls,
        lat_deg: float,
        lon_deg: float,
        altitude: float = 0.0,
        body_radius: float = 600_000.0,
    ) -> TopocentricFrame:
        """
        Analytical spherical construction of an ENU frame without requiring kRPC.
        Assumes standard spherical planet coordinates (Z is rotation axis, X is lon 0).
        """
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)
        r = body_radius + altitude

        pad_ecef = np.array(
            [
                r * np.cos(lat_rad) * np.cos(lon_rad),
                r * np.cos(lat_rad) * np.sin(lon_rad),
                r * np.sin(lat_rad),
            ],
            dtype=np.float64,
        )

        u_east = np.array(
            [-np.sin(lon_rad), np.cos(lon_rad), 0.0], dtype=np.float64
        )
        u_north = np.array(
            [
                -np.sin(lat_rad) * np.cos(lon_rad),
                -np.sin(lat_rad) * np.sin(lon_rad),
                np.cos(lat_rad),
            ],
            dtype=np.float64,
        )
        u_up = np.array(
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ],
            dtype=np.float64,
        )

        R = np.vstack([u_east, u_north, u_up])
        return cls(pad_position_ecef=pad_ecef, R_ecef_to_enu=R)

    @classmethod
    def from_krpc(
        cls,
        body: Any,
        target_lat: float,
        target_lon: float,
        target_alt: float = 0.0,
    ) -> TopocentricFrame:
        """
        Constructs the frame using live kRPC CelestialBody methods.
        Uses body.reference_frame (the planet's rotating ECEF frame).
        """
        ref_frame = body.reference_frame

        pad_pos_raw = np.array(
            body.surface_position(target_lat, target_lon, ref_frame),
            dtype=np.float64,
        )
        norm_pad = float(np.linalg.norm(pad_pos_raw))
        if norm_pad > 0.0 and target_alt != 0.0:
            pad_ecef = pad_pos_raw * (1.0 + target_alt / norm_pad)
        else:
            pad_ecef = pad_pos_raw

        north_sample = np.array(
            body.surface_position(target_lat + 0.01, target_lon, ref_frame),
            dtype=np.float64,
        )

        return cls.from_pad_and_north(
            pad_ecef=pad_ecef, north_ecef=north_sample
        )

    def ecef_to_enu_position(
        self, pos_ecef: npt.NDArray[np.float64] | tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Transforms a 3D position vector from ECEF to local landing pad ENU."""
        pos = np.asarray(pos_ecef, dtype=np.float64)
        return self.R_ecef_to_enu @ (pos - self.pad_position_ecef)

    def ecef_to_enu_velocity(
        self, vel_ecef: npt.NDArray[np.float64] | tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Transforms a 3D velocity vector from ECEF to local landing pad ENU."""
        vel = np.asarray(vel_ecef, dtype=np.float64)
        return self.R_ecef_to_enu @ vel

    def enu_to_ecef_position(
        self, pos_enu: npt.NDArray[np.float64] | tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Transforms a 3D position vector from local landing pad ENU back to ECEF."""
        pos = np.asarray(pos_enu, dtype=np.float64)
        return self.pad_position_ecef + (self.R_enu_to_ecef @ pos)

    def enu_to_ecef_velocity(
        self, vel_enu: npt.NDArray[np.float64] | tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Transforms a 3D velocity vector from local landing pad ENU back to ECEF."""
        vel = np.asarray(vel_enu, dtype=np.float64)
        return self.R_enu_to_ecef @ vel

    def enu_to_ecef_direction(
        self, vec_enu: npt.NDArray[np.float64] | tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """
        Transforms a control thrust direction vector from ENU to ECEF
        for commanding vessel.auto_pilot.target_direction in body.reference_frame.
        """
        vec = np.asarray(vec_enu, dtype=np.float64)
        norm_vec = float(np.linalg.norm(vec))
        if norm_vec == 0.0:
            # Fallback to local vertical (Up) in ECEF
            return self.R_enu_to_ecef[:, 2].copy()
        unit_vec_enu = vec / norm_vec
        return self.R_enu_to_ecef @ unit_vec_enu
