from __future__ import annotations

import krpc
from krpc.client import Client
from krpc.services.spacecenter import Vessel

from src.kerbal.coordinates import TopocentricFrame


def get_active_vessel(program_name: str = "program") -> tuple[Client, Vessel]:
    conn = krpc.connect(name=program_name)
    vessel = conn.space_center.active_vessel

    if vessel is None:
        raise ValueError("No active vessel found.")

    return conn, vessel


__all__ = ["TopocentricFrame", "get_active_vessel"]
