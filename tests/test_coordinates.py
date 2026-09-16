from __future__ import annotations

import numpy as np
import pytest

from src.guidance.models.aerodynamics import AerodynamicModel
from src.guidance.models.gravity import KERBIN_RADIUS, SphericalGravityModel
from src.guidance.ocp import LandingOCP, VehicleLimits
from src.guidance.trajectory import ReferenceTrajectory
from src.kerbal.coordinates import TopocentricFrame

# ==============================================================================
# Mock kRPC SpaceCenter Objects (matching kRPC Python API specification)
# https://krpc.github.io/krpc/latest/python/api/space-center/
# ==============================================================================


class MockReferenceFrame:
    """Mock for krpc.services.spacecenter.ReferenceFrame."""

    def __init__(self, name: str) -> None:
        self.name = name


class MockAutoPilot:
    """Mock for krpc.services.spacecenter.AutoPilot."""

    def __init__(self) -> None:
        self.reference_frame: MockReferenceFrame | None = None
        self.target_direction: tuple[float, float, float] | None = None
        self.target_roll: float = 0.0
        self.engaged: bool = False

    def engage(self) -> None:
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False


class MockControl:
    """Mock for krpc.services.spacecenter.Control."""

    def __init__(self) -> None:
        self.throttle: float = 0.0
        self.rcs: bool = False
        self.sas: bool = False


class MockCelestialBody:
    """
    Mock for krpc.services.spacecenter.CelestialBody.
    Calculates exact spherical coordinates in rotating reference frame (ECEF).
    """

    def __init__(
        self,
        name: str = "Kerbin",
        radius: float = 600_000.0,
        gravitational_parameter: float = 3.5316e12,
    ) -> None:
        self.name = name
        self.equatorial_radius = radius
        self.gravitational_parameter = gravitational_parameter
        self.reference_frame = MockReferenceFrame(f"{name}.reference_frame")
        self.non_rotating_reference_frame = MockReferenceFrame(
            f"{name}.non_rotating_reference_frame"
        )

    def surface_position(
        self,
        latitude: float,
        longitude: float,
        target_frame: MockReferenceFrame,
    ) -> tuple[float, float, float]:
        """
        Calculates position vector on body surface in body.reference_frame.
        Standard spherical coordinate mapping: Z is rotation axis, X is lon 0.
        """
        lat_rad = np.deg2rad(latitude)
        lon_rad = np.deg2rad(longitude)
        r = self.equatorial_radius

        x = float(r * np.cos(lat_rad) * np.cos(lon_rad))
        y = float(r * np.cos(lat_rad) * np.sin(lon_rad))
        z = float(r * np.sin(lat_rad))
        return (x, y, z)


class MockVessel:
    """Mock for krpc.services.spacecenter.Vessel."""

    def __init__(
        self,
        body: MockCelestialBody,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        mass: float = 15_000.0,
        available_thrust: float = 450_000.0,
    ) -> None:
        self.orbit_body = body
        self._position = position
        self._velocity = velocity
        self.mass = mass
        self.available_thrust = available_thrust
        self.auto_pilot = MockAutoPilot()
        self.control = MockControl()

    def position(
        self, reference_frame: MockReferenceFrame
    ) -> tuple[float, float, float]:
        return self._position

    def velocity(
        self, reference_frame: MockReferenceFrame
    ) -> tuple[float, float, float]:
        return self._velocity

    def set_state(
        self,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> None:
        self._position = position
        self._velocity = velocity


# ==============================================================================
# Unit Tests: Coordinate Transformations
# ==============================================================================


class TestTopocentricFrameUnit:
    @pytest.mark.parametrize(
        ("lat_deg", "lon_deg", "alt"),
        [
            (0.0, 0.0, 0.0),  # Equator at prime meridian
            (0.0, 90.0, 50.0),  # Equator at 90 deg East
            (-0.0972, -74.5577, 68.0),  # Kerbal Space Center launchpad
            (45.0, -120.0, 150.0),  # Mid-latitude
            (-80.0, 45.0, 0.0),  # Near south pole
        ],
    )
    def test_rotation_matrix_orthonormality_and_handedness(
        self, lat_deg: float, lon_deg: float, alt: float
    ) -> None:
        """
        Verifies that R is strictly orthogonal: R @ R.T = I,
        and right-handed: det(R) = +1.0.
        """
        frame = TopocentricFrame.from_lat_lon(lat_deg, lon_deg, alt)
        R = frame.R_ecef_to_enu

        # R @ R.T == I
        identity_approx = R @ R.T
        np.testing.assert_allclose(identity_approx, np.eye(3), atol=1e-12)

        # Transpose property
        np.testing.assert_allclose(frame.R_enu_to_ecef, R.T, atol=1e-12)

        # Determinant must be +1 for a proper rotation (SO(3))
        det = float(np.linalg.det(R))
        np.testing.assert_allclose(det, 1.0, atol=1e-12)

    def test_basis_vectors_alignment(self) -> None:
        """
        Verifies that points placed along the pad's local East, North, and Up
        axes in ECEF correctly map to the canonical basis in ENU.
        """
        # KSC Launchpad frame
        frame = TopocentricFrame.from_lat_lon(
            lat_deg=-0.0972, lon_deg=-74.5577, altitude=68.0
        )
        r_pad = frame.pad_position_ecef

        # Extract basis vectors in ECEF (columns of R_enu_to_ecef)
        u_east_ecef = frame.R_enu_to_ecef[:, 0]
        u_north_ecef = frame.R_enu_to_ecef[:, 1]
        u_up_ecef = frame.R_enu_to_ecef[:, 2]

        # 1. Pad position should map to origin in ENU
        r_pad_enu = frame.ecef_to_enu_position(r_pad)
        np.testing.assert_allclose(r_pad_enu, [0.0, 0.0, 0.0], atol=1e-9)

        # 2. Point 500m up
        r_up_ecef = r_pad + 500.0 * u_up_ecef
        np.testing.assert_allclose(
            frame.ecef_to_enu_position(r_up_ecef), [0.0, 0.0, 500.0], atol=1e-9
        )

        # 3. Point 250m north
        r_north_ecef = r_pad + 250.0 * u_north_ecef
        np.testing.assert_allclose(
            frame.ecef_to_enu_position(r_north_ecef),
            [0.0, 250.0, 0.0],
            atol=1e-9,
        )

        # 4. Point 150m east
        r_east_ecef = r_pad + 150.0 * u_east_ecef
        np.testing.assert_allclose(
            frame.ecef_to_enu_position(r_east_ecef),
            [150.0, 0.0, 0.0],
            atol=1e-9,
        )

    def test_round_trip_position_and_velocity_invariance(self) -> None:
        """Verifies lossless round-trip transformations between ENU and ECEF."""
        frame = TopocentricFrame.from_lat_lon(
            lat_deg=15.0, lon_deg=45.0, altitude=100.0
        )

        # Arbitrary vessel position in ENU: (East: 300m, North: -150m, Up: 2500m)
        enu_pos_orig = np.array([300.0, -150.0, 2500.0], dtype=np.float64)
        ecef_pos = frame.enu_to_ecef_position(enu_pos_orig)
        enu_pos_recovered = frame.ecef_to_enu_position(ecef_pos)
        np.testing.assert_allclose(enu_pos_recovered, enu_pos_orig, atol=1e-10)

        # Arbitrary velocity in ENU: (vx: -15m/s, vy: 5m/s, vz: -90m/s)
        enu_vel_orig = np.array([-15.0, 5.0, -90.0], dtype=np.float64)
        ecef_vel = frame.enu_to_ecef_velocity(enu_vel_orig)
        enu_vel_recovered = frame.ecef_to_enu_velocity(ecef_vel)
        np.testing.assert_allclose(enu_vel_recovered, enu_vel_orig, atol=1e-10)

    def test_enu_to_ecef_direction(self) -> None:
        """
        Verifies conversion of commanded thrust vectors in ENU
        to normalized unit direction vectors in ECEF.
        """
        frame = TopocentricFrame.from_lat_lon(lat_deg=0.0, lon_deg=0.0)

        # Commanded straight up thrust (opposing gravity)
        T_up_enu = np.array([0.0, 0.0, 300_000.0])
        dir_ecef = frame.enu_to_ecef_direction(T_up_enu)

        # Norm must be exactly 1.0
        assert np.isclose(float(np.linalg.norm(dir_ecef)), 1.0, atol=1e-12)

        # In equator at lon 0, local Up is [1, 0, 0] in spherical coordinates
        np.testing.assert_allclose(dir_ecef, [1.0, 0.0, 0.0], atol=1e-9)

        # Test zero vector fallback (should default safely to local vertical)
        zero_dir = frame.enu_to_ecef_direction(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(float(np.linalg.norm(zero_dir)), 1.0)
        np.testing.assert_allclose(zero_dir, [1.0, 0.0, 0.0], atol=1e-9)

    def test_invalid_construction_raises(self) -> None:
        """Verifies assertions when given degenerate input geometry."""
        # Zero pad position
        with pytest.raises(AssertionError, match="cannot be at planet center"):
            TopocentricFrame.from_pad_and_north(
                (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            )

        # North reference collinear with pad position (same line from center)
        pad = np.array([0.0, 0.0, 600_000.0])
        north_collinear = np.array([0.0, 0.0, 700_000.0])
        with pytest.raises(AssertionError, match="cannot be collinear"):
            TopocentricFrame.from_pad_and_north(pad, north_collinear)


# ==============================================================================
# Integration Tests: kRPC Data Types & MPC End-to-End Pipeline
# ==============================================================================


class TestKRPCIntegration:
    @pytest.fixture
    def krpc_environment(self) -> tuple[MockCelestialBody, MockVessel]:
        """Creates a mock Kerbin environment and landing booster vessel."""
        body = MockCelestialBody(name="Kerbin", radius=600_000.0)

        # KSC launchpad coordinates
        target_lat = -0.0972
        target_lon = -74.5577
        target_alt = 68.0

        # Position vessel 2,000m above pad with downrange offset (in ENU: X=200m, Y=-100m, Z=2000m)
        frame = TopocentricFrame.from_lat_lon(
            target_lat, target_lon, target_alt, body.equatorial_radius
        )
        pos_enu = np.array([200.0, -100.0, 2000.0], dtype=np.float64)
        vel_enu = np.array([-15.0, 8.0, -80.0], dtype=np.float64)

        pos_ecef = frame.enu_to_ecef_position(pos_enu)
        vel_ecef = frame.enu_to_ecef_velocity(vel_enu)

        # Create vessel with tuple outputs matching kRPC client specifications
        vessel = MockVessel(
            body=body,
            position=(
                float(pos_ecef[0]),
                float(pos_ecef[1]),
                float(pos_ecef[2]),
            ),
            velocity=(
                float(vel_ecef[0]),
                float(vel_ecef[1]),
                float(vel_ecef[2]),
            ),
            mass=15_000.0,
            available_thrust=450_000.0,
        )
        return body, vessel

    def test_from_krpc_construction(
        self, krpc_environment: tuple[MockCelestialBody, MockVessel]
    ) -> None:
        """
        Verifies that TopocentricFrame.from_krpc properly queries body.surface_position
        and builds an accurate topocentric frame.
        """
        body, _ = krpc_environment
        target_lat = -0.0972
        target_lon = -74.5577
        target_alt = 68.0

        # Construct frame using mock kRPC celestial body
        frame = TopocentricFrame.from_krpc(
            body=body,
            target_lat=target_lat,
            target_lon=target_lon,
            target_alt=target_alt,
        )

        # Check pad height
        norm_pad = float(np.linalg.norm(frame.pad_position_ecef))
        expected_radius = body.equatorial_radius + target_alt
        assert np.isclose(norm_pad, expected_radius, atol=1e-3)

        # Determinant must be +1
        det = float(np.linalg.det(frame.R_ecef_to_enu))
        assert np.isclose(det, 1.0, atol=1e-12)

    def test_krpc_tuple_input_handling(
        self, krpc_environment: tuple[MockCelestialBody, MockVessel]
    ) -> None:
        """
        Confirms that TopocentricFrame accepts raw Python 3-tuples
        as returned by vessel.position(...) and vessel.velocity(...) without error.
        """
        body, vessel = krpc_environment
        frame = TopocentricFrame.from_krpc(body, -0.0972, -74.5577, 68.0)

        raw_pos_tuple = vessel.position(body.reference_frame)
        raw_vel_tuple = vessel.velocity(body.reference_frame)

        assert isinstance(raw_pos_tuple, tuple)
        assert isinstance(raw_vel_tuple, tuple)

        pos_enu = frame.ecef_to_enu_position(raw_pos_tuple)
        vel_enu = frame.ecef_to_enu_velocity(raw_vel_tuple)

        assert isinstance(pos_enu, np.ndarray)
        assert pos_enu.shape == (3,)
        # Should be approximately 2,000m altitude
        assert np.isclose(pos_enu[2], 2000.0, atol=1.0)
        assert np.isclose(vel_enu[2], -80.0, atol=1.0)

    def test_full_pipeline_krpc_telemetry_to_mpc_to_autopilot(
        self, krpc_environment: tuple[MockCelestialBody, MockVessel]
    ) -> None:
        """
        Validates the entire end-to-end control cycle:
        1. Query vessel telemetry (tuples in body.reference_frame).
        2. Transform to local ENU coordinates.
        3. Solve MPC SOCP for optimal thrust vector T_0.
        4. Transform T_0 back to ECEF throttle and auto_pilot.target_direction.
        5. Verify that autopilot and throttle commands are valid and decelerate towards target.
        """
        body, vessel = krpc_environment
        frame = TopocentricFrame.from_krpc(body, -0.0972, -74.5577, 68.0)

        # 1. Ingest telemetry from vessel
        r_ecef = vessel.position(body.reference_frame)
        v_ecef = vessel.velocity(body.reference_frame)
        mass = vessel.mass

        # 2. Transform to ENU
        r_enu = frame.ecef_to_enu_position(r_ecef)
        v_enu = frame.ecef_to_enu_velocity(v_ecef)

        # 3. Setup Guidance & Solve OCP
        gravity = SphericalGravityModel(
            planet_center_offset=np.array([0.0, 0.0, KERBIN_RADIUS])
        )
        aero = AerodynamicModel(cd_area=4.0)
        limits = VehicleLimits(
            thrust_min=40_000.0,
            thrust_max=vessel.available_thrust,
            isp=310.0,
            max_gimbal_angle_deg=25.0,
            glideslope_angle_deg=15.0,
            min_mass=7_500.0,
        )
        ocp = LandingOCP(limits, gravity, aero, solver="CLARABEL")

        r_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v_target = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        m_dry = 8_500.0

        ref_traj = ReferenceTrajectory.from_straight_line(
            r_enu,
            v_enu,
            mass,
            r_target,
            v_target,
            m_dry,
            duration=22.0,
            n_nodes=20,
        )

        solution = ocp.solve(
            initial_state=(r_enu, v_enu, mass),
            target_state=(r_target, v_target),
            ref_traj=ref_traj,
            enable_glideslope=True,
        )

        assert solution.optimal is True

        # 4. Convert MPC output to kRPC Actuation Commands
        T_0_enu = solution.commanded_thrust
        Gamma_0 = solution.commanded_magnitude

        # Throttle command in [0.0, 1.0]
        throttle_cmd = float(
            np.clip(Gamma_0 / vessel.available_thrust, 0.0, 1.0)
        )

        # Direction command: unit vector in body.reference_frame (ECEF)
        target_dir_ecef = frame.enu_to_ecef_direction(T_0_enu)
        target_dir_tuple = (
            float(target_dir_ecef[0]),
            float(target_dir_ecef[1]),
            float(target_dir_ecef[2]),
        )

        # Apply commands to vessel autopilot and throttle
        vessel.control.throttle = throttle_cmd
        vessel.auto_pilot.reference_frame = body.reference_frame
        vessel.auto_pilot.target_direction = target_dir_tuple
        vessel.auto_pilot.engage()

        # 5. Assertions on Commanded Actuation
        assert 0.0 < vessel.control.throttle <= 1.0
        assert vessel.auto_pilot.engaged is True
        assert vessel.auto_pilot.reference_frame == body.reference_frame

        # The commanded direction in ECEF must be a unit vector
        dir_norm = float(
            np.linalg.norm(np.array(vessel.auto_pilot.target_direction))
        )
        assert np.isclose(dir_norm, 1.0, atol=1e-9)

        # Commanded thrust must have a positive vertical component (opposing gravity)
        u_up_ecef = frame.R_enu_to_ecef[:, 2]
        vertical_projection = float(np.dot(target_dir_ecef, u_up_ecef))
        assert vertical_projection > np.cos(
            np.deg2rad(limits.max_gimbal_angle_deg)
        )
