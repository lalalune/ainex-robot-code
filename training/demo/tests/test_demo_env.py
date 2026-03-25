"""Tests for DemoEnv -- CPU MuJoCo demo environment.

Validates that the environment creates, resets, steps, renders,
and produces telemetry in the correct bridge-compatible format.

Run:
    python -m pytest training/demo/tests/test_demo_env.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_env():
    """Create a DemoEnv with a red ball at (2, 0, 0.05)."""
    from training.mujoco.demo_env import DemoEnv

    env = DemoEnv(
        target_position=(2.0, 0.0, 0.05),
        target_color=(1.0, 0.0, 0.0, 1.0),
        target_size=0.05,
        camera_width=320,
        camera_height=240,
        timestep=0.004,
    )
    yield env
    env.close()


@pytest.fixture
def demo_env_offset():
    """Create a DemoEnv with the ball at a different position."""
    from training.mujoco.demo_env import DemoEnv

    env = DemoEnv(
        target_position=(1.0, 1.0, 0.05),
        camera_width=160,
        camera_height=120,
    )
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Creation / Reset
# ---------------------------------------------------------------------------

class TestCreation:
    def test_creates_successfully(self, demo_env):
        assert demo_env.model is not None
        assert demo_env.data is not None

    def test_reset_returns_telemetry(self, demo_env):
        telemetry = demo_env.reset()
        assert isinstance(telemetry, dict)
        assert "joint_positions" in telemetry
        assert "imu_roll" in telemetry
        assert "imu_pitch" in telemetry
        assert "gyro" in telemetry
        assert "walking" in telemetry
        assert "battery_mv" in telemetry

    def test_reset_clears_step_count(self, demo_env):
        demo_env.step()
        demo_env.step()
        assert demo_env.step_count > 0
        demo_env.reset()
        assert demo_env.step_count == 0


# ---------------------------------------------------------------------------
# Step / Telemetry
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_telemetry(self, demo_env):
        demo_env.reset()
        telemetry = demo_env.step()
        assert isinstance(telemetry, dict)

    def test_step_increments_count(self, demo_env):
        demo_env.reset()
        demo_env.step()
        demo_env.step()
        demo_env.step()
        assert demo_env.step_count == 3

    def test_step_with_joint_targets(self, demo_env):
        demo_env.reset()
        targets = {"r_hip_pitch": -0.3, "l_hip_pitch": 0.3}
        telemetry = demo_env.step(joint_targets=targets)
        assert isinstance(telemetry["joint_positions"], dict)
        # Joint positions dict should include all actuated joints.
        assert len(telemetry["joint_positions"]) > 0

    def test_step_n(self, demo_env):
        demo_env.reset()
        telemetry = demo_env.step_n(10)
        assert demo_env.step_count == 10
        assert isinstance(telemetry, dict)

    def test_telemetry_joint_positions_are_floats(self, demo_env):
        demo_env.reset()
        telemetry = demo_env.step()
        for name, value in telemetry["joint_positions"].items():
            assert isinstance(name, str), f"key {name!r} is not str"
            assert isinstance(value, float), f"value for {name} is not float"

    def test_telemetry_imu_are_floats(self, demo_env):
        demo_env.reset()
        telemetry = demo_env.step()
        assert isinstance(telemetry["imu_roll"], float)
        assert isinstance(telemetry["imu_pitch"], float)

    def test_telemetry_gyro_shape(self, demo_env):
        demo_env.reset()
        telemetry = demo_env.step()
        gyro = telemetry["gyro"]
        assert isinstance(gyro, list)
        assert len(gyro) == 3
        for v in gyro:
            assert isinstance(v, float)

    def test_telemetry_battery(self, demo_env):
        demo_env.reset()
        telemetry = demo_env.step()
        assert isinstance(telemetry["battery_mv"], int)
        assert telemetry["battery_mv"] > 0

    def test_telemetry_walking_flag(self, demo_env):
        demo_env.reset()
        # Before any joint targets, walking should be False.
        telemetry = demo_env.step()
        assert telemetry["walking"] is False

        # After providing targets, walking should be True.
        telemetry = demo_env.step(joint_targets={"r_hip_pitch": -0.1})
        assert telemetry["walking"] is True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_render_ego_shape(self, demo_env):
        demo_env.reset()
        frame = demo_env.render_ego()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (240, 320, 3)

    def test_render_ego_nonzero(self, demo_env):
        demo_env.reset()
        frame = demo_env.render_ego()
        # The frame should not be all black (there is a floor, skybox, robot).
        assert frame.sum() > 0

    def test_render_ego_different_size(self, demo_env_offset):
        demo_env_offset.reset()
        frame = demo_env_offset.render_ego()
        assert frame.shape == (120, 160, 3)


# ---------------------------------------------------------------------------
# Target / Robot positions
# ---------------------------------------------------------------------------

class TestPositions:
    def test_target_position(self, demo_env):
        demo_env.reset()
        pos = demo_env.get_target_position()
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)
        np.testing.assert_allclose(pos, [2.0, 0.0, 0.05], atol=0.01)

    def test_target_position_offset(self, demo_env_offset):
        demo_env_offset.reset()
        pos = demo_env_offset.get_target_position()
        np.testing.assert_allclose(pos[:2], [1.0, 1.0], atol=0.01)

    def test_robot_position(self, demo_env):
        demo_env.reset()
        pos = demo_env.get_robot_position()
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)
        # Robot starts near origin.
        assert abs(pos[0]) < 0.5
        assert abs(pos[1]) < 0.5

    def test_robot_yaw(self, demo_env):
        demo_env.reset()
        yaw = demo_env.get_robot_yaw()
        assert isinstance(yaw, float)
        # Initial yaw should be near zero.
        assert abs(yaw) < 0.5

    def test_distance_to_target(self, demo_env):
        demo_env.reset()
        dist = demo_env.distance_to_target()
        assert isinstance(dist, float)
        # Ball at (2, 0), robot near origin -> distance ~ 2.0
        assert 1.5 < dist < 2.5

    def test_bearing_to_target(self, demo_env):
        demo_env.reset()
        bearing = demo_env.bearing_to_target()
        assert isinstance(bearing, float)
        # Ball is directly ahead (x=2, y=0) -> bearing ~ 0.
        assert abs(bearing) < 0.5


# ---------------------------------------------------------------------------
# is_target_reached
# ---------------------------------------------------------------------------

class TestTargetReached:
    def test_not_reached_initially(self, demo_env):
        demo_env.reset()
        assert demo_env.is_target_reached(threshold=0.3) is False

    def test_reached_with_large_threshold(self, demo_env):
        demo_env.reset()
        # Distance ~ 2m; a threshold of 3m means "reached".
        assert demo_env.is_target_reached(threshold=3.0) is True

    def test_not_reached_with_small_threshold(self, demo_env):
        demo_env.reset()
        assert demo_env.is_target_reached(threshold=0.1) is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_close_is_idempotent(self, demo_env):
        demo_env.close()
        demo_env.close()  # Should not raise.

    def test_joint_names(self, demo_env):
        from training.mujoco import ainex_constants as consts
        assert demo_env.joint_names == consts.ALL_JOINT_NAMES

    def test_default_pose_shape(self, demo_env):
        pose = demo_env.default_pose
        assert isinstance(pose, np.ndarray)
        assert pose.shape[0] == demo_env.model.nu
