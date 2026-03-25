"""Tests for BraxTargetSkill — target-reaching deployment skill."""

from __future__ import annotations

import math

import numpy as np
import pytest

from training.rl.skills.base_skill import SkillParams, SkillStatus
from training.rl.skills.brax_target_skill import (
    BraxTargetSkill,
    FALLBACK_VX_MAX,
    FALLBACK_VYAW_GAIN,
    FALLBACK_VYAW_MAX,
    NUM_LEG_JOINTS,
)
from training.rl.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skill() -> BraxTargetSkill:
    """Create a BraxTargetSkill in fallback mode (no checkpoint)."""
    return BraxTargetSkill(checkpoint_path="/nonexistent/checkpoint")


# ---------------------------------------------------------------------------
# Init / fallback mode
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_no_checkpoint_uses_fallback(self, skill: BraxTargetSkill) -> None:
        assert skill.using_fallback is True
        assert skill.is_loaded is False
        assert skill._fallback_walk is not None

    def test_name(self, skill: BraxTargetSkill) -> None:
        assert skill.name == "walk_to_target"

    def test_action_dim(self, skill: BraxTargetSkill) -> None:
        assert skill.action_dim == 12

    def test_requires_rl(self, skill: BraxTargetSkill) -> None:
        assert skill.requires_rl is True


# ---------------------------------------------------------------------------
# set_target
# ---------------------------------------------------------------------------

class TestSetTarget:
    def test_set_target_stores_values(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=1.5, y=0.3)
        assert skill._target_x == pytest.approx(1.5)
        assert skill._target_y == pytest.approx(0.3)
        assert skill._target_set is True

    def test_set_target_updates_distance(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=3.0, y=4.0)
        assert skill.target_distance == pytest.approx(5.0)

    def test_set_target_updates_bearing(self, skill: BraxTargetSkill) -> None:
        # Straight ahead
        skill.set_target(x=1.0, y=0.0)
        assert skill.target_bearing == pytest.approx(0.0)

        # 90 degrees left
        skill.set_target(x=0.0, y=1.0)
        assert skill.target_bearing == pytest.approx(math.pi / 2)

    def test_set_target_zero(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=0.0, y=0.0)
        assert skill.target_distance == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# set_target_world
# ---------------------------------------------------------------------------

class TestSetTargetWorld:
    def test_convert_world_to_body_straight_ahead(
        self, skill: BraxTargetSkill
    ) -> None:
        """Robot facing +x, target 2m ahead in world = 2m ahead in body."""
        target_world = np.array([3.0, 0.0])
        robot_pos = np.array([1.0, 0.0])
        robot_yaw = 0.0

        skill.set_target_world(target_world, robot_pos, robot_yaw)

        assert skill._target_x == pytest.approx(2.0, abs=1e-6)
        assert skill._target_y == pytest.approx(0.0, abs=1e-6)

    def test_convert_world_to_body_rotated_90(
        self, skill: BraxTargetSkill
    ) -> None:
        """Robot facing +y (yaw = pi/2), target 2m to the right in world (+x)
        should be 2m to the right in body (negative y)."""
        target_world = np.array([3.0, 0.0])
        robot_pos = np.array([1.0, 0.0])
        robot_yaw = math.pi / 2  # facing +y

        skill.set_target_world(target_world, robot_pos, robot_yaw)

        # delta_world = [2, 0]
        # body_x = 2*cos(-pi/2) - 0*sin(-pi/2) = 0
        # body_y = 2*sin(-pi/2) + 0*cos(-pi/2) = -2
        assert skill._target_x == pytest.approx(0.0, abs=1e-6)
        assert skill._target_y == pytest.approx(-2.0, abs=1e-6)

    def test_convert_world_to_body_3d_input(
        self, skill: BraxTargetSkill
    ) -> None:
        """3D arrays should work — only first 2 dims matter."""
        target_world = np.array([5.0, 5.0, 0.0])
        robot_pos = np.array([5.0, 3.0, 0.0])
        robot_yaw = 0.0

        skill.set_target_world(target_world, robot_pos, robot_yaw)

        assert skill._target_x == pytest.approx(0.0, abs=1e-6)
        assert skill._target_y == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Fallback velocity computation
# ---------------------------------------------------------------------------

class TestFallbackVelocity:
    def test_vx_proportional_to_distance(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=0.15, y=0.0)
        vx, _ = skill._compute_fallback_velocity()
        assert vx == pytest.approx(0.15, abs=1e-6)

    def test_vx_clamped_at_max(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=10.0, y=0.0)
        vx, _ = skill._compute_fallback_velocity()
        assert vx == pytest.approx(FALLBACK_VX_MAX)

    def test_vx_zero_when_at_target(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=0.0, y=0.0)
        vx, _ = skill._compute_fallback_velocity()
        assert vx == pytest.approx(0.0)

    def test_vyaw_proportional_to_bearing(
        self, skill: BraxTargetSkill
    ) -> None:
        # Target 45 degrees left
        skill.set_target(x=1.0, y=1.0)
        _, vyaw = skill._compute_fallback_velocity()
        expected_bearing = math.atan2(1.0, 1.0)  # pi/4
        expected_vyaw = np.clip(
            expected_bearing * FALLBACK_VYAW_GAIN,
            -FALLBACK_VYAW_MAX,
            FALLBACK_VYAW_MAX,
        )
        assert vyaw == pytest.approx(expected_vyaw, abs=1e-6)

    def test_vyaw_clamped(self, skill: BraxTargetSkill) -> None:
        # Target directly left: bearing = pi/2
        skill.set_target(x=0.001, y=10.0)
        _, vyaw = skill._compute_fallback_velocity()
        assert abs(vyaw) <= FALLBACK_VYAW_MAX + 1e-9


# ---------------------------------------------------------------------------
# target_reached
# ---------------------------------------------------------------------------

class TestTargetReached:
    def test_reached_when_within_threshold(
        self, skill: BraxTargetSkill
    ) -> None:
        skill.set_target(x=0.1, y=0.1)
        # distance = sqrt(0.01 + 0.01) ~ 0.141, which is < 0.3 threshold
        assert skill.target_reached is True

    def test_not_reached_when_outside_threshold(
        self, skill: BraxTargetSkill
    ) -> None:
        skill.set_target(x=1.0, y=1.0)
        # distance = sqrt(2) ~ 1.414, which is > 0.3 threshold
        assert skill.target_reached is False

    def test_custom_threshold(self) -> None:
        skill = BraxTargetSkill(
            checkpoint_path="/nonexistent", arrival_threshold=2.0
        )
        skill.set_target(x=1.0, y=1.0)
        # distance ~ 1.414 < 2.0
        assert skill.target_reached is True


# ---------------------------------------------------------------------------
# get_action_from_telemetry
# ---------------------------------------------------------------------------

class TestGetActionFromTelemetry:
    def test_returns_12_dim_output(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=1.0, y=0.0)
        action, status = skill.get_action_from_telemetry(
            imu_roll=0.0,
            imu_pitch=0.0,
            joint_positions=np.zeros(NUM_LEG_JOINTS, dtype=np.float32),
        )
        assert action.shape == (NUM_LEG_JOINTS,)
        assert status == SkillStatus.RUNNING

    def test_returns_completed_when_target_reached(
        self, skill: BraxTargetSkill
    ) -> None:
        skill.set_target(x=0.1, y=0.0)  # close enough
        _, status = skill.get_action_from_telemetry()
        assert status == SkillStatus.COMPLETED

    def test_returns_completed_on_duration_expired(self) -> None:
        skill = BraxTargetSkill(checkpoint_path="/nonexistent")
        skill.reset(SkillParams(duration_sec=0.02))
        skill.set_target(x=5.0, y=0.0)  # far away
        # First step (step 1 * 0.02 = 0.02 >= 0.02)
        _, status = skill.get_action_from_telemetry()
        assert status == SkillStatus.COMPLETED


# ---------------------------------------------------------------------------
# get_action
# ---------------------------------------------------------------------------

class TestGetAction:
    def test_returns_12_dim_output(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=2.0, y=0.0)
        obs = np.zeros(135, dtype=np.float32)  # BraxWalkSkill obs dim
        action, status = skill.get_action(obs)
        assert action.shape == (NUM_LEG_JOINTS,)

    def test_default_pose_returned_when_target_reached(
        self, skill: BraxTargetSkill
    ) -> None:
        skill.set_target(x=0.05, y=0.0)
        obs = np.zeros(135, dtype=np.float32)
        action, status = skill.get_action(obs)
        assert status == SkillStatus.COMPLETED
        np.testing.assert_array_almost_equal(action, skill.default_pose)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_target(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=1.0, y=2.0)
        skill.reset()
        assert skill._target_set is False
        assert skill._target_x == 0.0
        assert skill._target_y == 0.0

    def test_reset_clears_step_counter(self, skill: BraxTargetSkill) -> None:
        skill.set_target(x=5.0, y=0.0)
        skill.get_action_from_telemetry()
        skill.get_action_from_telemetry()
        assert skill._step == 2
        skill.reset()
        assert skill._step == 0


# ---------------------------------------------------------------------------
# Registry aliases
# ---------------------------------------------------------------------------

class TestRegistryAliases:
    def test_walk_to_target_alias(self) -> None:
        registry = SkillRegistry()
        skill = BraxTargetSkill(checkpoint_path="/nonexistent")
        registry.register(skill)

        assert registry.get("walk to target") is skill

    def test_go_to_alias(self) -> None:
        registry = SkillRegistry()
        skill = BraxTargetSkill(checkpoint_path="/nonexistent")
        registry.register(skill)

        assert registry.get("go to") is skill

    def test_navigate_to_alias(self) -> None:
        registry = SkillRegistry()
        skill = BraxTargetSkill(checkpoint_path="/nonexistent")
        registry.register(skill)

        assert registry.get("navigate to") is skill

    def test_approach_alias(self) -> None:
        registry = SkillRegistry()
        skill = BraxTargetSkill(checkpoint_path="/nonexistent")
        registry.register(skill)

        assert registry.get("approach") is skill

    def test_direct_name_lookup(self) -> None:
        registry = SkillRegistry()
        skill = BraxTargetSkill(checkpoint_path="/nonexistent")
        registry.register(skill)

        assert registry.get("walk_to_target") is skill
