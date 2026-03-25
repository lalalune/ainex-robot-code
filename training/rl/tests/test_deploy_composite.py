"""Tests for DeployComposite safety and servo conversion."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from training.rl.deploy.deploy_composite import (
    DeployComposite,
    FALL_PITCH_THRESHOLD,
    FALL_ROLL_THRESHOLD,
    MAX_JOINT_DELTA,
)
from training.rl.skills.composite_skill import NUM_TOTAL_JOINTS
from training.mujoco.ainex_constants import ALL_JOINT_NAMES
from bridge.isaaclab.joint_map import joint_name_to_servo_id


@pytest.fixture
def deployer():
    """DeployComposite with mocked CompositeSkill to avoid checkpoint loading."""
    with patch("training.rl.deploy.deploy_composite.CompositeSkill") as mock_cls:
        mock_skill = MagicMock()
        mock_skill.get_full_action.return_value = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
        mock_cls.return_value = mock_skill
        d = DeployComposite(
            walking_checkpoint="/tmp/nonexistent",
            upper_checkpoint="/tmp/nonexistent",
            dry_run=True,
            duration=1.0,
        )
        yield d


class TestSafetyClamping:
    def test_ramp_zero_returns_near_default(self, deployer):
        """At ramp=0.0, output should be close to default (zeros)."""
        targets = np.ones(NUM_TOTAL_JOINTS, dtype=np.float32) * 0.5
        clamped = deployer.safety_clamp(targets, ramp_factor=0.0)
        np.testing.assert_allclose(clamped, 0.0, atol=MAX_JOINT_DELTA + 1e-5)

    def test_delta_clamped(self, deployer):
        """Per-step delta must not exceed MAX_JOINT_DELTA."""
        deployer._last_targets = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
        targets = np.ones(NUM_TOTAL_JOINTS, dtype=np.float32) * 10.0
        clamped = deployer.safety_clamp(targets, ramp_factor=1.0)
        delta = clamped - np.zeros(NUM_TOTAL_JOINTS)
        assert np.all(np.abs(delta) <= MAX_JOINT_DELTA + 1e-6)

    def test_ramp_full_converges(self, deployer):
        """At ramp=1.0 with small targets, output converges after many steps."""
        targets = np.ones(NUM_TOTAL_JOINTS, dtype=np.float32) * 0.05
        for _ in range(20):
            clamped = deployer.safety_clamp(targets, ramp_factor=1.0)
        np.testing.assert_allclose(clamped, targets, atol=1e-3)


class TestFallDetection:
    def test_no_fall_at_zero(self, deployer):
        deployer._imu_roll = 0.0
        deployer._imu_pitch = 0.0
        assert deployer.check_fall() is False

    def test_fall_on_pitch(self, deployer):
        deployer._imu_pitch = FALL_PITCH_THRESHOLD + 0.1
        assert deployer.check_fall() is True

    def test_fall_on_roll(self, deployer):
        deployer._imu_roll = -(FALL_ROLL_THRESHOLD + 0.1)
        assert deployer.check_fall() is True

    def test_no_fall_below_threshold(self, deployer):
        deployer._imu_pitch = FALL_PITCH_THRESHOLD - 0.1
        deployer._imu_roll = FALL_ROLL_THRESHOLD - 0.1
        assert deployer.check_fall() is False


class TestServoConversion:
    def test_all_24_joints_mapped(self, deployer):
        """Every joint in ALL_JOINT_NAMES must produce a valid servo command."""
        targets = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
        cmds = deployer.joint_targets_to_servo_commands(targets)
        assert len(cmds) == NUM_TOTAL_JOINTS
        ids_seen = set()
        for cmd in cmds:
            assert 0 <= cmd["position"] <= 1000
            ids_seen.add(cmd["id"])
        assert len(ids_seen) == NUM_TOTAL_JOINTS

    def test_zero_radians_maps_to_center_pulse(self, deployer):
        """0 radians should map to center pulse (~500)."""
        targets = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
        cmds = deployer.joint_targets_to_servo_commands(targets)
        for cmd in cmds:
            assert cmd["position"] == 500


class TestTaskObsConsistency:
    def test_deploy_composite_matches_rl_wave_skill(self, deployer):
        """Task obs computation must be identical between deploy and skill."""
        from training.rl.skills.rl_wave_skill import RLWaveSkill

        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")

        for elapsed in [0.0, 0.25, 0.5, 1.0, 2.37]:
            obs_deploy = deployer.compute_task_obs(elapsed)
            obs_skill = skill._compute_task_obs(elapsed)
            np.testing.assert_allclose(
                obs_deploy, obs_skill, atol=1e-6,
                err_msg=f"Mismatch at elapsed={elapsed}",
            )
