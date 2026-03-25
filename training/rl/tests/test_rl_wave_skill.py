"""Tests for RLWaveSkill — the RL-trained wave policy."""

from __future__ import annotations

import math
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from training.rl.skills.base_skill import SkillParams, SkillStatus
from training.rl.skills.rl_wave_skill import (
    RLWaveSkill,
    TASK_OBS_DIM,
    WAVE_FREQUENCY,
    WAVE_AMPLITUDE,
    WAVE_SHOULDER_PITCH,
    WAVE_ELBOW_PITCH,
)
from training.rl.skills.composite_skill import NUM_TOTAL_JOINTS


class TestRLWaveSkillInit:
    """Test initialization and fallback behavior."""

    def test_fallback_when_no_checkpoint(self, tmp_path):
        """Should fall back to scripted WaveSkill when checkpoint missing."""
        skill = RLWaveSkill(
            walking_checkpoint=str(tmp_path / "nonexistent_walk"),
            wave_checkpoint=str(tmp_path / "nonexistent_wave"),
        )
        assert skill.using_fallback is True
        assert skill.composite is None

    def test_name_and_attrs(self):
        """Skill name and metadata are correct."""
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        assert skill.name == "wave"
        assert skill.action_dim == NUM_TOTAL_JOINTS
        assert skill.requires_rl is True


class TestTaskObservation:
    """Test wave phase task observation computation."""

    def test_task_obs_shape(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        obs = skill._compute_task_obs(0.0)
        assert obs.shape == (TASK_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_task_obs_at_zero(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        obs = skill._compute_task_obs(0.0)
        # At t=0: sin(0)=0, cos(0)=1
        assert abs(obs[0]) < 1e-6  # sin(phase)
        assert abs(obs[1] - 1.0) < 1e-6  # cos(phase)
        # Wave targets
        assert abs(obs[2] - WAVE_SHOULDER_PITCH) < 1e-6
        assert abs(obs[3]) < 1e-6  # sho_roll = amplitude * sin(0) = 0
        assert abs(obs[4] - WAVE_ELBOW_PITCH) < 1e-6

    def test_task_obs_quarter_period(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        # Quarter period: t = 1/(4*freq)
        t = 1.0 / (4.0 * WAVE_FREQUENCY)
        obs = skill._compute_task_obs(t)
        # At quarter period: sin(pi/2)=1, cos(pi/2)=0
        assert abs(obs[0] - 1.0) < 1e-5  # sin(phase) ≈ 1
        assert abs(obs[1]) < 1e-5  # cos(phase) ≈ 0
        # sho_roll = amplitude * sin(pi/2) = amplitude
        assert abs(obs[3] - WAVE_AMPLITUDE) < 1e-5


class TestGetAction:
    """Test get_action and get_action_from_telemetry in fallback mode."""

    def test_get_action_fallback_returns_correct_shape(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        skill.reset()
        obs = np.zeros(48, dtype=np.float32)
        action, status = skill.get_action(obs)
        # Fallback WaveSkill returns 24-dim
        assert action.shape == (24,)
        assert status == SkillStatus.RUNNING

    def test_get_action_from_telemetry_fallback(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        skill.reset()
        action, status = skill.get_action_from_telemetry(
            imu_roll=0.0, imu_pitch=0.0,
        )
        assert action.shape == (24,)
        assert status == SkillStatus.RUNNING

    def test_completes_after_duration(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent", duration_sec=0.01)
        skill.reset()
        time.sleep(0.02)
        # After duration, fallback get_action_from_telemetry should complete
        # (but fallback doesn't have duration logic — the RL path does)
        # For RL path, the completion is handled before fallback delegation
        # So this tests the RL completion path
        skill._using_fallback = False
        skill._composite = None  # will return zeros
        action, status = skill.get_action_from_telemetry()
        assert status == SkillStatus.COMPLETED

    def test_reset_with_custom_duration(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent", duration_sec=5.0)
        params = SkillParams(duration_sec=2.0)
        skill.reset(params)
        assert skill._duration == 2.0

    def test_reset_without_params_uses_default(self):
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent", duration_sec=5.0)
        skill.reset()
        assert skill._duration == 5.0


class TestRegistryIntegration:
    """Test that RLWaveSkill integrates with the skill registry."""

    def test_registry_lookup(self):
        from training.rl.skills.registry import SkillRegistry
        registry = SkillRegistry()
        skill = RLWaveSkill(wave_checkpoint="/tmp/nonexistent")
        registry.register(skill)

        assert registry.get("wave") is skill
        assert registry.get("wave hello") is skill
        assert registry.get("greet") is skill
        assert registry.get("say hello") is skill
