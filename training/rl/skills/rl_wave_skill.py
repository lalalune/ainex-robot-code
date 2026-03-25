"""RL-trained wave skill using CompositeSkill (walking + upper body).

Loads the trained MuJoCo wave policy checkpoint and runs it through the
CompositeSkill infrastructure, which combines BraxWalkSkill (legs) with
UpperBodySkill (upper body) to produce 24-dim joint targets.

Falls back to the scripted WaveSkill if the checkpoint file is missing.

Usage:
    skill = RLWaveSkill()
    skill.reset()
    action, status = skill.get_action(obs)

    # Preferred deployment interface:
    action, status = skill.get_action_from_telemetry(
        gyro=gyro, imu_roll=0.0, imu_pitch=0.0, joint_positions=jp,
    )
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np

from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus
from training.rl.skills.composite_skill import CompositeSkill, NUM_TOTAL_JOINTS
from training.mujoco import ainex_constants as consts

logger = logging.getLogger(__name__)

# Wave environment parameters (from training/mujoco/wave_env.py)
WAVE_FREQUENCY = 2.0       # Hz
WAVE_AMPLITUDE = 0.4       # rad (shoulder roll)
WAVE_SHOULDER_PITCH = -1.2  # rad (arm up)
WAVE_ELBOW_PITCH = -0.8    # rad (elbow bent)
WAVE_ELBOW_YAW = 0.0       # rad

# Task observation dimension: sin(phase) + cos(phase) + wave_target(4) = 6
TASK_OBS_DIM = 6


class RLWaveSkill(BaseSkill):
    """RL-trained wave skill using CompositeSkill (walking + upper body).

    Loads the trained wave checkpoint via CompositeSkill, which combines
    a frozen walking policy (legs) with a trained upper body policy
    (head + arms). The wave gesture is driven by a sinusoidal phase
    signal passed as task observation to the upper body policy.

    Falls back to the scripted WaveSkill if the checkpoint is not found.
    """

    name = "wave"
    action_dim = NUM_TOTAL_JOINTS  # 24
    requires_rl = True

    DEFAULT_WALKING_CHECKPOINT = "checkpoints/mujoco_locomotion_v13_flat_feet"
    DEFAULT_WAVE_CHECKPOINT = "checkpoints/mujoco_wave/final_params"

    WAVE_FREQUENCY = WAVE_FREQUENCY
    WAVE_DURATION = 3.0  # seconds

    def __init__(
        self,
        walking_checkpoint: str | None = None,
        wave_checkpoint: str | None = None,
        walk_vx: float = 0.3,
        duration_sec: float = 3.0,
    ):
        self._walking_checkpoint = walking_checkpoint or self.DEFAULT_WALKING_CHECKPOINT
        self._wave_checkpoint = wave_checkpoint or self.DEFAULT_WAVE_CHECKPOINT
        self._walk_vx = walk_vx
        self._default_duration = duration_sec
        self._duration = duration_sec

        self._composite: CompositeSkill | None = None
        self._fallback_skill: BaseSkill | None = None
        self._using_fallback = False

        # Timing state
        self._start_time: float = 0.0
        self._elapsed: float = 0.0

        # Try to load the composite skill with trained checkpoint
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the RL checkpoint; fall back to scripted if missing."""
        wave_ckpt_path = Path(self._wave_checkpoint)
        # The wave checkpoint path may point to final_params or the directory above it
        # Check both the path itself and the parent directory for config.json
        ckpt_dir = wave_ckpt_path
        if ckpt_dir.name == "final_params":
            ckpt_dir = ckpt_dir.parent

        if not ckpt_dir.exists() or not (ckpt_dir / "config.json").exists():
            logger.warning(
                "Wave checkpoint not found at %s — falling back to scripted WaveSkill",
                ckpt_dir,
            )
            self._load_fallback()
            return

        try:
            self._composite = CompositeSkill(
                walking_checkpoint=self._walking_checkpoint,
                upper_checkpoint=str(ckpt_dir),
                task_obs_dim=TASK_OBS_DIM,
            )
            self._composite.set_command(vx=self._walk_vx)
            self._using_fallback = False
            logger.info(
                "RLWaveSkill loaded: walk=%s upper=%s",
                self._walking_checkpoint,
                ckpt_dir,
            )
        except Exception:
            logger.exception("Failed to load wave checkpoint — falling back to scripted")
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Load the scripted WaveSkill as fallback."""
        from training.rl.skills.wave_skill import WaveSkill
        self._fallback_skill = WaveSkill()
        self._using_fallback = True

    def reset(self, params: SkillParams | None = None) -> None:
        """Reset the skill for a new execution."""
        if params and params.duration_sec > 0:
            self._duration = params.duration_sec
        else:
            self._duration = self._default_duration

        self._start_time = time.monotonic()
        self._elapsed = 0.0

        if self._using_fallback and self._fallback_skill is not None:
            self._fallback_skill.reset(params)
        elif self._composite is not None:
            self._composite.reset()
            self._composite.set_command(vx=self._walk_vx)

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, SkillStatus]:
        """Compute one step of the wave skill.

        The obs input is a generic observation vector. For the composite
        policy we extract what we can, but the preferred interface for
        deployment is get_action_from_telemetry().

        Args:
            obs: Observation vector (shape varies by caller).

        Returns:
            (action, status) where action is 24-dim joint targets (radians).
        """
        if self._using_fallback and self._fallback_skill is not None:
            return self._fallback_skill.get_action(obs)

        # Update elapsed time
        now = time.monotonic()
        self._elapsed = now - self._start_time

        # Check duration
        if self._elapsed >= self._duration:
            # Return neutral pose
            action = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
            return action, SkillStatus.COMPLETED

        # Compute task observation (wave phase)
        task_obs = self._compute_task_obs(self._elapsed)

        # Run composite policy with minimal telemetry
        # Extract IMU hints from obs if available
        imu_roll = 0.0
        imu_pitch = 0.0
        if obs.shape[0] >= 9:
            # Bridge obs convention: index 7=imu_roll, 8=imu_pitch
            imu_roll = float(obs[7]) if not np.isnan(obs[7]) else 0.0
            imu_pitch = float(obs[8]) if not np.isnan(obs[8]) else 0.0

        if self._composite is not None:
            targets = self._composite.get_full_action(
                imu_roll=imu_roll,
                imu_pitch=imu_pitch,
                task_obs=task_obs,
            )
        else:
            targets = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)

        return targets, SkillStatus.RUNNING

    def get_action_from_telemetry(
        self,
        gyro: np.ndarray | None = None,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        joint_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SkillStatus]:
        """Structured telemetry interface (preferred for deployment).

        Args:
            gyro: Angular velocity [wx, wy, wz] in body frame.
            imu_roll: Roll angle in radians.
            imu_pitch: Pitch angle in radians.
            joint_positions: All 24 joint positions in radians.

        Returns:
            (action, status) where action is 24-dim joint targets.
        """
        if self._using_fallback and self._fallback_skill is not None:
            obs = np.zeros(48, dtype=np.float32)
            return self._fallback_skill.get_action(obs)

        # Update elapsed time
        now = time.monotonic()
        self._elapsed = now - self._start_time

        # Check duration
        if self._elapsed >= self._duration:
            action = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)
            return action, SkillStatus.COMPLETED

        # Compute task observation (wave phase)
        task_obs = self._compute_task_obs(self._elapsed)

        if self._composite is not None:
            targets = self._composite.get_full_action(
                gyro=gyro,
                imu_roll=imu_roll,
                imu_pitch=imu_pitch,
                joint_positions=joint_positions,
                task_obs=task_obs,
            )
        else:
            targets = np.zeros(NUM_TOTAL_JOINTS, dtype=np.float32)

        return targets, SkillStatus.RUNNING

    def _compute_task_obs(self, elapsed: float) -> np.ndarray:
        """Compute wave-specific task observation.

        From wave_env.py, the task obs is 6-dim:
        - sin(phase), cos(phase)                                       (2)
        - wave targets [sho_pitch, sho_roll, el_pitch, el_yaw]        (4)

        The wave target for shoulder roll oscillates sinusoidally.
        """
        phase = elapsed * 2.0 * math.pi * WAVE_FREQUENCY
        sin_phase = math.sin(phase)
        cos_phase = math.cos(phase)

        # Wave targets matching wave_env.py _get_wave_target()
        sho_pitch = WAVE_SHOULDER_PITCH
        sho_roll = WAVE_AMPLITUDE * math.sin(phase)
        el_pitch = WAVE_ELBOW_PITCH
        el_yaw = WAVE_ELBOW_YAW

        return np.array(
            [sin_phase, cos_phase, sho_pitch, sho_roll, el_pitch, el_yaw],
            dtype=np.float32,
        )

    @property
    def using_fallback(self) -> bool:
        """Whether we are using the scripted fallback instead of RL policy."""
        return self._using_fallback

    @property
    def composite(self) -> CompositeSkill | None:
        """Access the underlying CompositeSkill (None if using fallback)."""
        return self._composite
