"""Composite skill: simultaneous walking + upper body control.

Runs BraxWalkSkill for legs and a separate upper body policy concurrently,
outputting a combined 24-dim joint target for all servos.

Architecture:
    BraxWalkSkill  → 12 leg joint targets
    UpperBodySkill → 12 upper body joint targets (head + arms)
    CompositeSkill → concatenated 24-dim joint targets → servo.set

Usage:
    from training.rl.skills.composite_skill import CompositeSkill
    skill = CompositeSkill(
        walking_checkpoint="checkpoints/mujoco_locomotion_v13_flat_feet",
        upper_checkpoint="checkpoints/mujoco_wave/final_params",
    )
    skill.set_command(vx=0.3)
    targets = skill.get_full_action(gyro, imu_roll, imu_pitch, joint_positions)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.rl.skills.brax_walk_skill import BraxWalkSkill, NUM_LEG_JOINTS
from training.mujoco import ainex_constants as consts

NUM_UPPER_JOINTS = consts.NUM_HEAD_ACTUATORS + consts.NUM_ARM_ACTUATORS  # 12
NUM_TOTAL_JOINTS = consts.NUM_ACTUATORS  # 24


class UpperBodySkill:
    """Upper body policy loaded from a Brax checkpoint.

    Maintains its own obs history and produces 12-dim upper body targets.
    """

    SINGLE_OBS_DIM = 42  # gyro(3) + gravity(3) + upper_pos(12) + upper_vel(12) + last_act(12)
    OBS_HISTORY_SIZE = 3
    TOTAL_OBS_DIM = SINGLE_OBS_DIM * OBS_HISTORY_SIZE
    ACTION_SCALE = 0.3

    def __init__(
        self,
        checkpoint_path: str | None = None,
        task_obs_dim: int = 0,
        default_upper_pose: np.ndarray | None = None,
    ) -> None:
        self._inference_fn = None
        self._task_obs_dim = task_obs_dim
        self._total_single = self.SINGLE_OBS_DIM + task_obs_dim
        self._total_obs_dim = self._total_single * self.OBS_HISTORY_SIZE

        self._last_action = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)
        self._obs_history = np.zeros(self._total_obs_dim, dtype=np.float32)
        self._last_positions = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)

        if default_upper_pose is not None:
            self._default_pose = default_upper_pose.copy()
        else:
            # Default upper body pose (all zeros for head/arms)
            self._default_pose = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load upper body policy checkpoint."""
        from training.mujoco.inference import load_policy
        self._inference_fn, _ = load_policy(path)

    def reset(self) -> None:
        self._last_action = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)
        self._obs_history = np.zeros(self._total_obs_dim, dtype=np.float32)
        self._last_positions = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)

    def get_action(
        self,
        gyro: np.ndarray | None = None,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        upper_joint_positions: np.ndarray | None = None,
        task_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute upper body joint targets.

        Args:
            gyro: Angular velocity [wx, wy, wz].
            imu_roll: Roll angle in radians.
            imu_pitch: Pitch angle in radians.
            upper_joint_positions: Current upper body joint positions (12-dim).
            task_obs: Task-specific observation (task_obs_dim).

        Returns:
            12-dim absolute joint targets for upper body.
        """
        if self._inference_fn is None:
            return self._default_pose.copy()

        if gyro is None:
            gyro = np.zeros(3, dtype=np.float32)

        gravity = np.array([
            np.sin(imu_roll),
            np.sin(imu_pitch),
            np.cos(imu_roll) * np.cos(imu_pitch),
        ], dtype=np.float32)

        if upper_joint_positions is not None:
            upper_pos = upper_joint_positions - self._default_pose
            upper_vel = (upper_joint_positions - self._last_positions) * 0.05
            self._last_positions = upper_joint_positions.copy()
        else:
            upper_pos = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)
            upper_vel = np.zeros(NUM_UPPER_JOINTS, dtype=np.float32)

        single_obs = np.concatenate([
            gyro,               # 3
            gravity,            # 3
            upper_pos,          # 12
            upper_vel,          # 12
            self._last_action,  # 12
        ]).astype(np.float32)

        if task_obs is not None:
            single_obs = np.concatenate([single_obs, task_obs])

        # Stack into history
        self._obs_history = np.roll(self._obs_history, single_obs.size)
        self._obs_history[:single_obs.size] = single_obs
        full_obs = self._obs_history.copy()

        action = self._inference_fn(full_obs)
        action = np.clip(action, -1.0, 1.0)

        targets = self._default_pose + action[:NUM_UPPER_JOINTS] * self.ACTION_SCALE
        self._last_action = action[:NUM_UPPER_JOINTS].astype(np.float32)
        return targets

    @property
    def is_loaded(self) -> bool:
        return self._inference_fn is not None


class CompositeSkill:
    """Combines walking + upper body skills into 24-dim joint control.

    Runs both policies in parallel, concatenates their outputs, and
    returns a full 24-dim joint target vector for all servos.
    """

    def __init__(
        self,
        walking_checkpoint: str | None = None,
        upper_checkpoint: str | None = None,
        task_obs_dim: int = 0,
    ) -> None:
        self._walk_skill = BraxWalkSkill(checkpoint_path=walking_checkpoint)
        self._upper_skill = UpperBodySkill(
            checkpoint_path=upper_checkpoint,
            task_obs_dim=task_obs_dim,
        )

    def set_command(
        self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0
    ) -> None:
        """Set velocity command for the walking policy."""
        self._walk_skill.set_command(vx, vy, vyaw)

    def reset(self) -> None:
        """Reset both skills."""
        self._walk_skill.reset()
        self._upper_skill.reset()

    def get_full_action(
        self,
        gyro: np.ndarray | None = None,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        joint_positions: np.ndarray | None = None,
        task_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute combined 24-dim joint targets.

        Args:
            gyro: Angular velocity [wx, wy, wz].
            imu_roll: Roll angle in radians.
            imu_pitch: Pitch angle in radians.
            joint_positions: All 24 joint positions (legs + upper body).
            task_obs: Task-specific observation for upper body skill.

        Returns:
            24-dim absolute joint targets [legs(12), upper_body(12)].
        """
        # Split joint positions
        leg_positions = None
        upper_positions = None
        if joint_positions is not None:
            leg_positions = joint_positions[:NUM_LEG_JOINTS]
            upper_positions = joint_positions[NUM_LEG_JOINTS:]

        # Walking policy → leg targets
        from training.rl.skills.base_skill import SkillStatus
        leg_targets, _ = self._walk_skill.get_action_from_telemetry(
            gyro=gyro,
            imu_roll=imu_roll,
            imu_pitch=imu_pitch,
            joint_positions=leg_positions,
        )

        # Upper body policy → upper targets
        upper_targets = self._upper_skill.get_action(
            gyro=gyro,
            imu_roll=imu_roll,
            imu_pitch=imu_pitch,
            upper_joint_positions=upper_positions,
            task_obs=task_obs,
        )

        return np.concatenate([leg_targets, upper_targets])

    @property
    def walk_skill(self) -> BraxWalkSkill:
        return self._walk_skill

    @property
    def upper_skill(self) -> UpperBodySkill:
        return self._upper_skill
