"""Target-reaching skill backed by Brax/JAX checkpoint.

Walk toward a target position (x, y) in body frame using the trained
TargetReaching policy.  When no target_reaching checkpoint exists, falls
back to the existing BraxWalkSkill by converting target direction into
velocity commands (vx proportional to distance, vyaw proportional to
bearing).

Usage:
    skill = BraxTargetSkill()
    skill.set_target(x=1.5, y=0.3)
    action, status = skill.get_action_from_telemetry(
        imu_roll=0.0, imu_pitch=0.0, joint_positions=feedback,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus
from training.rl.skills.brax_walk_skill import (
    BraxWalkSkill,
    NUM_LEG_JOINTS,
    ACTION_SCALE,
    CTRL_DT,
)

logger = logging.getLogger(__name__)

# Observation layout for TargetReaching (matches target.py):
# gyro(3) + gravity(3) + target_vec(2) + target_dist(1) + target_bearing(1)
# + joint_pos(12) + last_act(12) = 34
# Note: target.py uses 24 actuators (full body) in training, but deployment
# skill only controls the 12 leg joints.  We build the obs with 12-dim
# joint_pos and 12-dim last_act to match what the deployment checkpoint
# expects (58 per frame if 24-act, 34 per frame if 12-act).  The actual
# size is resolved from the checkpoint config at load time.
TARGET_OBS_EXTRA = 4  # target_vec(2) + target_dist(1) + target_bearing(1)

# Single obs size with 12-dim joints (deployment mode)
SINGLE_OBS_DIM_12 = 3 + 3 + 2 + 1 + 1 + 12 + 12  # = 34
# Single obs size with 24-dim joints (full training mode, matches target.py)
SINGLE_OBS_DIM_24 = 3 + 3 + 2 + 1 + 1 + 24 + 24  # = 58

OBS_HISTORY_SIZE = 3

# Fallback velocity gains
FALLBACK_VX_MAX = 0.3
FALLBACK_VYAW_GAIN = 2.0
FALLBACK_VYAW_MAX = 1.0


class BraxTargetSkill(BaseSkill):
    """Walk to a target position using the trained TargetReaching policy.

    Accepts a target (x, y) in body frame and produces leg joint targets
    that walk toward it. Uses the same Brax checkpoint format as the
    walking policy, but with extended observation (target_vec, target_dist,
    target_bearing appended).

    Fallback mode
    -------------
    If no target_reaching checkpoint is available, the skill delegates to
    BraxWalkSkill with velocity commands computed from the target direction:
    - vx  = clamp(target_dist, 0, 0.3)
    - vyaw = clamp(target_bearing * 2.0, -1.0, 1.0)
    This allows demonstration before the target-reaching policy is trained.
    """

    name = "walk_to_target"
    action_dim = NUM_LEG_JOINTS  # 12 leg joints
    requires_rl = True

    DEFAULT_CHECKPOINT = "checkpoints/mujoco_target_reaching"

    def __init__(
        self,
        checkpoint_path: str | None = None,
        arrival_threshold: float = 0.3,  # metres
    ) -> None:
        self._inference_fn = None
        self._config: dict | None = None
        self._arrival_threshold = arrival_threshold

        # Target in body frame
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._target_set: bool = False

        # State buffers
        self._last_action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self._last_positions = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        # Will be sized once we know the obs dim (from checkpoint or fallback)
        self._single_obs_dim: int = SINGLE_OBS_DIM_12
        self._obs_history = np.zeros(
            self._single_obs_dim * OBS_HISTORY_SIZE, dtype=np.float32
        )

        # Default standing pose (leg joints only)
        self._default_pose = np.array(
            [
                # Right leg: hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll
                0, 0, -0.3, 0.6, -0.3, 0,
                # Left leg
                0, 0, 0.3, -0.6, 0.3, 0,
            ],
            dtype=np.float32,
        )

        self._params = SkillParams()
        self._step = 0

        # Fallback walk skill (always available)
        self._fallback_walk: BraxWalkSkill | None = None
        self._using_fallback = True

        # Try loading the target-reaching checkpoint
        resolved_path = checkpoint_path or self.DEFAULT_CHECKPOINT
        if Path(resolved_path).exists():
            try:
                self.load_checkpoint(resolved_path)
                self._using_fallback = False
                logger.info(
                    "BraxTargetSkill: loaded target-reaching checkpoint from %s",
                    resolved_path,
                )
            except Exception:
                logger.warning(
                    "BraxTargetSkill: failed to load checkpoint %s, using fallback",
                    resolved_path,
                    exc_info=True,
                )
        else:
            logger.info(
                "BraxTargetSkill: no checkpoint at %s, using fallback walk skill",
                resolved_path,
            )

        if self._using_fallback:
            self._init_fallback()

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        """Load Brax checkpoint via training.mujoco.inference."""
        from training.mujoco.inference import load_policy

        self._inference_fn, self._config = load_policy(path)

        # Resolve observation dimensions from the loaded config
        obs_size = self._config.get("obs_size")
        if obs_size is not None:
            # obs_size = single_obs_dim * history_size (+ entity_slots maybe)
            enable_entities = self._config.get("enable_entity_slots", False)
            entity_dims = 0
            if enable_entities:
                try:
                    from perception.entity_slots.slot_config import TOTAL_ENTITY_DIMS
                    entity_dims = TOTAL_ENTITY_DIMS
                except ImportError:
                    entity_dims = 152  # 8 slots * 19 dims

            core_obs_size = obs_size - entity_dims
            if core_obs_size > 0 and core_obs_size % OBS_HISTORY_SIZE == 0:
                self._single_obs_dim = core_obs_size // OBS_HISTORY_SIZE
            else:
                self._single_obs_dim = SINGLE_OBS_DIM_24

            self._obs_history = np.zeros(
                self._single_obs_dim * OBS_HISTORY_SIZE, dtype=np.float32
            )

    # ------------------------------------------------------------------
    # Fallback initialisation
    # ------------------------------------------------------------------

    def _init_fallback(self) -> None:
        """Initialise fallback walk skill (loads walking checkpoint if available)."""
        walk_ckpt = "checkpoints/mujoco_locomotion_v13_flat_feet"
        ckpt = walk_ckpt if Path(walk_ckpt).exists() else None
        self._fallback_walk = BraxWalkSkill(checkpoint_path=ckpt)
        self._using_fallback = True

    # ------------------------------------------------------------------
    # Target setters
    # ------------------------------------------------------------------

    def set_target(self, x: float, y: float) -> None:
        """Set target position in body frame (metres).

        Args:
            x: forward distance (positive = ahead).
            y: lateral distance (positive = left).
        """
        self._target_x = float(x)
        self._target_y = float(y)
        self._target_set = True

    def set_target_world(
        self,
        target_world: np.ndarray,
        robot_pos: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """Set target from world-frame coordinates.

        Converts to body frame using robot's current pose.

        Args:
            target_world: (2,) or (3,) target position in world frame.
            robot_pos: (2,) or (3,) robot position in world frame.
            robot_yaw: robot heading in radians (0 = +x axis).
        """
        delta = np.asarray(target_world[:2], dtype=np.float64) - np.asarray(
            robot_pos[:2], dtype=np.float64
        )
        cos_yaw = np.cos(-robot_yaw)
        sin_yaw = np.sin(-robot_yaw)
        body_x = float(delta[0] * cos_yaw - delta[1] * sin_yaw)
        body_y = float(delta[0] * sin_yaw + delta[1] * cos_yaw)
        self.set_target(body_x, body_y)

    # ------------------------------------------------------------------
    # Target properties
    # ------------------------------------------------------------------

    @property
    def target_reached(self) -> bool:
        """Whether the robot has reached the target."""
        return self.target_distance <= self._arrival_threshold

    @property
    def target_distance(self) -> float:
        """Current distance to target in body frame."""
        return float(np.sqrt(self._target_x ** 2 + self._target_y ** 2))

    @property
    def target_bearing(self) -> float:
        """Bearing to target in radians (0 = straight ahead)."""
        return float(np.arctan2(self._target_y, self._target_x))

    # ------------------------------------------------------------------
    # Skill interface
    # ------------------------------------------------------------------

    def reset(self, params: SkillParams | None = None) -> None:
        self._params = params or SkillParams()
        self._step = 0
        self._last_action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self._obs_history = np.zeros(
            self._single_obs_dim * OBS_HISTORY_SIZE, dtype=np.float32
        )
        self._last_positions = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_set = False

        if self._fallback_walk is not None:
            self._fallback_walk.reset(params)

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, SkillStatus]:
        """Compute one step of target-reaching policy.

        Args:
            obs: Pre-built observation vector (full history) or partial obs.

        Returns:
            (joint_targets, status) where joint_targets are absolute radians
            for the 12 leg joints.
        """
        self._step += 1

        # Duration check
        if self._params.duration_sec > 0:
            elapsed = self._step * CTRL_DT
            if elapsed >= self._params.duration_sec:
                return self._default_pose.copy(), SkillStatus.COMPLETED

        # Target reached check
        if self._target_set and self.target_reached:
            return self._default_pose.copy(), SkillStatus.COMPLETED

        # Fallback mode: delegate to walk skill with computed velocity
        if self._using_fallback:
            return self._fallback_get_action(obs)

        # Native target-reaching policy
        if self._inference_fn is None:
            return self._default_pose.copy(), SkillStatus.RUNNING

        expected_dim = self._single_obs_dim * OBS_HISTORY_SIZE
        if obs.shape[0] == expected_dim:
            full_obs = obs
        else:
            full_obs = self._obs_history.copy()

        action = self._inference_fn(full_obs)
        action = np.clip(action, -1.0, 1.0)

        joint_targets = self._default_pose + action[:NUM_LEG_JOINTS] * ACTION_SCALE
        self._last_action = action[:NUM_LEG_JOINTS].astype(np.float32)
        return joint_targets, SkillStatus.RUNNING

    def get_action_from_telemetry(
        self,
        gyro: np.ndarray | None = None,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        joint_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SkillStatus]:
        """Structured telemetry interface.

        Computes target_vec, target_dist, target_bearing from the set
        target and current IMU, then builds the full observation for the
        policy.

        Args:
            gyro: Angular velocity [wx, wy, wz] in body frame.
            imu_roll: Roll angle in radians.
            imu_pitch: Pitch angle in radians.
            joint_positions: Current leg joint positions in radians (12-dim).

        Returns:
            (joint_targets, status) where joint_targets are absolute radians.
        """
        self._step += 1

        # Duration check
        if self._params.duration_sec > 0:
            elapsed = self._step * CTRL_DT
            if elapsed >= self._params.duration_sec:
                return self._default_pose.copy(), SkillStatus.COMPLETED

        # Target reached check
        if self._target_set and self.target_reached:
            return self._default_pose.copy(), SkillStatus.COMPLETED

        # Fallback mode
        if self._using_fallback:
            return self._fallback_get_action_from_telemetry(
                gyro=gyro,
                imu_roll=imu_roll,
                imu_pitch=imu_pitch,
                joint_positions=joint_positions,
            )

        # Native target-reaching policy
        if self._inference_fn is None:
            return self._default_pose.copy(), SkillStatus.RUNNING

        if gyro is None:
            gyro = np.zeros(3, dtype=np.float32)

        # Gravity from IMU orientation
        gravity = np.array(
            [
                np.sin(imu_roll),
                np.sin(imu_pitch),
                np.cos(imu_roll) * np.cos(imu_pitch),
            ],
            dtype=np.float32,
        )

        # Target info in body frame
        target_vec = np.array(
            [self._target_x, self._target_y], dtype=np.float32
        )
        target_dist_val = np.float32(self.target_distance)
        target_bearing_val = np.float32(self.target_bearing)

        # Joint positions relative to default
        if joint_positions is not None:
            leg_pos = joint_positions[:NUM_LEG_JOINTS] - self._default_pose
            leg_vel = (
                joint_positions[:NUM_LEG_JOINTS] - self._last_positions
            ) * 0.05
            self._last_positions = joint_positions[:NUM_LEG_JOINTS].copy()
        else:
            leg_pos = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
            leg_vel = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        # Build single obs matching target.py layout
        # gyro(3) + gravity(3) + target_vec(2) + target_dist(1)
        # + target_bearing(1) + joint_pos + last_act
        if self._single_obs_dim == SINGLE_OBS_DIM_24:
            # 24-actuator mode: pad leg data with zeros for head/arms
            joint_pos_full = np.zeros(24, dtype=np.float32)
            joint_pos_full[:NUM_LEG_JOINTS] = leg_pos
            last_act_full = np.zeros(24, dtype=np.float32)
            last_act_full[:NUM_LEG_JOINTS] = self._last_action
            single_obs = np.concatenate(
                [
                    gyro,                                   # 3
                    gravity,                                # 3
                    target_vec,                             # 2
                    np.array([target_dist_val]),             # 1
                    np.array([target_bearing_val]),          # 1
                    joint_pos_full,                          # 24
                    last_act_full,                           # 24
                ]
            ).astype(np.float32)  # 58
        else:
            # 12-actuator deployment mode
            single_obs = np.concatenate(
                [
                    gyro,                                   # 3
                    gravity,                                # 3
                    target_vec,                             # 2
                    np.array([target_dist_val]),             # 1
                    np.array([target_bearing_val]),          # 1
                    leg_pos,                                # 12
                    self._last_action,                      # 12
                ]
            ).astype(np.float32)  # 34

        # Stack into history
        full_obs = self._stack_history(single_obs)

        # Run policy
        action = self._inference_fn(full_obs)
        action = np.clip(action, -1.0, 1.0)

        joint_targets = self._default_pose + action[:NUM_LEG_JOINTS] * ACTION_SCALE
        self._last_action = action[:NUM_LEG_JOINTS].astype(np.float32)
        return joint_targets, SkillStatus.RUNNING

    # ------------------------------------------------------------------
    # Fallback velocity computation
    # ------------------------------------------------------------------

    def _compute_fallback_velocity(self) -> tuple[float, float]:
        """Compute (vx, vyaw) from target direction for fallback mode.

        Returns:
            (vx, vyaw) velocity commands for BraxWalkSkill.
        """
        dist = self.target_distance
        bearing = self.target_bearing

        vx = float(np.clip(dist, 0.0, FALLBACK_VX_MAX))
        vyaw = float(
            np.clip(
                bearing * FALLBACK_VYAW_GAIN,
                -FALLBACK_VYAW_MAX,
                FALLBACK_VYAW_MAX,
            )
        )
        return vx, vyaw

    def _fallback_get_action(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, SkillStatus]:
        """Delegate to BraxWalkSkill with velocity computed from target."""
        if self._fallback_walk is None:
            return self._default_pose.copy(), SkillStatus.RUNNING

        vx, vyaw = self._compute_fallback_velocity()
        self._fallback_walk.set_command(vx=vx, vy=0.0, vyaw=vyaw)
        return self._fallback_walk.get_action(obs)

    def _fallback_get_action_from_telemetry(
        self,
        gyro: np.ndarray | None = None,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        joint_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SkillStatus]:
        """Delegate to BraxWalkSkill telemetry interface with fallback velocity."""
        if self._fallback_walk is None:
            return self._default_pose.copy(), SkillStatus.RUNNING

        vx, vyaw = self._compute_fallback_velocity()
        self._fallback_walk.set_command(vx=vx, vy=0.0, vyaw=vyaw)
        return self._fallback_walk.get_action_from_telemetry(
            gyro=gyro,
            imu_roll=imu_roll,
            imu_pitch=imu_pitch,
            joint_positions=joint_positions,
        )

    # ------------------------------------------------------------------
    # History stacking
    # ------------------------------------------------------------------

    def _stack_history(self, obs: np.ndarray) -> np.ndarray:
        """Push new obs into front of history buffer."""
        self._obs_history = np.roll(self._obs_history, obs.size)
        self._obs_history[: obs.size] = obs
        return self._obs_history.copy()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def default_pose(self) -> np.ndarray:
        """Default standing pose for leg joints."""
        return self._default_pose.copy()

    @property
    def is_loaded(self) -> bool:
        """Whether a native target-reaching checkpoint is loaded."""
        return self._inference_fn is not None

    @property
    def using_fallback(self) -> bool:
        """Whether the skill is using the fallback walk-based approach."""
        return self._using_fallback
