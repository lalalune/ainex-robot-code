"""Wave-while-walking environment for AiNex.

Trains the right arm to perform a waving gesture while the frozen walking
policy controls the legs. The wave is a sinusoidal trajectory for the
right shoulder and elbow joints.

The robot walks forward at a moderate speed (0.3 m/s) while learning to
raise its right arm and wave it side to side. The left arm stays at the
default pose.

Extends CompositionalEnv: same frozen walking policy, same stability
rewards, plus wave trajectory tracking reward.

Usage:
    from training.mujoco.wave_env import WaveEnv, default_config
    env = WaveEnv()

    # Training:
    python3 -m training.mujoco.train_upper --task wave
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from training.mujoco import ainex_constants as consts
from training.mujoco.compositional_env import (
    CompositionalEnv,
    NUM_UPPER_JOINTS,
    default_config as base_default_config,
)


# Right arm joint indices within the upper body action space (12-dim):
# Upper body order: head_pan(0), head_tilt(1),
#   r_sho_pitch(2), r_sho_roll(3), r_el_pitch(4), r_el_yaw(5), r_gripper(6),
#   l_sho_pitch(7), l_sho_roll(8), l_el_pitch(9), l_el_yaw(10), l_gripper(11)
RIGHT_ARM_INDICES = [2, 3, 4, 5]  # shoulder pitch, roll, elbow pitch, yaw
NUM_WAVE_JOINTS = len(RIGHT_ARM_INDICES)


def default_config() -> config_dict.ConfigDict:
    """Default configuration for wave-while-walking."""
    cfg = base_default_config()

    # Upper body needs larger action range than legs
    cfg.action_scale = 1.5  # rad (legs use 0.3, arms need more range)

    # Walk forward while waving
    cfg.walk_command = [0.3, 0.0, 0.0]

    # Wave parameters
    cfg.wave_frequency = 2.0       # Hz — wave cycles per second
    cfg.wave_amplitude = 0.4       # rad — shoulder roll wave amplitude
    cfg.wave_shoulder_pitch = -1.2  # rad — raised arm target (negative = up)
    cfg.wave_elbow_pitch = -0.8    # rad — bent elbow target
    cfg.wave_elbow_yaw = 0.0       # rad — elbow yaw target

    # Task reward scales (added to base stability rewards)
    cfg.reward_config.scales.wave_tracking = 3.0     # track wave trajectory
    cfg.reward_config.scales.arm_raised = 2.0        # keep arm up
    cfg.reward_config.scales.left_arm_still = -0.5   # penalize left arm motion
    cfg.reward_config.scales.head_still = -0.3       # penalize unnecessary head motion

    return cfg


class WaveEnv(CompositionalEnv):
    """Wave the right arm while walking forward.

    The wave gesture:
        - Right shoulder pitch raised (arm goes up)
        - Right shoulder roll oscillates sinusoidally (wave side to side)
        - Right elbow bent
        - Left arm and head stay near default pose
    """

    def __init__(
        self,
        walking_checkpoint: str = "checkpoints/mujoco_locomotion_v13_flat_feet",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            walking_checkpoint=walking_checkpoint,
            config=config,
            config_overrides=config_overrides,
        )

    @property
    def _single_task_obs_size(self) -> int:
        """Task obs: wave_phase_sin(1) + wave_phase_cos(1) + wave_target(4) = 6."""
        return 6

    def _reset_task_info(
        self, info: dict, rng: jax.Array, data: mjx.Data
    ) -> dict:
        info["wave_phase"] = jp.float32(0.0)
        return info

    def _step_task_info_inplace(
        self, info: dict, data: mjx.Data, action: jax.Array
    ) -> None:
        # Advance wave phase
        phase = info["wave_phase"] + self._config.wave_frequency * self._config.ctrl_dt * 2.0 * jp.pi
        info["wave_phase"] = phase % (2.0 * jp.pi)

    def _get_wave_target(self, phase: jax.Array) -> jax.Array:
        """Compute the target right arm joint positions for the wave.

        Returns 4-dim target for [r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw].
        """
        sho_pitch = self._config.wave_shoulder_pitch
        sho_roll = self._config.wave_amplitude * jp.sin(phase)
        el_pitch = self._config.wave_elbow_pitch
        el_yaw = self._config.wave_elbow_yaw

        return jp.array([sho_pitch, sho_roll, el_pitch, el_yaw])

    def _get_task_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        """Task-specific obs: wave phase + target positions."""
        phase = info["wave_phase"]
        wave_target = self._get_wave_target(phase)
        return jp.concatenate([
            jp.array([jp.sin(phase)]),   # 1
            jp.array([jp.cos(phase)]),   # 1
            wave_target,                  # 4
        ])  # total = 6

    def _get_task_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        phase = info["wave_phase"]
        wave_target = self._get_wave_target(phase)

        # Current right arm joint positions (absolute, in radians)
        # Upper body qpos starts at index 7 + NUM_LEG_JOINTS = 19
        upper_qpos = data.qpos[7 + 12:]  # 12-dim: head(2) + r_arm(5) + l_arm(5)
        # Use static slice indices (not jp.array) for JIT compatibility
        right_arm_pos = jp.array([upper_qpos[i] for i in RIGHT_ARM_INDICES])

        # Default pose for upper body (numpy → jax for arithmetic)
        upper_default = jp.array(self._default_pose[12:])
        right_arm_default = jp.array([self._default_pose[12 + i] for i in RIGHT_ARM_INDICES])

        # Wave tracking: how close the right arm is to the target
        wave_error = jp.sum(jp.square(right_arm_pos - (right_arm_default + wave_target)))
        wave_tracking = jp.exp(-wave_error / 0.5)

        # Arm raised: bonus for having shoulder pitch near target
        sho_pitch_error = jp.square(
            right_arm_pos[0] - (right_arm_default[0] + self._config.wave_shoulder_pitch)
        )
        arm_raised = jp.exp(-sho_pitch_error / 0.3)

        # Left arm penalty: penalize deviation from default
        left_arm_pos = jp.array([upper_qpos[i] for i in range(7, 12)])
        left_arm_default = jp.array([self._default_pose[12 + i] for i in range(7, 12)])
        left_arm_still = jp.sum(jp.square(left_arm_pos - left_arm_default))

        # Head penalty: penalize deviation from default
        head_pos = upper_qpos[:2]  # head_pan, head_tilt
        head_default = jp.array(self._default_pose[12:14])
        head_still = jp.sum(jp.square(head_pos - head_default))

        return {
            "wave_tracking": wave_tracking,
            "arm_raised": arm_raised,
            "left_arm_still": left_arm_still,
            "head_still": head_still,
        }
