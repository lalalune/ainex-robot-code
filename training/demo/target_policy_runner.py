"""Run a trained TargetReaching policy in DemoEnv with proper observations.

Builds the exact observation vector that the TargetReaching env produces
during training: gyro(3) + gravity(3) + target_vec(2) + target_dist(1) +
target_bearing(1) + joint_pos(24) + last_act(24) = 58 per frame × 3 history = 174.

Usage:
    python -m training.demo.target_policy_runner
    python -m training.demo.target_policy_runner --checkpoint checkpoints/mujoco_target_v2
    python -m training.demo.target_policy_runner --ball-x 1.5 --ball-y 0.5 --steps 500
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import numpy as np

from training.mujoco.demo_env import DemoEnv
from training.mujoco.inference import load_policy
from training.mujoco import ainex_constants as consts


SINGLE_OBS_DIM = 58  # gyro(3) + gravity(3) + target(4) + joint_pos(24) + last_act(24)
OBS_HISTORY = 3
TOTAL_OBS_DIM = SINGLE_OBS_DIM * OBS_HISTORY
ACTION_SCALE = 0.3
NUM_ACTUATORS = consts.NUM_ACTUATORS  # 24


class TargetPolicyRunner:
    """Runs TargetReaching policy in DemoEnv with proper obs construction."""

    def __init__(
        self,
        checkpoint: str = "checkpoints/mujoco_target",
        target_position: tuple[float, float, float] = (2.0, 0.0, 0.05),
    ):
        self.env = DemoEnv(target_position=target_position)
        self.model = self.env.model
        self.data = self.env.data

        # Load policy
        self.policy_fn, self.config = load_policy(checkpoint)
        expected_obs = self.config.get("obs_size", TOTAL_OBS_DIM)
        print(f"Policy loaded: obs_size={expected_obs} action_size={self.config.get('action_size')}")

        # Sensor addresses
        self._gyro_adr = self._sensor_adr("gyro", 3)
        self._gravity_adr = self._sensor_adr("upvector", 3)

        # Actuator → qpos/dof index maps
        self._act_qpos_idx = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])
        self._act_dof_idx = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])

        # Default pose (qpos at init)
        self._default_pose = self.data.qpos[7:7 + NUM_ACTUATORS].copy()

        # State
        self.obs_history = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        self.last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)

    def _sensor_adr(self, name: str, dim: int) -> slice:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return slice(0, dim)
        adr = self.model.sensor_adr[sid]
        return slice(adr, adr + dim)

    def get_obs(self) -> np.ndarray:
        """Build the 58-dim single-frame observation matching TargetReaching._get_obs."""
        gyro = self.data.sensordata[self._gyro_adr].copy()
        gravity = self.data.sensordata[self._gravity_adr].copy()

        # Target in body frame
        robot_xy = self.data.qpos[:2].copy()
        target_xy = self.env.get_target_position()[:2]
        delta = target_xy - robot_xy

        yaw = self.env.get_robot_yaw()
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        target_vec = np.array([
            delta[0] * cos_yaw - delta[1] * sin_yaw,
            delta[0] * sin_yaw + delta[1] * cos_yaw,
        ], dtype=np.float32)

        target_dist = float(np.linalg.norm(delta))
        target_angle = math.atan2(delta[1], delta[0])
        target_bearing = math.atan2(
            math.sin(target_angle - yaw),
            math.cos(target_angle - yaw),
        )

        # Joint positions (all 24, relative to default)
        joint_pos = self.data.qpos[7:7 + NUM_ACTUATORS] - self._default_pose

        obs = np.concatenate([
            gyro,                                          # 3
            gravity,                                       # 3
            target_vec,                                    # 2
            np.array([target_dist], dtype=np.float32),     # 1
            np.array([target_bearing], dtype=np.float32),  # 1
            joint_pos,                                     # 24
            self.last_action,                              # 24
        ]).astype(np.float32)  # 58

        return obs

    def stack_history(self, obs: np.ndarray) -> np.ndarray:
        self.obs_history = np.roll(self.obs_history, obs.size)
        self.obs_history[:obs.size] = obs
        return self.obs_history.copy()

    def step(self) -> tuple[np.ndarray, dict]:
        """Run one policy step. Returns (action, telemetry)."""
        obs = self.get_obs()
        full_obs = self.stack_history(obs)

        action = self.policy_fn(full_obs)
        if isinstance(action, tuple):
            action = action[0]
        action = np.array(action, dtype=np.float32).flatten()[:NUM_ACTUATORS]
        action = np.clip(action, -1.0, 1.0)

        # Apply scaled action as joint targets
        targets = self._default_pose + action * ACTION_SCALE
        joint_dict = {
            consts.ALL_JOINT_NAMES[i]: float(targets[i])
            for i in range(NUM_ACTUATORS)
        }

        self.last_action = action.copy()
        tel = self.env.step(joint_dict)
        return action, tel

    def run(self, num_steps: int = 500, print_interval: int = 50) -> dict:
        """Run the policy for N steps, return summary."""
        self.env.reset()
        self.obs_history[:] = 0.0
        self.last_action[:] = 0.0

        print(f"\nTarget: {self.env.get_target_position()}")
        print(f"Initial distance: {self.env.distance_to_target():.3f}m")
        print()
        print(f"{'Step':>5} | {'Pos XY':>12} | {'Dist':>6} | {'Pitch':>7} | {'Roll':>7} | {'L_foot':>7} | {'R_foot':>7}")
        print("-" * 70)

        min_dist = self.env.distance_to_target()
        fell = False

        for step in range(num_steps):
            action, tel = self.step()

            dist = self.env.distance_to_target()
            min_dist = min(min_dist, dist)

            if abs(tel["imu_pitch"]) > 1.0 or abs(tel["imu_roll"]) > 1.0:
                fell = True
                print(f"  *** FELL at step {step} (pitch={tel['imu_pitch']:.2f}, roll={tel['imu_roll']:.2f})")
                break

            if step % print_interval == 0:
                pos = self.env.get_robot_position()
                # Foot heights
                mujoco.mj_forward(self.model, self.data)
                l_z = self.data.site_xpos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
                ][2]
                r_z = self.data.site_xpos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
                ][2]
                print(f"{step:5d} | ({pos[0]:+5.2f},{pos[1]:+5.2f}) | {dist:5.2f}m | "
                      f"{tel['imu_pitch']:+.4f} | {tel['imu_roll']:+.4f} | "
                      f"{l_z:.4f}  | {r_z:.4f}")

            if self.env.is_target_reached(0.15):
                print(f"\n  *** TARGET REACHED at step {step}! ***")
                break

        result = {
            "steps": step + 1,
            "final_distance": self.env.distance_to_target(),
            "min_distance": min_dist,
            "fell": fell,
            "reached": self.env.is_target_reached(0.15),
        }
        print(f"\nResult: dist={result['final_distance']:.3f}m min={result['min_distance']:.3f}m fell={fell} reached={result['reached']}")
        return result

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Run target-reaching policy in MuJoCo")
    parser.add_argument("--checkpoint", default="checkpoints/mujoco_target")
    parser.add_argument("--ball-x", type=float, default=2.0)
    parser.add_argument("--ball-y", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    runner = TargetPolicyRunner(
        checkpoint=args.checkpoint,
        target_position=(args.ball_x, args.ball_y, 0.05),
    )
    runner.run(num_steps=args.steps)
    runner.close()


if __name__ == "__main__":
    main()
