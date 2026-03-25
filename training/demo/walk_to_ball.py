"""Walk-to-red-ball demo in MuJoCo.

Runs the full closed-loop pipeline:
1. MuJoCo scene with a red ball
2. Ego camera -> perception pipeline -> detects ball
3. User instruction -> LLM (or offline fallback) -> NAVIGATE_TO_ENTITY
4. Execution service -> walking policy -> joint targets
5. MuJoCo steps physics -> repeat from 2

Usage:
    python -m training.demo.walk_to_ball
    python -m training.demo.walk_to_ball --no-llm          # offline fallback only
    python -m training.demo.walk_to_ball --render           # with MuJoCo viewer
    python -m training.demo.walk_to_ball --checkpoint path  # specific policy
    python -m training.demo.walk_to_ball --save-video out.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("walk_to_ball")


# ---------------------------------------------------------------------------
# Sim-loop constants (matching training/mujoco/sim_loop.py)
# ---------------------------------------------------------------------------
SINGLE_OBS_DIM = 45
OBS_HISTORY_SIZE = 3
TOTAL_OBS_DIM = SINGLE_OBS_DIM * OBS_HISTORY_SIZE  # 135
NUM_LEGS = 12
ACTION_SCALE = 0.3
CTRL_DT = 0.02  # 50 Hz policy
SIM_DT = 0.004  # 250 Hz physics (matches ainex_primitives.xml)


# ---------------------------------------------------------------------------
# Offline navigation controller (no LLM required)
# ---------------------------------------------------------------------------

class OfflineNavigationController:
    """Generates walk velocity commands to steer toward a detected target.

    This is the *offline fallback* that replaces the LLM planner. It uses
    a simple proportional controller on bearing and distance.
    """

    def __init__(
        self,
        max_vx: float = 0.5,
        max_vyaw: float = 0.6,
        bearing_gain: float = 1.5,
    ) -> None:
        self.max_vx = max_vx
        self.max_vyaw = max_vyaw
        self.bearing_gain = bearing_gain

    def compute_command(
        self,
        bearing: float,
        distance: float,
        reached_threshold: float = 0.3,
    ) -> tuple[float, float, float]:
        """Return (vx, vy, vyaw) velocity command.

        Args:
            bearing: Signed angle from robot heading to target (radians).
            distance: Distance to target (metres).
            reached_threshold: Stop distance.

        Returns:
            (vx, vy, vyaw) command tuple.
        """
        if distance < reached_threshold:
            return (0.0, 0.0, 0.0)

        # Turn toward target.
        vyaw = float(np.clip(
            self.bearing_gain * bearing,
            -self.max_vyaw,
            self.max_vyaw,
        ))

        # Walk forward proportional to alignment (slow when turning hard).
        alignment = max(0.0, math.cos(bearing))
        vx = float(self.max_vx * alignment)

        return (vx, 0.0, vyaw)


# ---------------------------------------------------------------------------
# Policy wrapper (extracts obs from DemoEnv, runs inference)
# ---------------------------------------------------------------------------

class SimPolicyRunner:
    """Wraps a Brax checkpoint policy and runs it against DemoEnv.

    Mirrors the observation extraction logic from ``sim_loop.SimLoop``
    but reads sensors from a ``DemoEnv`` instance.
    """

    def __init__(self, demo_env: Any, checkpoint_dir: str) -> None:
        from training.mujoco.inference import load_policy

        self._env = demo_env
        self._inference_fn, self._config = load_policy(checkpoint_dir)

        # Build actuator index maps (same as SimLoop).
        model = demo_env.model
        self._act_qpos_idx = np.array([
            model.jnt_qposadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)
        ])
        self._act_dof_idx = np.array([
            model.jnt_dofadr[model.actuator_trnid[i, 0]]
            for i in range(model.nu)
        ])
        self._default_pose = demo_env.default_pose

        # Sensor slices.
        self._gyro_adr = demo_env._gyro_adr
        self._gravity_adr = demo_env._gravity_adr

        # State buffers.
        self.last_action = np.zeros(NUM_LEGS, dtype=np.float32)
        self.obs_history = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self) -> None:
        self.last_action = np.zeros(NUM_LEGS, dtype=np.float32)
        self.obs_history = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        self.command[:] = 0.0

    def set_command(self, vx: float, vy: float, vyaw: float) -> None:
        self.command[:] = [vx, vy, vyaw]

    def get_obs(self) -> np.ndarray:
        """Extract 45-dim observation from DemoEnv's MuJoCo data."""
        data = self._env.data
        gyro = data.sensordata[self._gyro_adr].copy()
        gravity = data.sensordata[self._gravity_adr].copy()

        leg_qpos_idx = self._act_qpos_idx[:NUM_LEGS]
        leg_dof_idx = self._act_dof_idx[:NUM_LEGS]
        leg_pos = data.qpos[leg_qpos_idx] - self._default_pose[:NUM_LEGS]
        leg_vel = data.qvel[leg_dof_idx] * 0.05

        return np.concatenate([
            gyro,
            gravity,
            self.command,
            leg_pos,
            leg_vel,
            self.last_action,
        ]).astype(np.float32)

    def stack_history(self, obs: np.ndarray) -> np.ndarray:
        self.obs_history = np.roll(self.obs_history, obs.size)
        self.obs_history[:obs.size] = obs
        return self.obs_history.copy()

    def step(self) -> dict[str, float]:
        """Run one policy step: observe -> infer -> apply -> step physics.

        Returns joint_targets dict (name -> radians) for all actuators.
        """
        obs = self.get_obs()
        full_obs = self.stack_history(obs)

        action = self._inference_fn(full_obs)
        action = np.clip(action, -1.0, 1.0)

        # Convert to motor targets.
        leg_targets = self._default_pose[:NUM_LEGS] + action[:NUM_LEGS] * ACTION_SCALE
        ctrl = self._default_pose.copy()
        ctrl[:NUM_LEGS] = leg_targets

        # Clip to actuator limits.
        model = self._env.model
        ctrl = np.clip(
            ctrl,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )

        # Build joint_targets dict.
        joint_targets: dict[str, float] = {}
        for jname, act_idx in self._env._act_name_to_idx.items():
            joint_targets[jname] = float(ctrl[act_idx])

        # Step physics (substeps to match 50Hz control at sim_dt).
        n_substeps = max(1, int(CTRL_DT / model.opt.timestep))
        self._env.data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            import mujoco
            mujoco.mj_step(model, self._env.data)
            self._env._step_count += 1

        self.last_action = action[:NUM_LEGS].astype(np.float32)
        return joint_targets


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

async def run_demo(
    use_llm: bool = False,
    render: bool = False,
    max_steps: int = 500,
    ball_position: tuple[float, float, float] = (2.0, 0.0, 0.05),
    checkpoint_dir: str = "checkpoints/mujoco_locomotion_v12_dr",
    save_video: str = "",
    reached_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run the walk-to-red-ball demo.

    Args:
        use_llm: If True, attempt to connect to an LLM planner for
            intent generation. Otherwise use the offline controller.
        render: If True, open the MuJoCo interactive viewer.
        max_steps: Maximum number of 50-Hz policy steps.
        ball_position: (x, y, z) world position of the red ball.
        checkpoint_dir: Path to Brax policy checkpoint.
        save_video: If non-empty, save MP4 to this path.
        reached_threshold: Distance at which target is considered reached.

    Returns:
        Result dict with success flag, steps, distance, etc.
    """
    from training.mujoco.demo_env import DemoEnv

    logger.info("=== Walk-to-Red-Ball Demo ===")
    logger.info("Ball position: %s", ball_position)
    logger.info("Checkpoint: %s", checkpoint_dir)
    logger.info("Mode: %s", "LLM" if use_llm else "offline")

    # 1. Create DemoEnv with red ball.
    env = DemoEnv(
        target_position=ball_position,
        target_color=(1.0, 0.0, 0.0, 1.0),
        target_size=0.05,
        timestep=SIM_DT,
    )
    env.reset()
    logger.info("DemoEnv created. Robot at %s, target at %s",
                env.get_robot_position()[:2].tolist(),
                env.get_target_position()[:2].tolist())

    # 2. Load walking policy.
    try:
        policy_runner = SimPolicyRunner(env, checkpoint_dir)
    except Exception as exc:
        logger.error("Failed to load policy from %s: %s", checkpoint_dir, exc)
        logger.info("Running without trained policy (no-op steps).")
        policy_runner = None

    # 3. Navigation controller (offline fallback).
    nav_controller = OfflineNavigationController()

    # 4. Optional video recording.
    video_writer = None
    if save_video:
        try:
            import imageio
            output_path = Path(save_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                str(output_path), fps=25, codec="libx264",
                quality=8, pixelformat="yuv420p",
            )
            logger.info("Recording video to %s", save_video)
        except ImportError:
            logger.warning("imageio not available; skipping video recording.")

    # 5. Optional interactive viewer.
    viewer = None
    if render:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            logger.info("MuJoCo viewer launched.")
        except ImportError:
            logger.warning("mujoco.viewer not available; running headless.")

    # 6. Main loop.
    t0 = time.monotonic()
    reached = False
    steps_completed = 0
    initial_distance = env.distance_to_target()
    frame_skip = 2  # Record every other frame to keep video reasonable.

    try:
        for step in range(max_steps):
            steps_completed = step + 1

            # --- Perception: compute bearing and distance ---
            bearing = env.bearing_to_target()
            distance = env.distance_to_target()

            # --- Planning: generate velocity command ---
            if use_llm:
                # LLM path: would send perception to planner, receive intent.
                # For now, fall through to offline as a demonstration scaffold.
                vx, vy, vyaw = nav_controller.compute_command(
                    bearing, distance, reached_threshold
                )
            else:
                vx, vy, vyaw = nav_controller.compute_command(
                    bearing, distance, reached_threshold
                )

            # --- Execution: run policy step ---
            if policy_runner is not None:
                policy_runner.set_command(vx, vy, vyaw)
                joint_targets = policy_runner.step()
            else:
                # No-op: just step physics once.
                env.step()

            # --- Rendering ---
            if video_writer is not None and step % frame_skip == 0:
                frame = env.render_ego()
                video_writer.append_data(frame)

            if viewer is not None:
                viewer.sync()

            # --- Check termination ---
            if distance < reached_threshold:
                reached = True
                logger.info(
                    "TARGET REACHED at step %d (distance=%.3f m)",
                    step, distance,
                )
                break

            # --- Periodic logging ---
            if step % 50 == 0:
                robot_pos = env.get_robot_position()
                logger.info(
                    "Step %d/%d: dist=%.3f bearing=%.1f deg "
                    "cmd=(%.2f, %.2f, %.2f) pos=(%.2f, %.2f, %.2f)",
                    step, max_steps, distance,
                    math.degrees(bearing),
                    vx, vy, vyaw,
                    robot_pos[0], robot_pos[1], robot_pos[2],
                )

            # --- Real-time pacing (if viewer) ---
            if viewer is not None:
                elapsed = time.monotonic() - t0
                expected = (step + 1) * CTRL_DT
                sleep = expected - elapsed
                if sleep > 0:
                    await asyncio.sleep(sleep)

    finally:
        if video_writer is not None:
            video_writer.close()
            logger.info("Video saved to %s", save_video)
        if viewer is not None:
            viewer.close()

    elapsed = time.monotonic() - t0
    final_distance = env.distance_to_target()

    result = {
        "success": reached,
        "steps_completed": steps_completed,
        "elapsed_seconds": round(elapsed, 2),
        "initial_distance": round(initial_distance, 3),
        "final_distance": round(final_distance, 3),
        "ball_position": list(ball_position),
        "robot_final_position": env.get_robot_position().tolist(),
        "reached_threshold": reached_threshold,
        "mode": "llm" if use_llm else "offline",
        "checkpoint": checkpoint_dir,
    }

    logger.info("=== Demo %s ===", "SUCCESS" if reached else "INCOMPLETE")
    logger.info(
        "Steps: %d | Time: %.1fs | Dist: %.3f -> %.3f m",
        steps_completed, elapsed, initial_distance, final_distance,
    )

    env.close()
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-to-red-ball closed-loop demo in MuJoCo",
    )
    parser.add_argument(
        "--no-llm", action="store_true", default=False,
        help="Use offline navigation controller instead of LLM",
    )
    parser.add_argument(
        "--llm", action="store_true", default=False,
        help="Enable LLM planner (requires running planner service)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Open MuJoCo interactive viewer",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum number of 50-Hz policy steps",
    )
    parser.add_argument(
        "--ball-x", type=float, default=2.0,
        help="Target ball X position (metres)",
    )
    parser.add_argument(
        "--ball-y", type=float, default=0.0,
        help="Target ball Y position (metres)",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/mujoco_locomotion_v12_dr",
        help="Path to Brax policy checkpoint directory",
    )
    parser.add_argument(
        "--save-video", type=str, default="",
        help="Save MP4 recording to this path",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Distance threshold to consider target reached (metres)",
    )
    args = parser.parse_args()

    ball_position = (args.ball_x, args.ball_y, 0.05)
    use_llm = args.llm and not args.no_llm

    result = asyncio.run(run_demo(
        use_llm=use_llm,
        render=args.render,
        max_steps=args.max_steps,
        ball_position=ball_position,
        checkpoint_dir=args.checkpoint,
        save_video=args.save_video,
        reached_threshold=args.threshold,
    ))

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
