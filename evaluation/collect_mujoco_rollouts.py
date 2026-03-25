#!/usr/bin/env python3
"""Collect dense control trajectories from trained MuJoCo policies.

Runs trained PPO policies in the MuJoCo simulation environment at 50 Hz
control frequency and records full state/action/reward trajectories. Output
is JSONL format compatible with RLDS and LeRobot dataset converters.

Supports all trained task types:
    - locomotion: Joystick velocity tracking with random commands
    - target_reaching: Walk to randomly placed target positions
    - wave: Wave gesture while walking forward
    - compositional: Multi-task upper body + locomotion

Usage:
    # Collect locomotion rollouts
    python collect_mujoco_rollouts.py \\
        --task locomotion \\
        --episodes 1000 \\
        --checkpoint checkpoints/mujoco_locomotion_v13_flat_feet \\
        --output rollouts/

    # Collect target reaching rollouts
    python collect_mujoco_rollouts.py \\
        --task target_reaching \\
        --episodes 500 \\
        --checkpoint checkpoints/walk_to_target \\
        --output rollouts/

    # Collect all tasks
    python collect_mujoco_rollouts.py \\
        --task all \\
        --episodes 1000 \\
        --output rollouts/

    # With domain randomization variations
    python collect_mujoco_rollouts.py \\
        --task locomotion \\
        --episodes 2000 \\
        --domain-rand \\
        --output rollouts/

    # Export to LeRobot format after collection
    python collect_mujoco_rollouts.py \\
        --convert-lerobot \\
        --input rollouts/locomotion_rollouts.jsonl \\
        --output rollouts/lerobot/
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("collect_mujoco_rollouts")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent.parent.parent
AINEX_CODE = WORKSPACE / "ainex-robot-code"
CHECKPOINT_DIR = AINEX_CODE / "checkpoints"

CONTROL_DT = 0.02       # 50 Hz control frequency
EPISODE_LENGTH = 1000    # Steps per episode (20 seconds at 50 Hz)

# Default checkpoints per task
DEFAULT_CHECKPOINTS = {
    "locomotion": "mujoco_locomotion_v13_flat_feet",
    "target_reaching": "walk_to_target",
    "wave": "mujoco_wave_v2",
    "compositional": "multi_task",
}

# Velocity command sampling ranges for locomotion
VELOCITY_RANGES = {
    "forward": (-0.3, 1.2),   # m/s
    "lateral": (-0.4, 0.4),   # m/s
    "yaw": (-0.8, 0.8),       # rad/s
}


# ---------------------------------------------------------------------------
# Data tracking
# ---------------------------------------------------------------------------

@dataclass
class RolloutStats:
    """Track rollout collection statistics."""
    task: str
    target_episodes: int
    completed_episodes: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    successful_episodes: int = 0  # No early termination (fall)
    start_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.completed_episodes == 0:
            return 0.0
        return self.successful_episodes / self.completed_episodes

    @property
    def mean_reward(self) -> float:
        if self.completed_episodes == 0:
            return 0.0
        return self.total_reward / self.completed_episodes

    @property
    def mean_episode_length(self) -> float:
        if self.completed_episodes == 0:
            return 0.0
        return self.total_steps / self.completed_episodes

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        elapsed = self.elapsed_seconds
        eps = self.completed_episodes / max(elapsed, 1)
        return (
            f"Task: {self.task}\n"
            f"  Episodes: {self.completed_episodes}/{self.target_episodes}\n"
            f"  Total steps: {self.total_steps:,}\n"
            f"  Mean reward: {self.mean_reward:.2f}\n"
            f"  Mean length: {self.mean_episode_length:.0f} steps\n"
            f"  Success rate: {self.success_rate:.1%}\n"
            f"  Elapsed: {elapsed:.1f}s ({eps:.1f} ep/s)\n"
            f"  ETA: {(self.target_episodes - self.completed_episodes) / max(eps, 0.001):.0f}s"
        )


# ---------------------------------------------------------------------------
# Environment + policy loading
# ---------------------------------------------------------------------------

def load_task_env_and_policy(task: str, checkpoint: str, domain_rand: bool = False):
    """Load the MuJoCo environment and trained policy for a task.

    Returns:
        (env, policy_fn, config, obs_size, action_size)
    """
    import jax

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = CHECKPOINT_DIR / checkpoint

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    logger.info(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # Load config to detect env type
    config_path = ckpt_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            ckpt_config = json.load(f)
    else:
        ckpt_config = {}

    env_name = ckpt_config.get("env", "")

    if task == "locomotion" or (task == "auto" and "Target" not in env_name):
        from training.mujoco.joystick import Joystick, default_config
        env_config = default_config()
        env = Joystick(config=env_config)
    elif task == "target_reaching" or (task == "auto" and "Target" in env_name):
        from training.mujoco.target import TargetReaching, default_config
        env_config = default_config()
        env = TargetReaching(config=env_config)
    elif task == "wave":
        from training.mujoco.wave_env import WaveEnv, default_config
        env_config = default_config()
        env = WaveEnv(config=env_config)
    elif task == "compositional":
        from training.mujoco.compositional_env import CompositionalEnv, default_config
        env_config = default_config()
        env = CompositionalEnv(config=env_config)
    else:
        raise ValueError(f"Unknown task: {task}. Valid: locomotion, target_reaching, wave, compositional")

    # Load policy
    from training.mujoco.inference import load_policy_jax
    policy_fn, loaded_config, loaded_env = load_policy_jax(str(ckpt_path))

    obs_size = ckpt_config.get("obs_size", None)
    action_size = ckpt_config.get("action_size", env.action_size)

    logger.info(f"Environment: {type(env).__name__}")
    logger.info(f"  Obs size: {obs_size}, Action size: {action_size}")
    logger.info(f"  Episode length: {env_config.episode_length}")
    logger.info(f"  Control dt: {env_config.ctrl_dt}s ({1.0/env_config.ctrl_dt:.0f} Hz)")

    return env, policy_fn, ckpt_config, obs_size, action_size


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_single_episode(
    env,
    policy_fn,
    rng,
    episode_id: str,
    task: str,
    n_steps: int = EPISODE_LENGTH,
    obs_size: Optional[int] = None,
    action_size: Optional[int] = None,
    command: Optional[list[float]] = None,
) -> tuple[list[dict], dict]:
    """Run one episode and collect timestep data.

    Returns:
        (steps, episode_info) where steps is a list of per-timestep dicts
        and episode_info contains episode-level metadata.
    """
    import jax
    import jax.numpy as jp
    from training.schema.canonical import adapt_state_vector

    # Reset environment
    reset_rng, rng = jax.random.split(rng)
    state = jax.jit(env.reset)(reset_rng)

    # Set velocity command for locomotion
    if command is not None and "command" in state.info:
        state.info["command"] = jp.array(command)
    elif task == "locomotion" and "command" in state.info:
        # Random command for variety
        cmd_rng, rng = jax.random.split(rng)
        vx = float(jax.random.uniform(cmd_rng, minval=VELOCITY_RANGES["forward"][0],
                                       maxval=VELOCITY_RANGES["forward"][1]))
        cmd_rng, rng = jax.random.split(rng)
        vy = float(jax.random.uniform(cmd_rng, minval=VELOCITY_RANGES["lateral"][0],
                                       maxval=VELOCITY_RANGES["lateral"][1]))
        cmd_rng, rng = jax.random.split(rng)
        vyaw = float(jax.random.uniform(cmd_rng, minval=VELOCITY_RANGES["yaw"][0],
                                         maxval=VELOCITY_RANGES["yaw"][1]))
        state.info["command"] = jp.array([vx, vy, vyaw])
        command = [vx, vy, vyaw]

    step_fn = jax.jit(env.step)
    steps = []
    total_reward = 0.0
    fell = False

    for step_idx in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        obs = state.obs

        # Adapt observation size if needed
        if obs_size is not None and obs.shape[0] != obs_size:
            obs = jp.array(adapt_state_vector(np.array(obs).tolist(), obs_size))

        # Run policy
        action, _ = policy_fn(obs, act_rng)

        # Adapt action size if needed
        if action_size is not None and action.shape[0] != action_size:
            action = jp.array(adapt_state_vector(np.array(action).tolist(), action_size))

        # Step environment
        state = step_fn(state, action)

        # Extract info
        done = bool(state.done)
        reward = float(state.reward)
        total_reward += reward

        # Get robot state
        torso_body_id = env._torso_body_id
        torso_height = float(state.data.xpos[torso_body_id, 2])
        torso_xy = np.array(state.data.xpos[torso_body_id, :2]).tolist()

        # Get command for this step
        if "command" in state.info:
            step_command = np.array(state.info["command"]).tolist()
        elif "target_pos" in state.info:
            step_command = np.array(state.info["target_pos"]).tolist()
        else:
            step_command = [0.0, 0.0, 0.0]

        step_data = {
            "episode_id": episode_id,
            "step": step_idx,
            "timestamp": round(step_idx * CONTROL_DT, 4),
            "task": task,
            "observation": np.array(obs).tolist(),
            "action": np.array(action).tolist(),
            "reward": round(reward, 6),
            "done": done,
            "info": {
                "torso_height": round(torso_height, 4),
                "torso_xy": [round(v, 4) for v in torso_xy],
                "command": [round(v, 4) for v in step_command],
                "qpos": np.array(state.data.qpos).tolist(),
                "qvel": np.array(state.data.qvel).tolist(),
            },
        }
        steps.append(step_data)

        if done:
            fell = True
            break

    episode_info = {
        "episode_id": episode_id,
        "task": task,
        "steps": len(steps),
        "total_reward": round(total_reward, 4),
        "fell": fell,
        "command": command,
        "final_torso_height": round(float(state.data.xpos[torso_body_id, 2]), 4),
    }

    return steps, episode_info


def collect_rollouts(
    task: str,
    episodes: int,
    checkpoint: str,
    output_dir: Path,
    domain_rand: bool = False,
    seed: int = 0,
    command: Optional[list[float]] = None,
    compress: bool = True,
):
    """Collect multiple episodes of rollout data for a task."""
    import jax

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load environment and policy
    env, policy_fn, config, obs_size, action_size = load_task_env_and_policy(
        task, checkpoint, domain_rand
    )

    stats = RolloutStats(task=task, target_episodes=episodes)
    output_file = output_dir / f"{task}_rollouts.jsonl"

    logger.info(f"Collecting {episodes} episodes for task '{task}'")
    logger.info(f"Output: {output_file}")

    rng = jax.random.PRNGKey(seed)

    with open(output_file, "w") as f:
        for ep_idx in range(episodes):
            ep_rng, rng = jax.random.split(rng)
            episode_id = f"ep_{ep_idx:05d}"

            try:
                steps, ep_info = collect_single_episode(
                    env=env,
                    policy_fn=policy_fn,
                    rng=ep_rng,
                    episode_id=episode_id,
                    task=task,
                    n_steps=EPISODE_LENGTH,
                    obs_size=obs_size,
                    action_size=action_size,
                    command=command,
                )

                # Write each step as a JSONL line
                for step_data in steps:
                    f.write(json.dumps(step_data, separators=(",", ":")) + "\n")

                # Update stats
                stats.completed_episodes += 1
                stats.total_steps += len(steps)
                stats.total_reward += ep_info["total_reward"]
                if not ep_info["fell"]:
                    stats.successful_episodes += 1

                # Progress logging
                if (ep_idx + 1) % 100 == 0 or (ep_idx + 1) == episodes:
                    logger.info(f"\n{stats.summary()}")

            except Exception as exc:
                logger.error(f"Error in episode {episode_id}: {exc}")
                continue

    logger.info(f"\nCollection complete for task '{task}'")
    logger.info(stats.summary())

    # Compress
    if compress and output_file.exists():
        gz_path = output_dir / f"{task}_rollouts.jsonl.gz"
        logger.info(f"Compressing to {gz_path}...")
        with open(output_file, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                f_out.writelines(f_in)
        original_mb = output_file.stat().st_size / (1024 * 1024)
        compressed_mb = gz_path.stat().st_size / (1024 * 1024)
        logger.info(f"  {original_mb:.1f} MB -> {compressed_mb:.1f} MB "
                     f"({compressed_mb/max(original_mb,0.001)*100:.0f}%)")

    # Save stats
    stats_data = {
        "task": task,
        "episodes": stats.completed_episodes,
        "total_steps": stats.total_steps,
        "mean_reward": stats.mean_reward,
        "mean_episode_length": stats.mean_episode_length,
        "success_rate": stats.success_rate,
        "elapsed_seconds": stats.elapsed_seconds,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "domain_rand": domain_rand,
        "obs_size": obs_size,
        "action_size": action_size,
        "control_dt": CONTROL_DT,
        "collected_at": datetime.now().isoformat(),
    }
    stats_path = output_dir / f"{task}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def convert_to_rlds(input_path: Path, output_dir: Path):
    """Convert JSONL rollouts to RLDS-compatible TFRecord format.

    Requires tensorflow-datasets to be installed.
    """
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError:
        logger.error("tensorflow and tensorflow-datasets required for RLDS conversion")
        logger.error("Install with: pip install tensorflow tensorflow-datasets")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {input_path} to RLDS format at {output_dir}")

    # Read episodes
    episodes: dict[str, list[dict]] = {}
    with open(input_path) as f:
        for line in f:
            step = json.loads(line)
            ep_id = step["episode_id"]
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(step)

    # Build RLDS episodes
    def episode_generator():
        for ep_id, steps in sorted(episodes.items()):
            rlds_steps = []
            for i, s in enumerate(steps):
                obs_array = np.array(s["observation"], dtype=np.float32)
                action_array = np.array(s["action"], dtype=np.float32)
                command = s.get("info", {}).get("command", [0, 0, 0])

                # Build language instruction from command
                if s["task"] == "locomotion":
                    lang = (f"Walk at velocity: forward={command[0]:.2f} m/s, "
                            f"lateral={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s")
                elif s["task"] == "target_reaching":
                    lang = f"Walk to target at ({command[0]:.2f}, {command[1]:.2f})"
                elif s["task"] == "wave":
                    lang = "Wave while walking forward"
                else:
                    lang = f"Execute {s['task']} task"

                rlds_steps.append({
                    "observation": {"state": obs_array},
                    "action": action_array,
                    "reward": np.float32(s["reward"]),
                    "is_first": i == 0,
                    "is_last": i == len(steps) - 1,
                    "is_terminal": s["done"],
                    "language_instruction": lang,
                })
            yield {"steps": rlds_steps}

    # Write as TFRecord
    logger.info(f"Writing {len(episodes)} episodes to RLDS format")

    # Save as simple JSON-based RLDS for portability
    rlds_path = output_dir / "rlds_episodes.jsonl"
    count = 0
    with open(rlds_path, "w") as f:
        for ep_data in episode_generator():
            # Convert numpy arrays to lists for JSON
            serializable = {
                "steps": [
                    {
                        "observation": {"state": s["observation"]["state"].tolist()},
                        "action": s["action"].tolist(),
                        "reward": float(s["reward"]),
                        "is_first": s["is_first"],
                        "is_last": s["is_last"],
                        "is_terminal": s["is_terminal"],
                        "language_instruction": s["language_instruction"],
                    }
                    for s in ep_data["steps"]
                ]
            }
            f.write(json.dumps(serializable, separators=(",", ":")) + "\n")
            count += 1
    logger.info(f"Wrote {count} RLDS episodes to {rlds_path}")


def convert_to_lerobot(input_path: Path, output_dir: Path):
    """Convert JSONL rollouts to LeRobot dataset format.

    Creates a directory structure compatible with LeRobotDataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {input_path} to LeRobot format at {output_dir}")

    # Read all steps grouped by episode
    episodes: dict[str, list[dict]] = {}
    with open(input_path) as f:
        for line in f:
            step = json.loads(line)
            ep_id = step["episode_id"]
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(step)

    # Sort episode IDs for consistent indexing
    sorted_ep_ids = sorted(episodes.keys())
    ep_id_to_index = {ep_id: idx for idx, ep_id in enumerate(sorted_ep_ids)}

    # Collect all frames in LeRobot format
    frames = []
    for ep_id in sorted_ep_ids:
        steps = episodes[ep_id]
        ep_idx = ep_id_to_index[ep_id]
        for step in steps:
            frame = {
                "observation.state": step["observation"],
                "action": step["action"],
                "episode_index": ep_idx,
                "frame_index": step["step"],
                "timestamp": step["timestamp"],
                "next.reward": step["reward"],
                "next.done": step["done"],
            }
            frames.append(frame)

    # Write frames
    frames_path = output_dir / "data.jsonl"
    with open(frames_path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame, separators=(",", ":")) + "\n")

    # Write episode metadata
    episode_metadata = []
    for ep_id in sorted_ep_ids:
        steps = episodes[ep_id]
        ep_idx = ep_id_to_index[ep_id]
        episode_metadata.append({
            "episode_index": ep_idx,
            "episode_id": ep_id,
            "num_frames": len(steps),
            "total_reward": sum(s["reward"] for s in steps),
            "task": steps[0]["task"] if steps else "unknown",
        })

    meta_path = output_dir / "episodes.jsonl"
    with open(meta_path, "w") as f:
        for meta in episode_metadata:
            f.write(json.dumps(meta, separators=(",", ":")) + "\n")

    # Write dataset info
    sample_step = next(iter(next(iter(episodes.values()))), {})
    obs_dim = len(sample_step.get("observation", []))
    act_dim = len(sample_step.get("action", []))

    info = {
        "dataset_name": "ainex_control_trajectories",
        "robot_type": "ainex",
        "fps": int(1.0 / CONTROL_DT),
        "num_episodes": len(episodes),
        "num_frames": len(frames),
        "observation_dim": obs_dim,
        "action_dim": act_dim,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [obs_dim]},
            "action": {"dtype": "float32", "shape": [act_dim]},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
        "created_at": datetime.now().isoformat(),
    }
    info_path = output_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"LeRobot dataset: {len(episodes)} episodes, {len(frames)} frames")
    logger.info(f"  Observation dim: {obs_dim}, Action dim: {act_dim}")
    logger.info(f"  Output: {output_dir}")


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_stats(output_dir: Path):
    """Aggregate per-task stats into a single stats.json."""
    stats_files = sorted(output_dir.glob("*_stats.json"))
    if not stats_files:
        return

    combined = {
        "tasks": {},
        "total_episodes": 0,
        "total_steps": 0,
        "collected_at": datetime.now().isoformat(),
    }

    for sf in stats_files:
        with open(sf) as f:
            task_stats = json.load(f)
        task_name = task_stats.get("task", sf.stem.replace("_stats", ""))
        combined["tasks"][task_name] = task_stats
        combined["total_episodes"] += task_stats.get("episodes", 0)
        combined["total_steps"] += task_stats.get("total_steps", 0)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Aggregate stats: {combined['total_episodes']} episodes, "
                f"{combined['total_steps']} steps across {len(combined['tasks'])} tasks")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect dense control trajectories from MuJoCo policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks:
    locomotion        Joystick velocity tracking with random commands
    target_reaching   Walk to randomly placed target positions
    wave              Wave gesture while walking forward
    compositional     Multi-task upper body + locomotion
    all               Collect all tasks with default checkpoints

Examples:
    python collect_mujoco_rollouts.py --task locomotion --episodes 1000
    python collect_mujoco_rollouts.py --task all --episodes 500
    python collect_mujoco_rollouts.py --task locomotion --episodes 100 --command 0.5,0,0
    python collect_mujoco_rollouts.py --convert-lerobot --input rollouts/locomotion_rollouts.jsonl
        """,
    )
    parser.add_argument(
        "--task", type=str, default="locomotion",
        help="Task type: locomotion, target_reaching, wave, compositional, all (default: locomotion)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Number of episodes per task (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to policy checkpoint directory. Uses default per task if not specified.",
    )
    parser.add_argument(
        "--output", type=str, default="rollouts/",
        help="Output directory (default: rollouts/)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--domain-rand", action="store_true",
        help="Enable domain randomization during collection",
    )
    parser.add_argument(
        "--command", type=str, default=None,
        help="Fixed velocity command as comma-separated values: vx,vy,vyaw (locomotion only)",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Do not create gzipped copies of JSONL files",
    )

    # Format conversion flags
    convert_group = parser.add_argument_group("format conversion")
    convert_group.add_argument(
        "--convert-rlds", action="store_true",
        help="Convert collected JSONL to RLDS format",
    )
    convert_group.add_argument(
        "--convert-lerobot", action="store_true",
        help="Convert collected JSONL to LeRobot dataset format",
    )
    convert_group.add_argument(
        "--input", type=str, default=None,
        help="Input JSONL file for format conversion (used with --convert-*)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output).resolve()

    # Handle format conversion mode
    if args.convert_rlds or args.convert_lerobot:
        if not args.input:
            # Auto-discover JSONL files in output dir
            jsonl_files = sorted(output_dir.glob("*_rollouts.jsonl"))
            if not jsonl_files:
                parser.error("No JSONL files found. Specify --input or collect data first.")
        else:
            jsonl_files = [Path(args.input).resolve()]

        for jsonl_file in jsonl_files:
            if not jsonl_file.exists():
                logger.error(f"Input file not found: {jsonl_file}")
                continue
            task_name = jsonl_file.stem.replace("_rollouts", "")
            if args.convert_rlds:
                convert_to_rlds(jsonl_file, output_dir / "rlds" / task_name)
            if args.convert_lerobot:
                convert_to_lerobot(jsonl_file, output_dir / "lerobot" / task_name)
        return

    # Parse fixed command if specified
    command = None
    if args.command:
        try:
            command = [float(v) for v in args.command.split(",")]
            if len(command) != 3:
                parser.error("--command requires exactly 3 values: vx,vy,vyaw")
        except ValueError:
            parser.error("--command values must be numeric")

    # Determine tasks
    if args.task == "all":
        tasks = list(DEFAULT_CHECKPOINTS.keys())
    else:
        tasks = [args.task]

    # Validate tasks
    valid_tasks = set(DEFAULT_CHECKPOINTS.keys())
    for t in tasks:
        if t not in valid_tasks:
            parser.error(f"Unknown task '{t}'. Valid: {sorted(valid_tasks)}")

    # Collect rollouts for each task
    all_stats = []
    for task in tasks:
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS.get(task, "")
        if not checkpoint:
            logger.error(f"No checkpoint specified for task '{task}' and no default available")
            continue

        try:
            stats = collect_rollouts(
                task=task,
                episodes=args.episodes,
                checkpoint=checkpoint,
                output_dir=output_dir,
                domain_rand=args.domain_rand,
                seed=args.seed,
                command=command if task == "locomotion" else None,
                compress=not args.no_compress,
            )
            all_stats.append(stats)
        except FileNotFoundError as exc:
            logger.error(f"Checkpoint not found for task '{task}': {exc}")
            logger.error(f"Train first with: python -m training.mujoco.train {'--target' if task == 'target_reaching' else ''}")
            continue
        except Exception as exc:
            logger.error(f"Failed to collect rollouts for task '{task}': {exc}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate stats
    if all_stats:
        aggregate_stats(output_dir)

    # Auto-convert to RLDS/LeRobot if requested
    if args.convert_rlds or args.convert_lerobot:
        for task in tasks:
            jsonl_file = output_dir / f"{task}_rollouts.jsonl"
            if jsonl_file.exists():
                if args.convert_rlds:
                    convert_to_rlds(jsonl_file, output_dir / "rlds" / task)
                if args.convert_lerobot:
                    convert_to_lerobot(jsonl_file, output_dir / "lerobot" / task)

    logger.info("\nAll done.")
    total_eps = sum(s.completed_episodes for s in all_stats)
    total_steps = sum(s.total_steps for s in all_stats)
    logger.info(f"Total: {total_eps} episodes, {total_steps:,} steps across {len(tasks)} tasks")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
