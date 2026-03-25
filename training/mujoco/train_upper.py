"""Train upper body skills using compositional environments.

Trains upper body policies (arms + head) while a frozen walking policy
controls the legs. Uses the same Brax PPO pipeline as the walking policy.

Usage:
    python3 -m training.mujoco.train_upper --task wave
    python3 -m training.mujoco.train_upper --task wave --num-timesteps 500000000
    python3 -m training.mujoco.train_upper --task wave --walking-checkpoint checkpoints/mujoco_locomotion_v13_flat_feet
"""

import argparse
import functools
import json
import time
from datetime import datetime
from pathlib import Path

import jax

from training.mujoco.train import make_ppo_config


TASKS = {
    "wave": {
        "module": "training.mujoco.wave_env",
        "class": "WaveEnv",
        "default_config_fn": "default_config",
        "default_checkpoint_dir": "checkpoints/mujoco_wave",
    },
}


def train_upper(
    task: str = "wave",
    walking_checkpoint: str = "checkpoints/mujoco_locomotion_v13_flat_feet",
    num_timesteps: int = 500_000_000,
    num_envs: int = 4096,
    checkpoint_dir: str | None = None,
    seed: int = 0,
    num_evals: int = 25,
):
    """Run PPO training for an upper body task."""
    import importlib
    from brax.training.agents.ppo.train import train as ppo_train
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.io import model as brax_model
    from mujoco_playground._src.wrapper import wrap_for_brax_training

    task_info = TASKS[task]
    module = importlib.import_module(task_info["module"])
    env_cls = getattr(module, task_info["class"])
    default_config_fn = getattr(module, task_info["default_config_fn"])

    print("=" * 60)
    print(f"AiNex Upper Body Training: {task}")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Create environment
    env_config = default_config_fn()
    env = env_cls(
        walking_checkpoint=walking_checkpoint,
        config=env_config,
    )

    print(f"Environment: {task_info['class']}")
    print(f"  Action size: {env.action_size} (upper body only)")
    print(f"  Walking checkpoint: {walking_checkpoint}")
    print(f"  Obs size: {env._config.obs_history_size * env._single_obs_size}")
    print(f"  Sim dt: {env_config.sim_dt}, Ctrl dt: {env_config.ctrl_dt}")
    print(f"  Episode length: {env_config.episode_length}")
    print()

    # PPO config — slightly smaller network for upper body
    ppo_cfg = make_ppo_config(num_timesteps, num_evals=num_evals)
    ppo_cfg["num_envs"] = num_envs
    ppo_cfg["policy_hidden_layer_sizes"] = (128, 128, 128)
    ppo_cfg["value_hidden_layer_sizes"] = (256, 256, 256)
    print(f"PPO Config:")
    for k, v in ppo_cfg.items():
        print(f"  {k}: {v}")
    print()

    # Checkpoint directory
    ckpt_dir = Path(checkpoint_dir or task_info["default_checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    obs_size = env._config.obs_history_size * env._single_obs_size
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "env": task_info["class"],
            "task": task,
            "walking_checkpoint": walking_checkpoint,
            "ppo": ppo_cfg,
            "env_config": {k: str(v) for k, v in dict(env_config).items()},
            "obs_size": obs_size,
            "action_size": env.action_size,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    # Training metrics
    train_metrics = []
    best_reward = float("-inf")
    start_time = time.time()

    def progress_callback(num_steps, metrics):
        nonlocal best_reward
        elapsed = time.time() - start_time
        reward = float(metrics.get("eval/episode_reward", 0))
        train_metrics.append({
            "steps": int(num_steps),
            "reward": reward,
            "elapsed": elapsed,
        })

        print(f"Step {num_steps:>10,} | "
              f"Reward: {reward:>8.2f} | "
              f"Time: {elapsed:>6.1f}s | "
              f"FPS: {num_steps/max(elapsed,1):.0f}",
              flush=True)

        if reward > best_reward:
            best_reward = reward
            print(f"  New best reward: {reward:.2f}", flush=True)

        with open(ckpt_dir / "metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)

    print("Starting training...")
    print()

    make_inference_fn, params, _ = ppo_train(
        environment=env,
        num_timesteps=ppo_cfg["num_timesteps"],
        episode_length=env_config.episode_length,
        num_evals=ppo_cfg["num_evals"],
        reward_scaling=ppo_cfg["reward_scaling"],
        normalize_observations=ppo_cfg["normalize_observations"],
        action_repeat=ppo_cfg["action_repeat"],
        unroll_length=ppo_cfg["unroll_length"],
        num_minibatches=ppo_cfg["num_minibatches"],
        num_updates_per_batch=ppo_cfg["num_updates_per_batch"],
        discounting=ppo_cfg["discounting"],
        learning_rate=ppo_cfg["learning_rate"],
        entropy_cost=ppo_cfg["entropy_cost"],
        num_envs=ppo_cfg["num_envs"],
        batch_size=ppo_cfg["batch_size"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        wrap_env=True,
        wrap_env_fn=functools.partial(wrap_for_brax_training),
        network_factory=lambda obs_size, action_size, preprocess_observations_fn: ppo_networks.make_ppo_networks(
            obs_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=ppo_cfg["policy_hidden_layer_sizes"],
            value_hidden_layer_sizes=ppo_cfg["value_hidden_layer_sizes"],
        ),
        seed=seed,
        progress_fn=progress_callback,
        save_checkpoint_path=str((ckpt_dir / "brax_ckpt").resolve()),
    )

    elapsed = time.time() - start_time
    print()
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best reward: {best_reward:.2f}")

    final_path = ckpt_dir / "final_params"
    brax_model.save_params(final_path, params)
    print(f"Saved params: {final_path}")

    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"\nCheckpoints saved to {ckpt_dir}/")
    return make_inference_fn, params


def main():
    parser = argparse.ArgumentParser(
        description="Train AiNex upper body skills"
    )
    parser.add_argument(
        "--task", type=str, default="wave",
        choices=list(TASKS.keys()),
        help=f"Task to train. Options: {list(TASKS.keys())}",
    )
    parser.add_argument(
        "--walking-checkpoint", type=str,
        default="checkpoints/mujoco_locomotion_v13_flat_feet",
        help="Path to frozen walking policy checkpoint",
    )
    parser.add_argument("--num-timesteps", type=int, default=500_000_000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-evals", type=int, default=25)
    args = parser.parse_args()

    train_upper(
        task=args.task,
        walking_checkpoint=args.walking_checkpoint,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        num_evals=args.num_evals,
    )


if __name__ == "__main__":
    main()
