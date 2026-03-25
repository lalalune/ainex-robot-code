"""Evaluate a trained bridge-aligned policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.models.bridge_policy import BridgePolicyNetwork
from training.schema.canonical import AINEX_ACTION_DIM, AINEX_STATE_DIM

ACTION_NAMES: tuple[str, ...] = (
    "walk_x",
    "walk_y",
    "walk_yaw",
    "walk_height",
    "walk_speed",
    "head_pan",
    "head_tilt",
)


def load_bridge_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    states: list[list[float]] = []
    actions: list[list[float]] = []

    with dataset_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            raw = line.strip()
            if raw == "":
                continue
            record = json.loads(raw)
            state = record.get("state")
            action = record.get("action")
            if (
                isinstance(state, list)
                and len(state) == AINEX_STATE_DIM
                and isinstance(action, list)
                and len(action) == AINEX_ACTION_DIM
            ):
                states.append([float(value) for value in state])
                actions.append([float(value) for value in action])

    if not states:
        raise ValueError(f"No valid examples found in {dataset_path}")

    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def load_bridge_policy(checkpoint_path: Path) -> BridgePolicyNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "bridge_policy_v1":
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    hidden_dims_raw = checkpoint.get("hidden_dims", (128, 128))
    hidden_dims = (int(hidden_dims_raw[0]), int(hidden_dims_raw[1]))
    model = BridgePolicyNetwork(
        obs_dim=int(checkpoint.get("obs_dim", AINEX_STATE_DIM)),
        action_dim=int(checkpoint.get("action_dim", AINEX_ACTION_DIM)),
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_bridge_policy(
    checkpoint_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> dict[str, float]:
    states, actions = load_bridge_dataset(dataset_path)
    model = load_bridge_policy(checkpoint_path)

    with torch.no_grad():
        pred = model(torch.from_numpy(states)).cpu().numpy()

    mse = float(np.mean((pred - actions) ** 2))
    mae = float(np.mean(np.abs(pred - actions)))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "num_examples": float(states.shape[0]),
        "mse": mse,
        "mae": mae,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    sample_count = min(20, states.shape[0])
    fig, axes = plt.subplots(AINEX_ACTION_DIM, 1, figsize=(10, 2 * AINEX_ACTION_DIM), sharex=True)
    x = np.arange(sample_count)
    for index, axis in enumerate(axes):
        axis.plot(x, actions[:sample_count, index], label="target", marker="o")
        axis.plot(x, pred[:sample_count, index], label="pred", marker="x")
        axis.set_ylabel(ACTION_NAMES[index])
        axis.grid(True, alpha=0.3)
        if index == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_plot.png", dpi=150)
    plt.close(fig)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bridge-aligned policy checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    metrics = evaluate_bridge_policy(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
    )
    print(metrics)


if __name__ == "__main__":
    main()
