"""Train a bridge-aligned 7-D policy from supervised traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.models.bridge_policy import BridgePolicyNetwork
from training.schema.canonical import AINEX_ACTION_DIM, AINEX_STATE_DIM


def load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    states: list[list[float]] = []
    actions: list[list[float]] = []

    with path.open("r", encoding="utf-8") as infile:
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
        raise ValueError(f"No valid training examples found in {path}")

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )


def train_bridge_policy(
    dataset_path: Path,
    output_path: Path,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> dict[str, float]:
    states, actions = load_dataset(dataset_path)
    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model = BridgePolicyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    final_loss = 0.0
    for _epoch in range(epochs):
        for batch_states, batch_actions in loader:
            optimizer.zero_grad()
            pred = model(batch_states)
            loss = loss_fn(pred, batch_actions)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "bridge_policy_v1",
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "hidden_dims": model.hidden_dims,
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )

    return {
        "examples": float(len(dataset)),
        "final_loss": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bridge-aligned policy from supervised traces")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train_bridge_policy(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(metrics)


if __name__ == "__main__":
    main()
