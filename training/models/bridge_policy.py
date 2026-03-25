"""Bridge-aligned 7-D policy model.

This model consumes the canonical bridge observation vector and predicts the
canonical 7-D walk/head action vector expected by the OpenPI/bridge runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from training.schema.canonical import AINEX_ACTION_DIM, AINEX_STATE_DIM


class BridgePolicyNetwork(nn.Module):
    """Small MLP for canonical bridge observations."""

    def __init__(
        self,
        obs_dim: int = AINEX_STATE_DIM,
        action_dim: int = AINEX_ACTION_DIM,
        hidden_dims: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


@dataclass(frozen=True)
class BridgePolicyCheckpoint:
    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, int]
    state_dict: dict[str, torch.Tensor]
