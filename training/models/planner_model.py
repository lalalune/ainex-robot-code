"""Simple planner action classifier.

Given a game state (entity features, player state), predict the canonical action.
This is the first planner model trained on Hyperscape behavior tick data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Action label vocabulary
ACTION_LABELS = [
    "streaming_duel",
    "idle",
    "gather",
    "move",
    "attack",
    "pickup",
    "questAccept",
    "questComplete",
    "lootGravestone",
    "firemake",
    "stop",
]
ACTION_TO_IDX = {label: idx for idx, label in enumerate(ACTION_LABELS)}
NUM_ACTIONS = len(ACTION_LABELS)

# Feature dimensions
MAX_ENTITIES = 10
ENTITY_FEATURE_DIM = 4  # type_idx, distance, health_norm, is_player
PLAYER_FEATURE_DIM = 5  # health_frac, in_combat, pos_x, pos_y, pos_z
FEATURE_DIM = PLAYER_FEATURE_DIM + MAX_ENTITIES * ENTITY_FEATURE_DIM

ENTITY_TYPES = ["player", "npc", "mob", "resource", "item", "object", "landmark", "unknown"]
ENTITY_TYPE_TO_IDX = {t: i / len(ENTITY_TYPES) for i, t in enumerate(ENTITY_TYPES)}


def encode_dataset_row(row: dict) -> tuple[np.ndarray, int] | None:
    """Encode a planner dataset row into (features, action_idx)."""
    action = row.get("selected_canonical_action", "")
    if action not in ACTION_TO_IDX:
        return None

    context = row.get("canonical_planner_context", {})
    player = context.get("player", {})
    entities = context.get("entities", [])

    # Player features
    health = player.get("health", 0) or 0
    max_health = player.get("maxHealth", 1) or 1
    health_frac = health / max(max_health, 1)
    in_combat = 1.0 if player.get("inCombat", False) else 0.0
    pos = player.get("position", [0, 0, 0]) or [0, 0, 0]

    player_features = [
        health_frac,
        in_combat,
        float(pos[0]) / 100.0 if len(pos) > 0 else 0.0,
        float(pos[1]) / 100.0 if len(pos) > 1 else 0.0,
        float(pos[2]) / 100.0 if len(pos) > 2 else 0.0,
    ]

    # Entity features (up to MAX_ENTITIES)
    entity_features = []
    for i in range(MAX_ENTITIES):
        if i < len(entities) and isinstance(entities[i], dict):
            e = entities[i]
            etype = ENTITY_TYPE_TO_IDX.get(e.get("type", "unknown"), 0.875)
            dist = float(e.get("distance", 0)) / 100.0
            e_health = float(e.get("health", 0) or 0) / 100.0
            is_player = 1.0 if e.get("type") == "player" else 0.0
            entity_features.extend([etype, dist, e_health, is_player])
        else:
            entity_features.extend([0.0, 0.0, 0.0, 0.0])

    features = np.array(player_features + entity_features, dtype=np.float32)
    return features, ACTION_TO_IDX[action]


class PlannerActionClassifier(nn.Module):
    """Simple MLP classifier for planner action prediction."""

    def __init__(self, input_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load planner dataset and encode into features/labels."""
    features_list = []
    labels_list = []

    with path.open("r") as f:
        for line in f:
            row = json.loads(line.strip())
            result = encode_dataset_row(row)
            if result is not None:
                features_list.append(result[0])
                labels_list.append(result[1])

    return np.array(features_list), np.array(labels_list)
