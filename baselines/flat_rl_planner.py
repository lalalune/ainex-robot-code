"""Flat RL baseline: single end-to-end PPO policy with no hierarchy.

Maps the full 163-dim canonical observation (11 proprioception + 152 entity
slots) directly to a 7-dim action vector.  There is no planner/skill
separation -- the policy output *is* the action.

For evaluation compatibility the raw 7-dim action is wrapped in a
GroundedIntent dict with intent ``NAVIGATE_TO_POSITION`` and the decoded
walk/head controls stored in ``constraints``.

The policy network is a simple MLP loaded from a PPO training checkpoint.
If no checkpoint is available, the planner returns random actions within the
valid control range (useful for sanity-checking the evaluation harness).

Action dimensions (7):
    0: walk_x     [-0.05, 0.05]
    1: walk_y     [-0.05, 0.05]
    2: walk_yaw   [-10.0, 10.0]
    3: walk_height [0.015, 0.06]
    4: walk_speed  [1, 4]         (discretised)
    5: head_pan   [-1.5, 1.5]
    6: head_tilt  [-1.0, 1.0]
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base_planner import BasePlanner, grounded_intent, idle_intent
from training.schema.canonical import (
    AINEX_ACTION_DIM,
    AINEX_PROPRIO_DIM,
    AINEX_STATE_DIM,
    WALK_HEIGHT_MAX,
    WALK_HEIGHT_MIN,
    WALK_SPEED_MAX,
    WALK_SPEED_MIN,
    WALK_X_RANGE,
    WALK_Y_RANGE,
    WALK_YAW_RANGE,
    HEAD_PAN_RANGE,
    HEAD_TILT_RANGE,
    IMU_RANGE,
    BATTERY_MAX,
    BATTERY_MIN,
    denormalize_value,
    normalize_value,
)
from training.schema.embodied_context import EmbodiedContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight MLP (no torch dependency at import time)
# ---------------------------------------------------------------------------

class _NumpyMLP:
    """Minimal feedforward MLP using only numpy.

    Used when torch is not available.  Supports loading weights from a
    numpy ``.npz`` checkpoint with keys ``w0``, ``b0``, ``w1``, ``b1``,
    ``w2``, ``b2`` for a 3-layer MLP with tanh activations.
    """

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.w0 = weights["w0"]
        self.b0 = weights["b0"]
        self.w1 = weights["w1"]
        self.b1 = weights["b1"]
        self.w2 = weights["w2"]
        self.b2 = weights["b2"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x -> action (no batch dim)."""
        h = np.tanh(x @ self.w0 + self.b0)
        h = np.tanh(h @ self.w1 + self.b1)
        out = np.tanh(h @ self.w2 + self.b2)
        return out


# ---------------------------------------------------------------------------
# FlatRLPlanner
# ---------------------------------------------------------------------------

class FlatRLPlanner(BasePlanner):
    """Flat end-to-end RL policy baseline.

    This is **not** a planner in the traditional sense.  It maps the full
    163-dim observation directly to a 7-dim low-level action.  We wrap it
    in the ``BasePlanner`` interface so it can be evaluated with the same
    harness as the other planners.

    Parameters
    ----------
    checkpoint_path:
        Path to a trained PPO checkpoint.  Supported formats:
        - ``.pt`` / ``.pth``: PyTorch ``state_dict`` for a
          ``BridgePolicyNetwork(obs_dim=163, action_dim=7)``
        - ``.npz``: Numpy archive with MLP weight matrices
        If empty, the planner produces random valid actions.
    hidden_dim:
        Hidden layer width for the MLP (must match training).
    name:
        Planner name for logging.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        hidden_dim: int = 256,
        name: str = "flat_rl",
    ) -> None:
        super().__init__(name=name)
        self.checkpoint_path = checkpoint_path
        self.hidden_dim = hidden_dim

        self._torch_policy: Any = None  # Optional torch Module
        self._numpy_policy: _NumpyMLP | None = None
        self._policy_loaded = False

        if checkpoint_path:
            self._load_policy(checkpoint_path)

    # -- public API --------------------------------------------------------

    def plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        """Infer a raw 7-dim action from the full observation."""
        return self._timed_plan(context)

    def reset(self) -> None:
        super().reset()

    # -- internals ---------------------------------------------------------

    def _do_plan(self, context: dict[str, Any] | EmbodiedContext) -> dict[str, Any]:
        ctx = self._ensure_context(context)
        obs = self._context_to_obs(ctx)

        if self._torch_policy is not None:
            action = self._infer_torch(obs)
        elif self._numpy_policy is not None:
            action = self._numpy_policy.forward(obs)
        else:
            # Random action within valid ranges (normalised to [-1, 1])
            action = np.random.uniform(-1.0, 1.0, size=AINEX_ACTION_DIM).astype(np.float32)

        # Denormalise action to physical units
        walk_x = float(np.clip(action[0], -1, 1)) * WALK_X_RANGE
        walk_y = float(np.clip(action[1], -1, 1)) * WALK_Y_RANGE
        walk_yaw = float(np.clip(action[2], -1, 1)) * WALK_YAW_RANGE
        walk_height = denormalize_value(float(np.clip(action[3], -1, 1)), WALK_HEIGHT_MIN, WALK_HEIGHT_MAX)
        walk_speed = int(round(denormalize_value(float(np.clip(action[4], -1, 1)), WALK_SPEED_MIN, WALK_SPEED_MAX)))
        head_pan = float(np.clip(action[5], -1, 1)) * HEAD_PAN_RANGE
        head_tilt = float(np.clip(action[6], -1, 1)) * HEAD_TILT_RANGE

        return grounded_intent(
            intent="NAVIGATE_TO_POSITION",
            target_position=(walk_x, walk_y, 0.0),
            source_action_name="flat_rl_action",
            reasoning=(
                f"FlatRL: raw=[{', '.join(f'{a:.3f}' for a in action)}] "
                f"decoded: walk=({walk_x:.4f}, {walk_y:.4f}, yaw={walk_yaw:.2f}), "
                f"height={walk_height:.4f}, speed={walk_speed}, "
                f"head=({head_pan:.3f}, {head_tilt:.3f})"
            ),
            constraints=[
                f"action_dim={AINEX_ACTION_DIM}",
                f"walk_x={walk_x:.5f}",
                f"walk_y={walk_y:.5f}",
                f"walk_yaw={walk_yaw:.4f}",
                f"walk_height={walk_height:.5f}",
                f"walk_speed={walk_speed}",
                f"head_pan={head_pan:.4f}",
                f"head_tilt={head_tilt:.4f}",
            ],
        )

    # -- observation encoding ----------------------------------------------

    @staticmethod
    def _context_to_obs(ctx: EmbodiedContext) -> np.ndarray:
        """Encode an ``EmbodiedContext`` into the flat 163-dim observation.

        Layout:
        - [0:11]  proprioception (normalised)
        - [11:163] entity slots (already normalised in EmbodiedContext)
        """
        obs = np.zeros(AINEX_STATE_DIM, dtype=np.float32)

        # -- Proprioception (11 dims) ----------------------------------------
        # walk_x: already small, normalise by range
        obs[0] = 0.0  # walk_x unknown from context, default 0
        obs[1] = 0.0  # walk_y
        obs[2] = 0.0  # walk_yaw
        obs[3] = 0.0  # walk_height (default mid-range)
        obs[4] = 0.0  # walk_speed (default mid-range)
        obs[5] = 0.0  # head_pan
        obs[6] = 0.0  # head_tilt
        obs[7] = normalize_value(ctx.imu_roll, -IMU_RANGE, IMU_RANGE)
        obs[8] = normalize_value(ctx.imu_pitch, -IMU_RANGE, IMU_RANGE)
        obs[9] = 1.0 if ctx.is_walking else 0.0
        obs[10] = normalize_value(float(ctx.battery_mv), BATTERY_MIN, BATTERY_MAX) if ctx.battery_mv > 0 else 0.0

        # -- Entity slots (152 dims) -----------------------------------------
        entity_arr = ctx.to_entity_slots_array()
        n = min(len(entity_arr), AINEX_STATE_DIM - AINEX_PROPRIO_DIM)
        obs[AINEX_PROPRIO_DIM : AINEX_PROPRIO_DIM + n] = entity_arr[:n]

        return obs

    # -- policy loading ----------------------------------------------------

    def _load_policy(self, path: str) -> None:
        """Load a policy checkpoint from disk."""
        p = Path(path)
        if not p.exists():
            logger.warning("FlatRL checkpoint not found: %s", path)
            return

        suffix = p.suffix.lower()

        if suffix in (".pt", ".pth"):
            self._load_torch_policy(p)
        elif suffix == ".npz":
            self._load_numpy_policy(p)
        else:
            logger.warning("Unrecognised checkpoint format: %s", suffix)

    def _load_torch_policy(self, path: Path) -> None:
        """Load a PyTorch policy checkpoint."""
        try:
            import torch

            state = torch.load(str(path), map_location="cpu", weights_only=True)

            # Try to import project-specific policy class first
            try:
                from training.models.bridge_policy import BridgePolicyNetwork
                policy = BridgePolicyNetwork(
                    obs_dim=AINEX_STATE_DIM,
                    action_dim=AINEX_ACTION_DIM,
                )
            except ImportError:
                # Fallback: build a generic MLP
                policy = torch.nn.Sequential(
                    torch.nn.Linear(AINEX_STATE_DIM, self.hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.hidden_dim, AINEX_ACTION_DIM),
                    torch.nn.Tanh(),
                )

            policy.load_state_dict(state)
            policy.eval()
            self._torch_policy = policy
            self._policy_loaded = True
            logger.info("Loaded PyTorch flat RL policy from %s", path)
        except Exception:
            logger.exception("Failed to load PyTorch checkpoint: %s", path)

    def _load_numpy_policy(self, path: Path) -> None:
        """Load a numpy MLP checkpoint."""
        try:
            data = np.load(str(path))
            self._numpy_policy = _NumpyMLP(data)
            self._policy_loaded = True
            logger.info("Loaded numpy flat RL policy from %s", path)
        except Exception:
            logger.exception("Failed to load numpy checkpoint: %s", path)

    def _infer_torch(self, obs: np.ndarray) -> np.ndarray:
        """Run a forward pass through the PyTorch policy."""
        import torch
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = self._torch_policy(obs_tensor).squeeze(0)
            return action_tensor.numpy()
