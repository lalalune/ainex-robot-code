"""OpenPI-compatible inference server for trained AiNex policies.

Wraps a trained policy checkpoint in an HTTP server that matches the
OpenPI inference protocol. This allows the existing openpi_loop.py to
deploy trained policies without any code changes.

Usage:
    python3 -m training.serve_policy \
        --checkpoint checkpoints/walk_stable/best.pt \
        --port 8000

    # Then run the policy loop pointing at this server:
    python3 -m training.runtime.openpi_loop \
        --bridge-uri ws://localhost:9100 \
        --openpi-url http://localhost:8000 \
        --task "walk forward" --hz 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np

from training.models.bridge_policy import BridgePolicyNetwork
from training.schema.canonical import (
    AINEX_ACTION_DIM,
    AINEX_SCHEMA_VERSION,
    AINEX_STATE_DIM,
    adapt_state_vector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy server
# ---------------------------------------------------------------------------

class PolicyServer:
    """Wraps a trained policy for inference serving."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._policy: object | None = None
        self._use_torch = False
        self._expected_obs_dim = AINEX_STATE_DIM
        self._bridge_compatible = False

    def load(self) -> None:
        """Load the policy from checkpoint.

        Supports Brax/JAX checkpoints (via training.mujoco.inference)
        and legacy PyTorch checkpoints.
        """
        ckpt_path = Path(self._checkpoint_path)

        if self._checkpoint_path == "baseline://forward":
            class _BaselineForwardPolicy:
                obs_dim = AINEX_STATE_DIM
                action_dim = AINEX_ACTION_DIM

                def get_action(self, obs: np.ndarray) -> np.ndarray:
                    _ = obs
                    return np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            self._policy = _BaselineForwardPolicy()
            self._use_torch = False
            self._expected_obs_dim = AINEX_STATE_DIM
            self._bridge_compatible = True
            logger.info("Loaded built-in baseline forward policy")
            return

        # Try Brax checkpoint first (has config.json + final_params)
        if (ckpt_path / "config.json").exists():
            from training.mujoco.inference import load_policy
            inference_fn, config = load_policy(str(ckpt_path))
            self._expected_obs_dim = int(config.get("obs_size", AINEX_STATE_DIM))

            class _BraxPolicyWrapper:
                def __init__(self, fn):
                    self._fn = fn
                    self.action_dim = int(config.get("action_size", AINEX_ACTION_DIM))
                def get_action(self, obs):
                    return self._fn(obs)
                def get_deterministic_action(self, obs_t):
                    import torch
                    obs_np = obs_t.squeeze(0).cpu().numpy()
                    action_np = self._fn(obs_np)
                    return torch.from_numpy(action_np).unsqueeze(0)

            self._policy = _BraxPolicyWrapper(inference_fn)
            self._use_torch = False
            self._bridge_compatible = int(config.get("action_size", 0)) == AINEX_ACTION_DIM
            logger.info(f"Loaded Brax policy from {self._checkpoint_path}")
            return

        # Legacy PyTorch checkpoint
        try:
            import torch
            ckpt = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)
            if isinstance(ckpt, dict) and ckpt.get("format") == "bridge_policy_v1":
                hidden_dims = tuple(int(value) for value in ckpt.get("hidden_dims", (128, 128)))
                model = BridgePolicyNetwork(
                    obs_dim=int(ckpt.get("obs_dim", AINEX_STATE_DIM)),
                    action_dim=int(ckpt.get("action_dim", AINEX_ACTION_DIM)),
                    hidden_dims=(hidden_dims[0], hidden_dims[1]),
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self._policy = model
                self._use_torch = True
                self._expected_obs_dim = model.obs_dim
                self._bridge_compatible = model.action_dim == AINEX_ACTION_DIM
                logger.info(f"Loaded bridge policy from {self._checkpoint_path}")
                return
            if hasattr(ckpt, "eval"):
                self._policy = ckpt
            elif isinstance(ckpt, dict) and "model" in ckpt and not isinstance(ckpt["model"], dict):
                self._policy = ckpt["model"]
                self._policy.eval()
            else:
                logger.warning(f"Cannot load policy from {ckpt_path}: unrecognized format")
                return
            self._use_torch = True
            self._expected_obs_dim = int(getattr(self._policy, "obs_dim", AINEX_STATE_DIM))
            self._bridge_compatible = self._expected_obs_dim > 0 and int(getattr(self._policy, "action_dim", AINEX_ACTION_DIM)) == AINEX_ACTION_DIM
            logger.info(f"Loaded PyTorch policy from {self._checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run inference on an observation.

        Args:
            observation: Dict matching OpenPI observation schema:
                - state: list of floats (canonical normalized bridge observation)
                - prompt: str (task description)
                - image: str (optional, base64 JPEG)

        Returns:
            Dict matching OpenPI action schema:
                - action: list of 7 floats (normalized action vector)
                - confidence: float (0-1)
        """
        if self._policy is None:
            return {
                "action": [0.0] * AINEX_ACTION_DIM,
                "confidence": 0.0,
                "schema_version": AINEX_SCHEMA_VERSION,
            }
        if not self._bridge_compatible:
            raise RuntimeError(
                "Checkpoint is not bridge-compatible: expected a 7-D canonical action policy"
            )

        # Extract state
        state = observation.get("state", [0.0] * AINEX_STATE_DIM)
        if not isinstance(state, list):
            state = [0.0] * AINEX_STATE_DIM
        obs = np.array(
            adapt_state_vector(state, self._expected_obs_dim),
            dtype=np.float32,
        )

        # Get action
        if self._use_torch:
            import torch
            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                action = self._policy.get_deterministic_action(obs_t)  # type: ignore[attr-defined]
                action_np = action.squeeze(0).cpu().numpy()
        elif hasattr(self._policy, "get_action"):
            action_np = self._policy.get_action(obs)  # type: ignore[attr-defined]
        else:
            action_np = np.zeros(AINEX_ACTION_DIM, dtype=np.float32)

        action_list = np.clip(action_np, -1.0, 1.0).tolist()

        return {
            "action": action_list,
            "confidence": 0.95,  # Trained policy has high confidence
            "schema_version": AINEX_SCHEMA_VERSION,
            "expected_obs_dim": self._expected_obs_dim,
        }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_server_instance: PolicyServer | None = None


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for /infer endpoint."""

    def do_POST(self) -> None:
        if self.path not in {"/infer", "/act"}:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            observation = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        start = time.monotonic()
        try:
            result = _server_instance.infer(observation)  # type: ignore[union-attr]
        except RuntimeError as error:
            response = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return
        elapsed_ms = (time.monotonic() - start) * 1000

        result["inference_ms"] = round(elapsed_ms, 2)

        response = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:
        """Health check endpoint."""
        if self.path == "/health":
            response = json.dumps({
                "status": "ok",
                "checkpoint": _server_instance._checkpoint_path if _server_instance else "",
                "schema_version": AINEX_SCHEMA_VERSION,
                "expected_obs_dim": _server_instance._expected_obs_dim if _server_instance else AINEX_STATE_DIM,
                "bridge_compatible": _server_instance._bridge_compatible if _server_instance else False,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP logging."""
        pass


def run_server(
    checkpoint_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cpu",
) -> None:
    """Start the inference server.

    Args:
        checkpoint_path: Path to policy checkpoint
        host: Bind address
        port: Port number
        device: PyTorch device
    """
    global _server_instance
    _server_instance = PolicyServer(checkpoint_path, device)
    _server_instance.load()

    server = HTTPServer((host, port), InferenceHandler)
    logger.info(f"Serving policy on http://{host}:{port}/infer")
    logger.info(f"Health check: http://{host}:{port}/health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down")
        server.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve trained AiNex policy via OpenPI protocol")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to policy checkpoint (.pt)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    run_server(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
    )


if __name__ == "__main__":
    main()
