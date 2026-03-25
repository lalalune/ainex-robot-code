"""Dense per-tick control logger for policy training data.

Writes one JSONL line per policy tick with the full observation vector,
action vector, timestamp, and terminal flag.  Linked to the planner
trajectory via trace_id so planner-level and control-level data can be
joined for training.

Usage:
    logger = ControlLogger(output_dir, trace_id="abc")
    logger.log_tick(obs_163, action_7, done=False)
    ...
    logger.close()
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class ControlLogger:
    """Append-only JSONL logger for dense control trajectories."""

    def __init__(
        self,
        output_dir: str | Path,
        trace_id: str = "",
        planner_step_id: str = "",
        canonical_action: str = "",
        target_entity_id: str = "",
        target_label: str = "",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        fname = f"control_{trace_id or 'notrace'}_{ts}.jsonl"
        self._path = self._output_dir / fname
        self._file = open(self._path, "a", encoding="utf-8")
        self._tick = 0
        self._trace_id = trace_id
        self._planner_step_id = planner_step_id

        # Write header line
        self._write({
            "type": "header",
            "trace_id": trace_id,
            "planner_step_id": planner_step_id,
            "canonical_action": canonical_action,
            "target_entity_id": target_entity_id,
            "target_label": target_label,
            "start_time_ms": ts,
        })

    def log_tick(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        info: dict | None = None,
    ) -> None:
        """Log a single policy tick."""
        self._write({
            "type": "tick",
            "tick": self._tick,
            "time_ms": int(time.time() * 1000),
            "observation": observation.tolist() if isinstance(observation, np.ndarray) else list(observation),
            "action": action.tolist() if isinstance(action, np.ndarray) else list(action),
            "reward": reward,
            "done": done,
            **({"info": info} if info else {}),
        })
        self._tick += 1

    def log_terminal(self, status: str, metrics: dict | None = None) -> None:
        """Log terminal event."""
        self._write({
            "type": "terminal",
            "tick": self._tick,
            "time_ms": int(time.time() * 1000),
            "status": status,
            **({"metrics": metrics} if metrics else {}),
        })

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def tick_count(self) -> int:
        return self._tick

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def __enter__(self) -> "ControlLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
