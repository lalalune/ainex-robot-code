"""RLDS Exporter -- converts AiNex JSONL trajectory traces to RLDS format.

RLDS (Reinforcement Learning Datasets) is the canonical format used by
OpenVLA, RT-X, and the broader robotics-learning ecosystem.  Each episode
is a dict with a ``steps`` sequence of (observation, action, reward,
is_terminal, ...) tuples stored as either TFRecord files or as a HuggingFace
dataset.

Usage
-----
    python -m training.export.rlds_exporter \
        --input  traces/ \
        --output rlds_dataset/ \
        --format hf
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrored from training.schema.canonical to stay self-contained)
# ---------------------------------------------------------------------------
AINEX_STATE_DIM = 163
AINEX_ACTION_DIM = 7
AINEX_SCHEMA_VERSION = "ainex-canonical-v1"
ACTION_KEYS = (
    "walk_x",
    "walk_y",
    "walk_yaw",
    "walk_height",
    "walk_speed",
    "head_pan",
    "head_tilt",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RLDSStep:
    """One transition in an RLDS episode."""

    observation_state: np.ndarray  # (AINEX_STATE_DIM,)
    observation_language: str
    observation_image: bytes | None  # raw PNG/JPEG bytes, or None
    action: np.ndarray  # (AINEX_ACTION_DIM,)
    reward: float
    is_terminal: bool
    is_first: bool
    is_last: bool
    # extra metadata carried per-step
    timestamp: str = ""
    inference_ms: float = 0.0
    tick_ms: float = 0.0
    confidence: float = 0.0


@dataclass
class RLDSEpisode:
    """One full episode (trajectory)."""

    episode_id: str
    task_description: str
    success: bool
    steps: list[RLDSStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_state(obs: dict[str, Any]) -> np.ndarray:
    """Extract or construct the 163-dim state vector from an observation dict."""
    if "state" in obs and isinstance(obs["state"], list):
        vec = np.array(obs["state"], dtype=np.float32)
        if vec.shape[0] < AINEX_STATE_DIM:
            vec = np.pad(vec, (0, AINEX_STATE_DIM - vec.shape[0]))
        return vec[:AINEX_STATE_DIM]
    # Fallback: zeros
    return np.zeros(AINEX_STATE_DIM, dtype=np.float32)


def _parse_action(row: dict[str, Any]) -> np.ndarray:
    """Extract the 7-dim canonical action vector."""
    # Prefer "clamped" (the actually-executed action), fall back to "action"
    src: dict[str, Any] | None = row.get("clamped") or row.get("action")
    if src is None:
        # Try the action_summary sub-dict (control-level traces)
        src = row.get("action_summary", {})

    vec = np.zeros(AINEX_ACTION_DIM, dtype=np.float32)
    for i, key in enumerate(ACTION_KEYS):
        if key in src:
            vec[i] = float(src[key])
    return vec


def _parse_image(obs: dict[str, Any]) -> bytes | None:
    """Return raw image bytes if the observation embeds one."""
    img_b64 = obs.get("image") or obs.get("image_base64")
    if img_b64 and isinstance(img_b64, str):
        try:
            return base64.b64decode(img_b64)
        except Exception:
            logger.warning("Failed to decode base64 image; skipping.")
    img_path = obs.get("image_path")
    if img_path and os.path.isfile(img_path):
        try:
            with open(img_path, "rb") as fh:
                return fh.read()
        except OSError:
            logger.warning("Failed to read image at %s", img_path)
    return None


def _parse_reward(row: dict[str, Any]) -> float:
    """Extract a scalar reward.  Many control-level traces have no explicit
    reward, so we default to 0.0."""
    if "reward" in row:
        return float(row["reward"])
    return 0.0


# ---------------------------------------------------------------------------
# Core exporter
# ---------------------------------------------------------------------------

class RLDSExporter:
    """Reads AiNex JSONL trace files and converts them to RLDS episodes.

    Supports two trace flavours:
    * **control-level** -- one JSONL line per tick with observation, clamped
      action, inference_ms, etc.  All lines sharing the same ``trace_id``
      form one episode.
    * **planner-level** -- a JSON file whose ``trace`` key is a list of steps,
      each with an ``action`` dict.
    """

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        output_format: Literal["hf", "tfrecord"] = "hf",
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_format = output_format

    # -- reading ----------------------------------------------------------

    def _iter_jsonl_files(self) -> Iterator[Path]:
        """Yield all JSONL files under *input_path*."""
        if self.input_path.is_file():
            yield self.input_path
            return
        for p in sorted(self.input_path.rglob("*.jsonl")):
            yield p

    def _iter_json_files(self) -> Iterator[Path]:
        """Yield all plain JSON files (planner traces) under *input_path*."""
        if self.input_path.is_file() and self.input_path.suffix == ".json":
            yield self.input_path
            return
        if self.input_path.is_dir():
            for p in sorted(self.input_path.rglob("*.json")):
                yield p

    @staticmethod
    def _rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d -- skipping malformed JSON: %s", path, lineno, exc)
        return rows

    @staticmethod
    def _rows_from_json(path: Path) -> list[dict[str, Any]]:
        """Load a planner-level JSON trace (``{result, trace}``)."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "trace" in data:
            trace = data["trace"]
            if isinstance(trace, list):
                return trace
        if isinstance(data, list):
            return data
        return []

    # -- episode assembly -------------------------------------------------

    def _group_by_episode(
        self,
        rows: list[dict[str, Any]],
        source_file: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group rows into episodes keyed by trace_id."""
        episodes: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tid = row.get("trace_id", source_file)
            episodes.setdefault(tid, []).append(row)
        # Sort each episode by step number
        for tid in episodes:
            episodes[tid].sort(key=lambda r: r.get("step", 0))
        return episodes

    def _build_episode(
        self,
        episode_id: str,
        rows: list[dict[str, Any]],
    ) -> RLDSEpisode:
        """Convert a list of step-rows into a single ``RLDSEpisode``."""
        # Infer task description from the prompt in the first observation
        first_obs = rows[0].get("observation", {})
        task = first_obs.get("prompt", "")
        if not task:
            summary = rows[0].get("observation_summary", {})
            task = summary.get("prompt", "unknown task")

        steps: list[RLDSStep] = []
        n = len(rows)
        for idx, row in enumerate(rows):
            obs = row.get("observation", {})
            state = _parse_state(obs)
            language = obs.get("prompt", task)
            image = _parse_image(obs)
            action = _parse_action(row)
            reward = _parse_reward(row)
            is_first = idx == 0
            is_last = idx == n - 1
            is_terminal = is_last  # may be overridden by explicit field

            if "is_terminal" in row:
                is_terminal = bool(row["is_terminal"])
            if "done" in row:
                is_terminal = bool(row["done"])

            action_summary = row.get("action_summary", {})
            step = RLDSStep(
                observation_state=state,
                observation_language=language,
                observation_image=image,
                action=action,
                reward=reward,
                is_terminal=is_terminal,
                is_first=is_first,
                is_last=is_last,
                timestamp=row.get("timestamp", ""),
                inference_ms=float(row.get("inference_ms", 0.0)),
                tick_ms=float(row.get("tick_ms", 0.0)),
                confidence=float(action_summary.get("confidence", 0.0)),
            )
            steps.append(step)

        # Determine success: heuristic -- last reward > 0 or explicit flag
        success = False
        if rows[-1].get("success") is not None:
            success = bool(rows[-1]["success"])
        elif steps and steps[-1].reward > 0:
            success = True

        metadata: dict[str, Any] = {
            "schema_version": rows[0].get("schema_version", ""),
            "num_steps": n,
        }

        return RLDSEpisode(
            episode_id=episode_id,
            task_description=task,
            success=success,
            steps=steps,
            metadata=metadata,
        )

    def read_episodes(self) -> list[RLDSEpisode]:
        """Parse all input files and return a list of RLDS episodes."""
        episodes: list[RLDSEpisode] = []

        # Control-level JSONL traces
        for path in self._iter_jsonl_files():
            rows = self._rows_from_jsonl(path)
            if not rows:
                continue
            grouped = self._group_by_episode(rows, path.stem)
            for eid, ep_rows in grouped.items():
                episodes.append(self._build_episode(eid, ep_rows))

        # Planner-level JSON traces
        for path in self._iter_json_files():
            rows = self._rows_from_json(path)
            if not rows:
                continue
            grouped = self._group_by_episode(rows, path.stem)
            for eid, ep_rows in grouped.items():
                episodes.append(self._build_episode(eid, ep_rows))

        logger.info("Loaded %d episodes from %s", len(episodes), self.input_path)
        return episodes

    # -- export -----------------------------------------------------------

    def _episode_to_dict(self, ep: RLDSEpisode) -> dict[str, Any]:
        """Serialise an episode into a plain dict matching RLDS schema."""
        steps_list: list[dict[str, Any]] = []
        for s in ep.steps:
            step_dict: dict[str, Any] = {
                "observation": {
                    "state": s.observation_state.tolist(),
                    "language_instruction": s.observation_language,
                    "has_image": s.observation_image is not None,
                },
                "action": s.action.tolist(),
                "reward": s.reward,
                "is_terminal": s.is_terminal,
                "is_first": s.is_first,
                "is_last": s.is_last,
            }
            if s.observation_image is not None:
                step_dict["observation"]["image"] = base64.b64encode(
                    s.observation_image
                ).decode("ascii")
            steps_list.append(step_dict)

        return {
            "episode_id": ep.episode_id,
            "task_description": ep.task_description,
            "success": ep.success,
            "steps": steps_list,
            "metadata": ep.metadata,
        }

    def export_hf(self, episodes: Sequence[RLDSEpisode]) -> Path:
        """Export to HuggingFace datasets format (Arrow / parquet)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning(
                "pyarrow not installed; falling back to JSONL-based HF export."
            )
            return self._export_hf_jsonl(episodes)

        out_dir = self.output_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Flatten all steps across episodes into a single table -- the
        # standard approach for RLDS-on-HF (each row = one step, with an
        # episode_index column for grouping).
        records: list[dict[str, Any]] = []
        for ep_idx, ep in enumerate(episodes):
            for step_idx, s in enumerate(ep.steps):
                rec: dict[str, Any] = {
                    "episode_index": ep_idx,
                    "step_index": step_idx,
                    "episode_id": ep.episode_id,
                    "task_description": ep.task_description,
                    "success": ep.success,
                    "language_instruction": s.observation_language,
                    "reward": s.reward,
                    "is_terminal": s.is_terminal,
                    "is_first": s.is_first,
                    "is_last": s.is_last,
                    "timestamp": s.timestamp,
                    "inference_ms": s.inference_ms,
                    "tick_ms": s.tick_ms,
                    "confidence": s.confidence,
                }
                # State and action as individual columns for efficient access
                for i in range(AINEX_STATE_DIM):
                    rec[f"observation.state.{i}"] = float(s.observation_state[i])
                for i, key in enumerate(ACTION_KEYS):
                    rec[f"action.{key}"] = float(s.action[i])
                records.append(rec)

        if not records:
            logger.warning("No steps to export.")
            return out_dir

        table = pa.Table.from_pylist(records)
        parquet_path = out_dir / "data.parquet"
        pq.write_table(table, str(parquet_path))

        # Write dataset_info.json for HF datasets compatibility
        info = {
            "description": "AiNex RLDS dataset",
            "schema_version": AINEX_SCHEMA_VERSION,
            "state_dim": AINEX_STATE_DIM,
            "action_dim": AINEX_ACTION_DIM,
            "action_keys": list(ACTION_KEYS),
            "num_episodes": len(episodes),
            "num_steps": len(records),
        }
        with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

        logger.info(
            "Exported %d episodes (%d steps) to HF parquet at %s",
            len(episodes),
            len(records),
            parquet_path,
        )
        return out_dir

    def _export_hf_jsonl(self, episodes: Sequence[RLDSEpisode]) -> Path:
        """Fallback JSONL export when pyarrow is unavailable."""
        out_dir = self.output_path
        out_dir.mkdir(parents=True, exist_ok=True)

        data_path = out_dir / "data.jsonl"
        count = 0
        with open(data_path, "w", encoding="utf-8") as fh:
            for ep_idx, ep in enumerate(episodes):
                for step_idx, s in enumerate(ep.steps):
                    rec: dict[str, Any] = {
                        "episode_index": ep_idx,
                        "step_index": step_idx,
                        "episode_id": ep.episode_id,
                        "task_description": ep.task_description,
                        "success": ep.success,
                        "language_instruction": s.observation_language,
                        "observation_state": s.observation_state.tolist(),
                        "action": s.action.tolist(),
                        "reward": s.reward,
                        "is_terminal": s.is_terminal,
                        "is_first": s.is_first,
                        "is_last": s.is_last,
                        "timestamp": s.timestamp,
                        "inference_ms": s.inference_ms,
                        "tick_ms": s.tick_ms,
                        "confidence": s.confidence,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    count += 1

        info = {
            "description": "AiNex RLDS dataset (JSONL fallback)",
            "schema_version": AINEX_SCHEMA_VERSION,
            "state_dim": AINEX_STATE_DIM,
            "action_dim": AINEX_ACTION_DIM,
            "action_keys": list(ACTION_KEYS),
            "num_episodes": len(episodes),
            "num_steps": count,
        }
        with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

        logger.info(
            "Exported %d episodes (%d steps) as JSONL to %s",
            len(episodes),
            count,
            data_path,
        )
        return out_dir

    def export_tfrecord(self, episodes: Sequence[RLDSEpisode]) -> Path:
        """Export to TFRecord format."""
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError(
                "TensorFlow is required for TFRecord export.  "
                "Install it with: pip install tensorflow"
            )

        out_dir = self.output_path
        out_dir.mkdir(parents=True, exist_ok=True)

        def _bytes_feature(value: bytes) -> tf.train.Feature:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_list_feature(values: Sequence[float]) -> tf.train.Feature:
            return tf.train.Feature(float_list=tf.train.FloatList(value=values))

        def _int64_feature(value: int) -> tf.train.Feature:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        total_steps = 0
        for ep_idx, ep in enumerate(episodes):
            tfrecord_path = out_dir / f"episode_{ep_idx:06d}.tfrecord"
            with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
                for s in ep.steps:
                    feature: dict[str, tf.train.Feature] = {
                        "observation/state": _float_list_feature(
                            s.observation_state.tolist()
                        ),
                        "observation/language_instruction": _bytes_feature(
                            s.observation_language.encode("utf-8")
                        ),
                        "action": _float_list_feature(s.action.tolist()),
                        "reward": _float_list_feature([s.reward]),
                        "is_terminal": _int64_feature(int(s.is_terminal)),
                        "is_first": _int64_feature(int(s.is_first)),
                        "is_last": _int64_feature(int(s.is_last)),
                        "episode_id": _bytes_feature(
                            ep.episode_id.encode("utf-8")
                        ),
                        "task_description": _bytes_feature(
                            ep.task_description.encode("utf-8")
                        ),
                    }
                    if s.observation_image is not None:
                        feature["observation/image"] = _bytes_feature(
                            s.observation_image
                        )
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example.SerializeToString())
                    total_steps += 1

        # metadata sidecar
        info = {
            "description": "AiNex RLDS dataset (TFRecord)",
            "schema_version": AINEX_SCHEMA_VERSION,
            "state_dim": AINEX_STATE_DIM,
            "action_dim": AINEX_ACTION_DIM,
            "action_keys": list(ACTION_KEYS),
            "num_episodes": len(episodes),
            "num_steps": total_steps,
        }
        with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

        logger.info(
            "Exported %d episodes (%d steps) as TFRecord to %s",
            len(episodes),
            total_steps,
            out_dir,
        )
        return out_dir

    def export(self) -> Path:
        """Main entry point: read episodes and export in the chosen format."""
        episodes = self.read_episodes()
        if not episodes:
            logger.warning("No episodes found; nothing to export.")
            self.output_path.mkdir(parents=True, exist_ok=True)
            return self.output_path

        if self.output_format == "hf":
            return self.export_hf(episodes)
        elif self.output_format == "tfrecord":
            return self.export_tfrecord(episodes)
        else:
            raise ValueError(f"Unsupported format: {self.output_format!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export AiNex JSONL traces to RLDS format.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL file or directory of trace files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the RLDS dataset.",
    )
    parser.add_argument(
        "--format",
        choices=["hf", "tfrecord"],
        default="hf",
        help="Output format (default: hf).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    exporter = RLDSExporter(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format,
    )
    out = exporter.export()
    print(f"Dataset written to {out}")


if __name__ == "__main__":
    main()
