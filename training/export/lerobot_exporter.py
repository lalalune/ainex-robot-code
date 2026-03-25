"""LeRobot Exporter -- converts AiNex JSONL traces to LeRobotDataset v2.0 format.

LeRobotDataset stores data as Parquet files with a specific column schema:
``episode_index``, ``frame_index``, ``timestamp``, ``observation.state``,
``action``, plus optional image frame references and episode metadata.

Usage
-----
    python -m training.export.lerobot_exporter \
        --input  traces/ \
        --output lerobot_dataset/
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
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

# LeRobotDataset v2.0 metadata version
LEROBOT_CODEBASE_VERSION = "v2.0"
LEROBOT_DATA_FORMAT = "parquet"
CHUNK_SIZE = 1000  # episodes per chunk directory

# ---------------------------------------------------------------------------
# Observation state dimension labels (163 total)
# ---------------------------------------------------------------------------
PROPRIO_NAMES = [
    "walk_x",
    "walk_y",
    "walk_yaw",
    "walk_height",
    "walk_speed",
    "head_pan",
    "head_tilt",
    "imu_roll",
    "imu_pitch",
    "is_walking",
    "battery_mv",
]

_ENTITY_SLOT_FIELDS = [
    "type_0", "type_1", "type_2", "type_3", "type_4", "type_5",
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "size_w", "size_h", "size_d",
    "confidence", "recency",
    "bearing_sin", "bearing_cos",
]
NUM_ENTITY_SLOTS = 8

OBSERVATION_STATE_NAMES: list[str] = list(PROPRIO_NAMES)
for _slot in range(NUM_ENTITY_SLOTS):
    for _field in _ENTITY_SLOT_FIELDS:
        OBSERVATION_STATE_NAMES.append(f"entity_{_slot}_{_field}")

assert len(OBSERVATION_STATE_NAMES) == AINEX_STATE_DIM, (
    f"Expected {AINEX_STATE_DIM} observation.state names, got {len(OBSERVATION_STATE_NAMES)}"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """One control tick, corresponding to one row in the LeRobot Parquet."""

    episode_index: int
    frame_index: int
    timestamp: float  # seconds since episode start
    observation_state: np.ndarray  # (AINEX_STATE_DIM,)
    action: np.ndarray  # (AINEX_ACTION_DIM,)
    language_instruction: str = ""
    # Optional image path (relative to dataset root)
    image_path: str | None = None
    # Extra metadata
    done: bool = False
    reward: float = 0.0
    inference_ms: float = 0.0


@dataclass
class EpisodeMeta:
    """Metadata for one episode in the episode index."""

    episode_index: int
    episode_id: str
    task_description: str
    num_frames: int
    success: bool
    total_reward: float
    duration_s: float
    schema_version: str = AINEX_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Parsing helpers (shared logic with rlds_exporter, kept self-contained)
# ---------------------------------------------------------------------------

def _parse_state(obs: dict[str, Any]) -> np.ndarray:
    if "state" in obs and isinstance(obs["state"], list):
        vec = np.array(obs["state"], dtype=np.float32)
        if vec.shape[0] < AINEX_STATE_DIM:
            vec = np.pad(vec, (0, AINEX_STATE_DIM - vec.shape[0]))
        return vec[:AINEX_STATE_DIM]
    return np.zeros(AINEX_STATE_DIM, dtype=np.float32)


def _parse_action(row: dict[str, Any]) -> np.ndarray:
    src: dict[str, Any] | None = row.get("clamped") or row.get("action")
    if src is None:
        src = row.get("action_summary", {})
    vec = np.zeros(AINEX_ACTION_DIM, dtype=np.float32)
    for i, key in enumerate(ACTION_KEYS):
        if key in src:
            vec[i] = float(src[key])
    return vec


def _parse_timestamp_s(row: dict[str, Any], base_ts: float | None) -> float:
    """Return seconds elapsed since episode start."""
    raw = row.get("timestamp", 0.0)
    if isinstance(raw, str):
        # ISO-8601 timestamp
        try:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(raw)
            epoch = dt.timestamp()
        except Exception:
            epoch = 0.0
    else:
        epoch = float(raw)

    if base_ts is None:
        return 0.0
    return epoch - base_ts


def _epoch_from_timestamp(raw: Any) -> float:
    if isinstance(raw, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            return 0.0
    return float(raw) if raw else 0.0


def _extract_image(obs: dict[str, Any], dest_dir: Path, episode_idx: int, frame_idx: int) -> str | None:
    """If the observation carries an image, save it and return a relative path."""
    img_b64 = obs.get("image") or obs.get("image_base64")
    if img_b64 and isinstance(img_b64, str):
        try:
            raw_bytes = base64.b64decode(img_b64)
        except Exception:
            return None
        rel = f"images/episode_{episode_idx:06d}/frame_{frame_idx:06d}.png"
        abs_path = dest_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_path, "wb") as fh:
            fh.write(raw_bytes)
        return rel

    src_path = obs.get("image_path")
    if src_path and os.path.isfile(src_path):
        rel = f"images/episode_{episode_idx:06d}/frame_{frame_idx:06d}{Path(src_path).suffix}"
        abs_path = dest_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, abs_path)
        return rel

    return None


# ---------------------------------------------------------------------------
# Core exporter
# ---------------------------------------------------------------------------

class LeRobotExporter:
    """Reads AiNex JSONL / JSON traces and writes a LeRobotDataset v2.0."""

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    # -- file discovery ---------------------------------------------------

    def _iter_jsonl_files(self) -> Iterator[Path]:
        if self.input_path.is_file() and self.input_path.suffix == ".jsonl":
            yield self.input_path
            return
        if self.input_path.is_dir():
            for p in sorted(self.input_path.rglob("*.jsonl")):
                yield p

    def _iter_json_files(self) -> Iterator[Path]:
        if self.input_path.is_file() and self.input_path.suffix == ".json":
            yield self.input_path
            return
        if self.input_path.is_dir():
            for p in sorted(self.input_path.rglob("*.json")):
                yield p

    # -- loading ----------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d -- skipping: %s", path, lineno, exc)
        return rows

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "trace" in data:
            trace = data["trace"]
            if isinstance(trace, list):
                return trace
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _group_by_episode(
        rows: list[dict[str, Any]], fallback_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        episodes: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tid = row.get("trace_id", fallback_id)
            episodes.setdefault(tid, []).append(row)
        for tid in episodes:
            episodes[tid].sort(key=lambda r: r.get("step", 0))
        return episodes

    # -- conversion -------------------------------------------------------

    def _episode_to_frames(
        self,
        episode_index: int,
        rows: list[dict[str, Any]],
    ) -> tuple[list[Frame], EpisodeMeta]:
        """Convert raw rows to a list of Frame objects + episode metadata."""
        first_obs = rows[0].get("observation", {})
        task = first_obs.get("prompt", "")
        if not task:
            task = rows[0].get("observation_summary", {}).get("prompt", "unknown task")

        # Determine base timestamp for relative timing
        base_ts = _epoch_from_timestamp(rows[0].get("timestamp", 0.0))
        episode_id = rows[0].get("trace_id", f"episode_{episode_index}")

        frames: list[Frame] = []
        total_reward = 0.0
        for frame_idx, row in enumerate(rows):
            obs = row.get("observation", {})
            state = _parse_state(obs)
            action = _parse_action(row)
            ts_s = _parse_timestamp_s(row, base_ts)
            reward = float(row.get("reward", 0.0))
            total_reward += reward

            done = bool(row.get("done", False))
            if frame_idx == len(rows) - 1:
                done = True  # last frame is always terminal

            img_path = _extract_image(obs, self.output_path, episode_index, frame_idx)

            frame = Frame(
                episode_index=episode_index,
                frame_index=frame_idx,
                timestamp=round(ts_s, 6),
                observation_state=state,
                action=action,
                language_instruction=obs.get("prompt", task),
                image_path=img_path,
                done=done,
                reward=reward,
                inference_ms=float(row.get("inference_ms", 0.0)),
            )
            frames.append(frame)

        last_ts = _epoch_from_timestamp(rows[-1].get("timestamp", 0.0))
        duration_s = max(0.0, last_ts - base_ts)

        success = False
        if rows[-1].get("success") is not None:
            success = bool(rows[-1]["success"])
        elif total_reward > 0:
            success = True

        meta = EpisodeMeta(
            episode_index=episode_index,
            episode_id=episode_id,
            task_description=task,
            num_frames=len(frames),
            success=success,
            total_reward=total_reward,
            duration_s=round(duration_s, 3),
        )
        return frames, meta

    def read_all(self) -> tuple[list[Frame], list[EpisodeMeta]]:
        """Parse every input file and return frames + episode metadata."""
        all_frames: list[Frame] = []
        all_meta: list[EpisodeMeta] = []
        ep_counter = 0

        # JSONL files
        for path in self._iter_jsonl_files():
            rows = self._load_jsonl(path)
            if not rows:
                continue
            grouped = self._group_by_episode(rows, path.stem)
            for _eid, ep_rows in grouped.items():
                frames, meta = self._episode_to_frames(ep_counter, ep_rows)
                all_frames.extend(frames)
                all_meta.append(meta)
                ep_counter += 1

        # JSON files
        for path in self._iter_json_files():
            rows = self._load_json(path)
            if not rows:
                continue
            grouped = self._group_by_episode(rows, path.stem)
            for _eid, ep_rows in grouped.items():
                frames, meta = self._episode_to_frames(ep_counter, ep_rows)
                all_frames.extend(frames)
                all_meta.append(meta)
                ep_counter += 1

        logger.info(
            "Loaded %d episodes (%d frames) from %s",
            len(all_meta),
            len(all_frames),
            self.input_path,
        )
        return all_frames, all_meta

    # -- writing ----------------------------------------------------------

    @staticmethod
    def _frame_to_record(f: Frame) -> dict[str, Any]:
        """Convert a single Frame to a flat dict suitable for Parquet."""
        rec: dict[str, Any] = {
            "episode_index": f.episode_index,
            "frame_index": f.frame_index,
            "timestamp": f.timestamp,
            "next.done": f.done,
            "next.reward": f.reward,
            "language_instruction": f.language_instruction,
            "inference_ms": f.inference_ms,
        }
        # observation.state as individual float columns
        for i in range(AINEX_STATE_DIM):
            rec[f"observation.state.{i}"] = float(f.observation_state[i])
        # action as individual float columns
        for i, key in enumerate(ACTION_KEYS):
            rec[f"action.{key}"] = float(f.action[i])
        # image reference
        if f.image_path:
            rec["observation.image_path"] = f.image_path
        return rec

    def _write_parquet(self, frames: list[Frame]) -> int:
        """Write per-episode parquet files under data/chunk-NNN/.

        Returns the number of chunk directories created.

        LeRobot v2.0 layout:
            data/chunk-000/episode_000000.parquet
            data/chunk-000/episode_000001.parquet
            ...
            data/chunk-001/episode_001000.parquet
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not installed; falling back to JSONL.")
            self._write_jsonl_fallback(frames)
            return 1

        if not frames:
            logger.warning("No frames to write.")
            return 0

        # Group frames by episode_index
        episodes: dict[int, list[Frame]] = {}
        for f in frames:
            episodes.setdefault(f.episode_index, []).append(f)

        chunks_written: set[int] = set()

        for ep_idx in sorted(episodes.keys()):
            chunk_idx = ep_idx // CHUNK_SIZE
            chunks_written.add(chunk_idx)

            chunk_dir = self.output_path / "data" / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            records = [self._frame_to_record(f) for f in episodes[ep_idx]]
            table = pa.Table.from_pylist(records)
            pq_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
            pq.write_table(table, str(pq_path), compression="snappy")
            logger.debug("Wrote %d frames to %s", len(records), pq_path)

        logger.info(
            "Wrote %d episodes (%d frames) across %d chunk(s)",
            len(episodes),
            len(frames),
            len(chunks_written),
        )
        return len(chunks_written)

    def _write_jsonl_fallback(self, frames: list[Frame]) -> Path:
        """JSONL fallback when pyarrow is unavailable."""
        data_dir = self.output_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / "train.jsonl"

        with open(out_path, "w", encoding="utf-8") as fh:
            for f in frames:
                rec: dict[str, Any] = {
                    "episode_index": f.episode_index,
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "observation.state": f.observation_state.tolist(),
                    "action": f.action.tolist(),
                    "next.done": f.done,
                    "next.reward": f.reward,
                    "language_instruction": f.language_instruction,
                    "inference_ms": f.inference_ms,
                }
                if f.image_path:
                    rec["observation.image_path"] = f.image_path
                fh.write(json.dumps(rec) + "\n")

        logger.info("Wrote %d frames (JSONL fallback) to %s", len(frames), out_path)
        return out_path

    def _write_episode_index(self, episodes: list[EpisodeMeta]) -> Path:
        """Write meta/episodes.jsonl."""
        meta_dir = self.output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        index_path = meta_dir / "episodes.jsonl"

        with open(index_path, "w", encoding="utf-8") as fh:
            for ep in episodes:
                rec = {
                    "episode_index": ep.episode_index,
                    "episode_id": ep.episode_id,
                    "task": ep.task_description,
                    "length": ep.num_frames,
                    "success": ep.success,
                    "total_reward": ep.total_reward,
                    "duration_s": ep.duration_s,
                    "schema_version": ep.schema_version,
                }
                fh.write(json.dumps(rec) + "\n")

        logger.info("Wrote episode index (%d episodes) to %s", len(episodes), index_path)
        return index_path

    def _write_info(
        self,
        episodes: list[EpisodeMeta],
        total_frames: int,
        total_chunks: int,
    ) -> Path:
        """Write meta/info.json with LeRobotDataset v2.0 metadata."""
        meta_dir = self.output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        info_path = meta_dir / "info.json"

        unique_tasks: list[str] = list({ep.task_description for ep in episodes})

        # Count video directories (one per episode that has images)
        video_dirs = self.output_path / "images"
        total_videos = 0
        if video_dirs.is_dir():
            total_videos = sum(1 for d in video_dirs.iterdir() if d.is_dir())

        info = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": "ainex_humanoid",
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": len(unique_tasks),
            "total_chunks": total_chunks,
            "total_videos": total_videos,
            "fps": 20,  # approximate from typical ~50ms ticks
            "data_path": f"data/chunk-{{chunk:03d}}/episode_{{episode:06d}}.parquet",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [AINEX_STATE_DIM],
                    "names": OBSERVATION_STATE_NAMES,
                },
                "action": {
                    "dtype": "float32",
                    "shape": [AINEX_ACTION_DIM],
                    "names": list(ACTION_KEYS),
                },
                "language_instruction": {
                    "dtype": "string",
                    "shape": [1],
                },
            },
            "tasks": unique_tasks,
            "chunks_size": CHUNK_SIZE,
            "schema_version": AINEX_SCHEMA_VERSION,
        }

        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)

        logger.info("Wrote dataset info to %s", info_path)
        return info_path

    def _write_tasks(self, episodes: list[EpisodeMeta]) -> Path:
        """Write meta/tasks.jsonl mapping task_index to description."""
        meta_dir = self.output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = meta_dir / "tasks.jsonl"

        seen: dict[str, int] = {}
        with open(tasks_path, "w", encoding="utf-8") as fh:
            for ep in episodes:
                desc = ep.task_description
                if desc not in seen:
                    idx = len(seen)
                    seen[desc] = idx
                    fh.write(json.dumps({"task_index": idx, "task": desc}) + "\n")

        logger.info("Wrote %d unique tasks to %s", len(seen), tasks_path)
        return tasks_path

    def export(self) -> Path:
        """Main entry point."""
        frames, episodes = self.read_all()
        if not frames:
            logger.warning("No data found; nothing to export.")
            self.output_path.mkdir(parents=True, exist_ok=True)
            return self.output_path

        self.output_path.mkdir(parents=True, exist_ok=True)
        total_chunks = self._write_parquet(frames)
        self._write_episode_index(episodes)
        self._write_info(episodes, total_frames=len(frames), total_chunks=total_chunks)
        self._write_tasks(episodes)

        logger.info("LeRobot dataset written to %s", self.output_path)
        return self.output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export AiNex JSONL traces to LeRobotDataset v2.0 format.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL file or directory of trace files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the LeRobot dataset.",
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

    exporter = LeRobotExporter(
        input_path=args.input,
        output_path=args.output,
    )
    out = exporter.export()
    print(f"Dataset written to {out}")


if __name__ == "__main__":
    main()
