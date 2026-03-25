"""Build a bridge-aligned supervised dataset from OpenPI loop traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.schema.canonical import (
    AINEX_ACTION_DIM,
    AINEX_SCHEMA_VERSION,
    AINEX_STATE_DIM,
    HEAD_PAN_RANGE,
    HEAD_TILT_RANGE,
    WALK_HEIGHT_MAX,
    WALK_HEIGHT_MIN,
    WALK_SPEED_MAX,
    WALK_SPEED_MIN,
    WALK_X_RANGE,
    WALK_Y_RANGE,
    WALK_YAW_RANGE,
    normalize_value,
)


def build_dataset(input_trace: Path, output_jsonl: Path) -> int:
    count = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with input_trace.open("r", encoding="utf-8") as infile, output_jsonl.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            raw = line.strip()
            if raw == "":
                continue
            record = json.loads(raw)
            observation = record.get("observation")
            action = record.get("clamped")
            if not isinstance(observation, dict) or not isinstance(action, dict):
                continue

            state = observation.get("state")
            if not isinstance(state, list) or len(state) != AINEX_STATE_DIM:
                continue

            action_vector = [
                normalize_value(float(action.get("walk_x", 0.0)), -WALK_X_RANGE, WALK_X_RANGE),
                normalize_value(float(action.get("walk_y", 0.0)), -WALK_Y_RANGE, WALK_Y_RANGE),
                normalize_value(float(action.get("walk_yaw", 0.0)), -WALK_YAW_RANGE, WALK_YAW_RANGE),
                normalize_value(float(action.get("walk_height", 0.036)), WALK_HEIGHT_MIN, WALK_HEIGHT_MAX),
                normalize_value(float(action.get("walk_speed", 2.0)), float(WALK_SPEED_MIN), float(WALK_SPEED_MAX)),
                normalize_value(float(action.get("head_pan", 0.0)), -HEAD_PAN_RANGE, HEAD_PAN_RANGE),
                normalize_value(float(action.get("head_tilt", 0.0)), -HEAD_TILT_RANGE, HEAD_TILT_RANGE),
            ]
            if len(action_vector) != AINEX_ACTION_DIM:
                continue

            dataset_record = {
                "schema_version": AINEX_SCHEMA_VERSION,
                "trace_id": record.get("trace_id", ""),
                "step": int(record.get("step", 0)),
                "prompt": str(observation.get("prompt", "")),
                "state": [float(value) for value in state],
                "action": action_vector,
            }
            outfile.write(json.dumps(dataset_record) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bridge-policy dataset from trace JSONL")
    parser.add_argument("--input-trace", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    args = parser.parse_args()

    count = build_dataset(Path(args.input_trace), Path(args.output_jsonl))
    print(f"Wrote {count} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
