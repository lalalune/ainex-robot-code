"""Join exported planner traces with bridge executor traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_planner_rows(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            raw = line.strip()
            if raw == "":
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                continue
            trace_id = str(record.get("trace_id", ""))
            planner_step_id = str(record.get("planner_step_id", ""))
            if trace_id == "" or planner_step_id == "":
                continue
            rows[(trace_id, planner_step_id)] = record
    return rows


def build_dataset(
    planner_jsonl: Path,
    executor_trace_jsonl: Path,
    output_jsonl: Path,
) -> int:
    planner_rows = _load_planner_rows(planner_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with executor_trace_jsonl.open("r", encoding="utf-8") as infile, output_jsonl.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            raw = line.strip()
            if raw == "":
                continue

            record = json.loads(raw)
            if not isinstance(record, dict):
                continue
            if str(record.get("kind", "")) != "policy_tick":
                continue

            trace_id = str(record.get("trace_id", ""))
            planner_step_id = str(record.get("planner_step_id", ""))
            if trace_id == "" or planner_step_id == "":
                continue

            planner_row = planner_rows.get((trace_id, planner_step_id))
            if planner_row is None:
                continue

            action_value = record.get("action", {})
            if not isinstance(action_value, dict):
                continue

            joint_positions = action_value.get("joint_positions", {})
            if not isinstance(joint_positions, dict):
                continue

            linked_row = {
                "schema_version": "planner-executor-linked-v1",
                "trace_id": trace_id,
                "planner_step_id": planner_step_id,
                "planner": planner_row,
                "executor": {
                    "step": int(record.get("step", 0)),
                    "timestamp": str(record.get("timestamp", "")),
                    "canonical_action": str(record.get("canonical_action", "")),
                    "target_entity_id": str(record.get("target_entity_id", "")),
                    "target_label": str(record.get("target_label", "")),
                    "joint_positions": joint_positions,
                    "clamped": record.get("clamped", {}),
                    "response_ok": bool(record.get("response_ok", False)),
                },
            }
            outfile.write(json.dumps(linked_row) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a linked planner/executor dataset from planner and bridge traces"
    )
    parser.add_argument("--planner-jsonl", type=str, required=True)
    parser.add_argument("--executor-trace-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    args = parser.parse_args()

    count = build_dataset(
        planner_jsonl=Path(args.planner_jsonl),
        executor_trace_jsonl=Path(args.executor_trace_jsonl),
        output_jsonl=Path(args.output_jsonl),
    )
    print(f"Wrote {count} linked dataset rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
