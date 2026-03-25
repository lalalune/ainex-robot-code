"""Build a planner-training dataset from exported Hyperscape SQL trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _extract_canonical_action(record: dict[str, object]) -> str:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        trace_action = metadata.get("selectedCanonicalAction")
        if isinstance(trace_action, str) and trace_action:
            return trace_action

    steps = record.get("steps", [])
    if not isinstance(steps, list):
        return ""

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        action_value = step.get("action", {})
        if not isinstance(action_value, dict):
            continue
        action_name = action_value.get("actionName")
        if isinstance(action_name, str) and action_name:
            return action_name
    return ""


def _extract_last_llm_prompt(record: dict[str, object]) -> str:
    steps = record.get("steps", [])
    if not isinstance(steps, list):
        return ""

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        llm_calls = step.get("llmCalls", [])
        if not isinstance(llm_calls, list) or not llm_calls:
            continue
        last_call = llm_calls[-1]
        if isinstance(last_call, dict):
            user_prompt = last_call.get("userPrompt")
            if isinstance(user_prompt, str):
                return user_prompt
    return ""


def _build_row_from_planner_context(record: dict, metadata: dict) -> dict | None:
    """Build a dataset row from the original autonomous_llm_selection format."""
    planner_context = metadata.get("canonicalPlannerContext")
    if not isinstance(planner_context, dict):
        return None

    return {
        "schema_version": "hyperscape-planner-dataset-v1",
        "trajectory_id": str(record.get("trajectory_id", "")),
        "agent_id": str(record.get("agent_id", "")),
        "trace_id": str(record.get("trace_id", "")),
        "planner_step_id": str(record.get("planner_step_id", "")),
        "task_prompt": _extract_last_llm_prompt(record),
        "canonical_planner_context": planner_context,
        "selected_canonical_action": _extract_canonical_action(record),
        "current_goal_type": metadata.get("currentGoalType", ""),
        "current_goal_description": metadata.get("currentGoalDescription", ""),
    }


def _build_row_from_behavior_tick(record: dict, metadata: dict) -> dict | None:
    """Build a dataset row from the embedded-behavior-tick format."""
    game_state = metadata.get("gameState")
    if not isinstance(game_state, dict):
        return None

    action_type = metadata.get("actionType", "")
    action_data = metadata.get("action", {})

    # Build a synthetic planner context from the game state
    nearby_entities = game_state.get("nearbyEntities", [])
    entities = []
    for e in nearby_entities:
        if isinstance(e, dict):
            entities.append({
                "entityId": e.get("id", ""),
                "label": e.get("name", ""),
                "type": e.get("type", "unknown"),
                "position": [0, 0, 0],
                "distance": e.get("distance", 0),
                "health": e.get("health"),
            })

    return {
        "schema_version": "hyperscape-planner-dataset-v1",
        "trajectory_id": str(record.get("trajectory_id", "")),
        "agent_id": str(record.get("agent_id", "")),
        "trace_id": str(record.get("trace_id", "")),
        "planner_step_id": str(record.get("planner_step_id", "")),
        "task_prompt": "",
        "canonical_planner_context": {
            "source": "embedded-behavior-tick",
            "player": {
                "position": game_state.get("position"),
                "health": game_state.get("health"),
                "maxHealth": game_state.get("maxHealth"),
                "inCombat": game_state.get("inCombat", False),
            },
            "entities": entities,
            "entityCount": game_state.get("nearbyEntityCount", len(entities)),
        },
        "selected_canonical_action": str(action_type),
        "action_details": action_data if isinstance(action_data, dict) else {},
        "current_goal_type": metadata.get("goalType", ""),
        "current_goal_description": metadata.get("goalDescription", ""),
        "agent_name": metadata.get("agentName", ""),
    }


def build_dataset(input_jsonl: Path, output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_jsonl.open("r", encoding="utf-8") as infile, output_jsonl.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            raw = line.strip()
            if raw == "":
                continue

            record = json.loads(raw)
            if not isinstance(record, dict):
                continue

            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            source = record.get("source", "")

            # Try behavior tick format first (more common in practice)
            if source == "embedded-behavior-tick":
                dataset_row = _build_row_from_behavior_tick(record, metadata)
            else:
                dataset_row = _build_row_from_planner_context(record, metadata)

            if dataset_row is None:
                continue

            outfile.write(json.dumps(dataset_row) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a planner-training dataset from exported planner trajectories"
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    args = parser.parse_args()

    count = build_dataset(Path(args.input_jsonl), Path(args.output_jsonl))
    print(f"Wrote {count} planner dataset rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
