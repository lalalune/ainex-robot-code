from __future__ import annotations

import json
from pathlib import Path

from training.datasets.build_linked_planner_executor_dataset import (
    build_dataset as build_linked_dataset,
)
from training.datasets.build_planner_dataset import (
    build_dataset as build_planner_dataset,
)


def test_build_planner_dataset(tmp_path: Path) -> None:
    planner_export = tmp_path / "planner_export.jsonl"
    planner_dataset = tmp_path / "planner_dataset.jsonl"

    planner_export.write_text(
        json.dumps(
            {
                "trajectory_id": "traj-1",
                "agent_id": "agent-1",
                "trace_id": "trace-1",
                "planner_step_id": "planner-step-1",
                "metadata": {
                    "canonicalPlannerContext": {
                        "schemaVersion": "embodied-planner-v1",
                        "entities": [],
                    },
                    "currentGoalType": "exploration",
                    "currentGoalDescription": "Find the red ball",
                },
                "steps": [
                    {
                        "llmCalls": [{"userPrompt": "Walk to the red ball"}],
                        "action": {"actionName": "NAVIGATE_TO_ENTITY"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = build_planner_dataset(planner_export, planner_dataset)
    assert count == 1

    row = json.loads(planner_dataset.read_text(encoding="utf-8").strip())
    assert row["trace_id"] == "trace-1"
    assert row["planner_step_id"] == "planner-step-1"
    assert row["selected_canonical_action"] == "NAVIGATE_TO_ENTITY"
    assert row["task_prompt"] == "Walk to the red ball"


def test_build_linked_planner_executor_dataset(tmp_path: Path) -> None:
    planner_dataset = tmp_path / "planner_dataset.jsonl"
    executor_trace = tmp_path / "executor_trace.jsonl"
    linked_dataset = tmp_path / "linked_dataset.jsonl"

    planner_dataset.write_text(
        json.dumps(
            {
                "trajectory_id": "traj-1",
                "trace_id": "trace-1",
                "planner_step_id": "planner-step-1",
                "selected_canonical_action": "NAVIGATE_TO_ENTITY",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    executor_trace.write_text(
        json.dumps(
            {
                "kind": "policy_tick",
                "trace_id": "trace-1",
                "planner_step_id": "planner-step-1",
                "step": 3,
                "timestamp": "2026-03-01T00:00:00Z",
                "canonical_action": "NAVIGATE_TO_ENTITY",
                "target_entity_id": "red-ball-01",
                "target_label": "Red Ball",
                "response_ok": True,
                "action": {
                    "joint_positions": {
                        "r_hip_yaw": 0.1,
                    }
                },
                "clamped": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    count = build_linked_dataset(planner_dataset, executor_trace, linked_dataset)
    assert count == 1

    row = json.loads(linked_dataset.read_text(encoding="utf-8").strip())
    assert row["trace_id"] == "trace-1"
    assert row["planner_step_id"] == "planner-step-1"
    assert row["executor"]["target_entity_id"] == "red-ball-01"
