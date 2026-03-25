"""Prepare planner trajectories for fine-tuning.

Reads Hyperscape planner trajectory JSONL (in ART/GRPO export format) and
converts to chat-completion fine-tuning format suitable for:
  - OpenAI fine-tuning API
  - Anthropic fine-tuning API
  - Local training with transformers + trl

Quality filtering, train/val/test splitting, and output formatting are all
handled here.

Usage
-----
    python -m training.finetune.prepare_dataset \
        --input planner_trajectories.jsonl \
        --output finetune_data/ \
        --min-reward 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default system prompt for the AiNex planner
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous robot planner for the AiNex humanoid robot. "
    "Given the robot's current state and environment observations, select "
    "the best high-level action and provide the parameters needed to execute "
    "it. Explain your reasoning briefly before choosing the action."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class TrajectoryRecord:
    """One trajectory loaded from JSONL (ART format or raw planner trace)."""

    trajectory_id: str
    scenario_id: str
    messages: list[ChatMessage]
    reward: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FineTuneExample:
    """One training example in chat-completion format."""

    messages: list[dict[str, str]]
    # Metadata (not sent to the model, but useful for provenance)
    trajectory_id: str = ""
    scenario_id: str = ""
    reward: float = 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_art_trajectory(row: dict[str, Any]) -> TrajectoryRecord | None:
    """Parse one line of ART-format JSONL."""
    messages_raw = row.get("messages", [])
    if not messages_raw:
        return None

    messages = [
        ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
        for m in messages_raw
        if m.get("content")
    ]
    if not messages:
        return None

    reward = float(row.get("reward", 0.0))
    meta = row.get("metadata", {})
    traj_id = meta.get("trajectoryId", row.get("trajectoryId", ""))
    scenario_id = meta.get("scenarioId", row.get("scenarioId", ""))

    # Determine success from metadata or reward
    metrics = row.get("metrics", {})
    final_status = meta.get("metrics", {}).get("finalStatus", "")
    success = final_status == "completed" or reward > 0

    return TrajectoryRecord(
        trajectory_id=traj_id,
        scenario_id=scenario_id,
        messages=messages,
        reward=reward,
        success=success,
        metadata=meta,
    )


def _load_planner_step_trajectory(row: dict[str, Any]) -> TrajectoryRecord | None:
    """Parse a planner-level trajectory where each row is a step with
    observation + action dicts (the Hyperscape control-loop format)."""
    # This format doesn't directly have messages; we construct them
    # from the observation prompt and action output.
    obs = row.get("observation", {})
    prompt = obs.get("prompt", "")
    if not prompt:
        prompt = row.get("observation_summary", {}).get("prompt", "")
    if not prompt:
        return None

    # Build a user message from the observation context
    state_summary = row.get("observation_summary", {})
    user_parts = [f"Task: {prompt}"]
    if state_summary.get("state_dim"):
        user_parts.append(f"State dimension: {state_summary['state_dim']}")
    if state_summary.get("has_image"):
        user_parts.append("Visual observation is available.")
    user_content = "\n".join(user_parts) + "\n\nWhat action should the robot take?"

    # Build an assistant message from the action
    action = row.get("clamped") or row.get("action") or row.get("action_summary", {})
    if not action:
        return None

    assistant_parts = [f"I will execute the following action for '{prompt}':"]
    for key, val in action.items():
        if isinstance(val, (int, float)):
            assistant_parts.append(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")
    assistant_content = "\n".join(assistant_parts)

    messages = [
        ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
        ChatMessage(role="assistant", content=assistant_content),
    ]

    reward = float(row.get("reward", 0.0))
    trace_id = row.get("trace_id", "")

    return TrajectoryRecord(
        trajectory_id=trace_id,
        scenario_id=row.get("canonical_action", ""),
        messages=messages,
        reward=reward,
        success=reward > 0,
        metadata={"step": row.get("step", 0)},
    )


def _is_hyperscape_export(row: dict[str, Any]) -> bool:
    """Return True if *row* uses the hyperscape-planner-export-v1 schema."""
    if row.get("schema_version") == "hyperscape-planner-export-v1":
        return True
    # Fallback heuristic: top-level ``steps`` list with expected sub-keys.
    steps = row.get("steps")
    if isinstance(steps, list) and steps:
        first = steps[0]
        if isinstance(first, dict) and {"action", "llmCalls"} & set(first.keys()):
            return True
    return False


def _load_hyperscape_export_trajectory(
    row: dict[str, Any],
) -> list[TrajectoryRecord]:
    """Parse one row of ``hyperscape-planner-export-v1`` JSONL.

    Each row is a *complete trajectory* that contains a ``steps[]`` array.
    Every step may carry one or more ``llmCalls`` – only calls with a real
    model (i.e. not the synthetic fallback) produce training examples.

    When a step has no real LLM call but *does* have ``environmentState`` and
    an ``action`` with a non-trivial ``actionName``, we still attempt to
    build a training example from the structured fields.

    Returns a list because a single trajectory row can yield zero-to-many
    ``TrajectoryRecord`` objects (one per usable step).
    """
    trajectory_id = row.get("trajectory_id", "")
    meta = row.get("metadata") or {}
    status = row.get("status", "")
    steps = row.get("steps", [])

    records: list[TrajectoryRecord] = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        step_number = step.get("stepNumber") or step.get("step_number", 0)
        reward = float(step.get("reward", 0.0))
        done = step.get("done", False)
        action = step.get("action") or {}
        env_state = step.get("environmentState") or {}
        llm_calls = step.get("llmCalls") or step.get("llm_calls") or []

        # ------------------------------------------------------------------
        # Strategy 1: Use a real (non-synthetic) LLM call if available
        # ------------------------------------------------------------------
        for call in llm_calls:
            if not isinstance(call, dict):
                continue
            model = call.get("model", "")
            # Skip synthetic placeholder calls
            if "synthetic" in model.lower() or "fallback" in model.lower():
                continue
            system_prompt = (call.get("systemPrompt") or "").strip()
            user_prompt = (call.get("userPrompt") or "").strip()
            response = (call.get("response") or "").strip()
            if not user_prompt or not response:
                continue

            messages = []
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            else:
                messages.append(
                    ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT)
                )
            messages.append(ChatMessage(role="user", content=user_prompt))
            messages.append(ChatMessage(role="assistant", content=response))

            # Build a scenario_id from the goal or action purpose
            goal_type = meta.get("currentGoalType", "")
            goal_desc = meta.get("currentGoalDescription", "")
            scenario_id = goal_type or call.get("purpose", "")

            records.append(
                TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    scenario_id=scenario_id,
                    messages=messages,
                    reward=reward,
                    success=action.get("success", False) or status == "completed",
                    metadata={
                        "step_number": step_number,
                        "action_name": action.get("actionName"),
                        "action_type": action.get("actionType"),
                        "model": model,
                        "goal_type": goal_type,
                        "goal_description": goal_desc,
                        "source": meta.get("source"),
                        "environment_state": env_state,
                    },
                )
            )
            # One record per real LLM call is enough per step
            break

        # ------------------------------------------------------------------
        # Strategy 2: No real LLM call – build from structured action data
        # ------------------------------------------------------------------
        # Only if we didn't already produce a record from an LLM call above
        if records and records[-1].trajectory_id == trajectory_id and any(
            r.metadata.get("step_number") == step_number for r in records
        ):
            continue  # already handled

        action_name = action.get("actionName", "")
        if not action_name or action_name in ("pending", ""):
            continue  # nothing actionable

        # Construct a user message from environment + metadata context
        user_parts: list[str] = []
        goal_desc = meta.get("currentGoalDescription", "")
        goal_type = meta.get("currentGoalType", "")
        if goal_desc:
            user_parts.append(f"Current goal: {goal_desc}")
        if goal_type:
            user_parts.append(f"Goal type: {goal_type}")
        if env_state:
            env_lines = [f"  {k}: {v}" for k, v in env_state.items()]
            user_parts.append("Environment state:\n" + "\n".join(env_lines))
        if not user_parts:
            continue
        user_parts.append("\nWhat action should be taken?")
        user_content = "\n".join(user_parts)

        # Construct an assistant message from the action
        params = action.get("parameters", {})
        assistant_parts = [f"ACTION: {action_name}"]
        if params:
            for k, v in params.items():
                assistant_parts.append(f"  {k}: {v}")
        assistant_content = "\n".join(assistant_parts)

        messages = [
            ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
            ChatMessage(role="assistant", content=assistant_content),
        ]

        records.append(
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                scenario_id=goal_type or action_name,
                messages=messages,
                reward=reward,
                success=action.get("success", False) or status == "completed",
                metadata={
                    "step_number": step_number,
                    "action_name": action_name,
                    "action_type": action.get("actionType"),
                    "source": meta.get("source"),
                    "environment_state": env_state,
                },
            )
        )

    return records


def load_trajectories(input_path: Path) -> list[TrajectoryRecord]:
    """Load trajectories from a JSONL file or directory of JSONL files."""
    paths: list[Path] = []
    if input_path.is_file():
        paths.append(input_path)
    elif input_path.is_dir():
        paths.extend(sorted(input_path.rglob("*.jsonl")))
        paths.extend(sorted(input_path.rglob("*.json")))
    else:
        logger.error("Input path does not exist: %s", input_path)
        return []

    records: list[TrajectoryRecord] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            if path.suffix == ".json":
                try:
                    data = json.load(fh)
                    rows = data if isinstance(data, list) else [data]
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse %s: %s", path, exc)
                    continue
            else:
                rows = []
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("%s:%d -- skipping: %s", path, lineno, exc)

        for row in rows:
            # Try hyperscape-planner-export-v1 format first (has "steps" array)
            if _is_hyperscape_export(row):
                records.extend(_load_hyperscape_export_trajectory(row))
                continue
            # Try ART format (has "messages" key)
            rec = _load_art_trajectory(row)
            if rec is None:
                # Fall back to planner-step format
                rec = _load_planner_step_trajectory(row)
            if rec is not None:
                records.append(rec)

    logger.info("Loaded %d trajectory records from %s", len(records), input_path)
    return records


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_trajectories(
    records: list[TrajectoryRecord],
    min_reward: float = 0.0,
    require_success: bool = False,
    min_messages: int = 2,
) -> list[TrajectoryRecord]:
    """Filter trajectories by quality criteria."""
    filtered: list[TrajectoryRecord] = []
    stats = {"total": len(records), "low_reward": 0, "not_success": 0, "too_short": 0}

    for rec in records:
        if rec.reward < min_reward:
            stats["low_reward"] += 1
            continue
        if require_success and not rec.success:
            stats["not_success"] += 1
            continue
        if len(rec.messages) < min_messages:
            stats["too_short"] += 1
            continue
        filtered.append(rec)

    logger.info(
        "Filtered %d -> %d trajectories (dropped: %d low reward, "
        "%d not success, %d too short)",
        stats["total"],
        len(filtered),
        stats["low_reward"],
        stats["not_success"],
        stats["too_short"],
    )
    return filtered


# ---------------------------------------------------------------------------
# Conversion to fine-tuning format
# ---------------------------------------------------------------------------

def trajectory_to_finetune(rec: TrajectoryRecord) -> FineTuneExample:
    """Convert a TrajectoryRecord to a chat-completion fine-tuning example."""
    messages: list[dict[str, str]] = []

    has_system = any(m.role == "system" for m in rec.messages)
    if not has_system:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    for m in rec.messages:
        messages.append({"role": m.role, "content": m.content})

    return FineTuneExample(
        messages=messages,
        trajectory_id=rec.trajectory_id,
        scenario_id=rec.scenario_id,
        reward=rec.reward,
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _deterministic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def split_dataset(
    examples: list[FineTuneExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[FineTuneExample]]:
    """Split examples into train/val/test.

    Uses scenario_id for stratified splitting when available so that
    examples from the same scenario stay in the same split.  Falls back
    to random shuffling otherwise.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    # Group by scenario
    scenario_groups: dict[str, list[FineTuneExample]] = {}
    for ex in examples:
        key = ex.scenario_id or ex.trajectory_id or str(id(ex))
        scenario_groups.setdefault(key, []).append(ex)

    # Sort keys deterministically then shuffle with seed
    keys = sorted(scenario_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    splits: dict[str, list[FineTuneExample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for key in train_keys:
        splits["train"].extend(scenario_groups[key])
    for key in val_keys:
        splits["val"].extend(scenario_groups[key])
    for key in test_keys:
        splits["test"].extend(scenario_groups[key])

    # If val or test ended up empty, move some from train
    if not splits["val"] and len(splits["train"]) > 2:
        splits["val"].append(splits["train"].pop())
    if not splits["test"] and len(splits["train"]) > 2:
        splits["test"].append(splits["train"].pop())

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_splits(
    splits: dict[str, list[FineTuneExample]],
    output_dir: Path,
    format: str = "openai",
) -> dict[str, Path]:
    """Write split JSONL files.

    Formats:
      - "openai": ``{"messages": [...]}`` per line (OpenAI / Anthropic API)
      - "sharegpt": ``{"conversations": [...]}`` per line (popular open-source)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split_name, examples in splits.items():
        if not examples:
            continue
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for ex in examples:
                if format == "sharegpt":
                    record = {
                        "conversations": [
                            {"from": _sharegpt_role(m["role"]), "value": m["content"]}
                            for m in ex.messages
                        ],
                    }
                else:
                    record = {"messages": ex.messages}

                # Always include provenance metadata
                record["_meta"] = {
                    "trajectory_id": ex.trajectory_id,
                    "scenario_id": ex.scenario_id,
                    "reward": ex.reward,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        paths[split_name] = path
        logger.info("Wrote %d examples to %s", len(examples), path)

    # Summary file
    summary = {
        "format": format,
        "splits": {k: len(v) for k, v in splits.items()},
        "total_examples": sum(len(v) for v in splits.values()),
    }
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return paths


def _sharegpt_role(role: str) -> str:
    return {"system": "system", "user": "human", "assistant": "gpt"}.get(role, role)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare planner trajectories for fine-tuning.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to planner trajectory JSONL file or directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for fine-tuning datasets.",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Minimum reward threshold for inclusion (default: 0.0).",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Only include successful trajectories.",
    )
    parser.add_argument(
        "--format",
        choices=["openai", "sharegpt"],
        default="openai",
        help="Output format (default: openai).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    records = load_trajectories(Path(args.input))
    if not records:
        logger.error("No trajectories loaded. Check your input path.")
        return

    filtered = filter_trajectories(
        records,
        min_reward=args.min_reward,
        require_success=args.require_success,
    )
    if not filtered:
        logger.error("All trajectories filtered out. Try lowering --min-reward.")
        return

    examples = [trajectory_to_finetune(r) for r in filtered]
    splits = split_dataset(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    paths = write_splits(splits, Path(args.output), format=args.format)

    print(f"Fine-tuning dataset written to {args.output}")
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    main()
