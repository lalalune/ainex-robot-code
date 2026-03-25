"""GRPO-inspired preference trainer for the AiNex planner.

Uses group-relative reward normalization (from GRPO / DeepSeekMath) to select
preference pairs, then trains via DPO (Direct Preference Optimization).
This is not full on-policy GRPO; it uses the reward-normalization stage to
construct offline preference data for off-policy DPO training.

Pipeline:
  1. Group trajectories by scenario (same initial state, different outcomes).
  2. Normalize rewards within each group: r_hat = (r - mu) / (sigma + eps).
  3. Construct preference pairs where reward gap exceeds a margin.
  4. Output a preference dataset for DPO training.
  5. Optionally run local LoRA DPO fine-tuning via transformers + trl.

Usage
-----
    python -m training.finetune.grpo_trainer \
        --input finetune_data/ \
        --output model/ \
        --epochs 3

    # Preference-pair export only (no training):
    python -m training.finetune.grpo_trainer \
        --input finetune_data/ \
        --output preference_data/ \
        --export-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryEntry:
    """One trajectory from the fine-tuning dataset."""

    trajectory_id: str
    scenario_id: str
    messages: list[dict[str, str]]
    reward: float

    @property
    def prompt_messages(self) -> list[dict[str, str]]:
        """All messages except the final assistant turn (the completion)."""
        if not self.messages:
            return []
        # Everything up to (but not including) the last assistant message
        last_assistant_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is None:
            return self.messages
        return self.messages[:last_assistant_idx]

    @property
    def completion(self) -> str:
        """The final assistant message (the response being optimized)."""
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                return m["content"]
        return ""


@dataclass
class PreferencePair:
    """A chosen/rejected pair for DPO training."""

    prompt: list[dict[str, str]]  # shared prefix (system + user turns)
    chosen: str  # high-reward completion
    rejected: str  # low-reward completion
    chosen_reward: float = 0.0
    rejected_reward: float = 0.0
    scenario_id: str = ""
    chosen_id: str = ""
    rejected_id: str = ""


@dataclass
class ScenarioGroup:
    """All trajectories sharing the same scenario."""

    scenario_id: str
    trajectories: list[TrajectoryEntry] = field(default_factory=list)
    normalized_rewards: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_finetune_data(input_path: Path) -> list[TrajectoryEntry]:
    """Load trajectories from the prepare_dataset output."""
    entries: list[TrajectoryEntry] = []
    paths: list[Path] = []

    if input_path.is_file():
        paths.append(input_path)
    elif input_path.is_dir():
        # Load all splits
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            p = input_path / name
            if p.exists():
                paths.append(p)
        # Also check for raw JSONL files
        for p in sorted(input_path.rglob("*.jsonl")):
            if p not in paths:
                paths.append(p)

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d -- skipping: %s", path, lineno, exc)
                    continue

                messages = row.get("messages", [])
                if not messages:
                    # sharegpt format
                    convs = row.get("conversations", [])
                    messages = [
                        {
                            "role": _from_sharegpt_role(c.get("from", "user")),
                            "content": c.get("value", ""),
                        }
                        for c in convs
                    ]

                meta = row.get("_meta", {})
                entry = TrajectoryEntry(
                    trajectory_id=meta.get("trajectory_id", ""),
                    scenario_id=meta.get("scenario_id", ""),
                    messages=messages,
                    reward=float(meta.get("reward", 0.0)),
                )
                entries.append(entry)

    logger.info("Loaded %d trajectory entries from %s", len(entries), input_path)
    return entries


def _from_sharegpt_role(role: str) -> str:
    return {"human": "user", "gpt": "assistant"}.get(role, role)


# ---------------------------------------------------------------------------
# Grouping and normalization
# ---------------------------------------------------------------------------

def group_by_scenario(entries: list[TrajectoryEntry]) -> list[ScenarioGroup]:
    """Group trajectories by scenario_id."""
    groups_dict: dict[str, list[TrajectoryEntry]] = {}
    for entry in entries:
        key = entry.scenario_id or entry.trajectory_id or str(id(entry))
        groups_dict.setdefault(key, []).append(entry)

    groups: list[ScenarioGroup] = []
    for scenario_id, trajs in sorted(groups_dict.items()):
        if len(trajs) < 2:
            # Need at least 2 trajectories to form a preference pair
            logger.debug(
                "Skipping scenario %s with only %d trajectory",
                scenario_id,
                len(trajs),
            )
            continue
        groups.append(ScenarioGroup(scenario_id=scenario_id, trajectories=trajs))

    logger.info(
        "Formed %d scenario groups from %d entries (min group size = 2)",
        len(groups),
        len(entries),
    )
    return groups


def normalize_rewards(groups: list[ScenarioGroup]) -> None:
    """Normalize rewards within each scenario group using GRPO normalization.

    For each group, compute:
        normalized_i = (reward_i - mean) / (std + eps)

    This is the core GRPO insight: relative performance within the same
    scenario matters more than absolute reward.
    """
    eps = 1e-8
    for group in groups:
        rewards = [t.reward for t in group.trajectories]
        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        std = math.sqrt(variance)

        group.normalized_rewards = [
            (r - mean) / (std + eps) for r in rewards
        ]


# ---------------------------------------------------------------------------
# Preference pair construction
# ---------------------------------------------------------------------------

def build_preference_pairs(
    groups: list[ScenarioGroup],
    margin: float = 0.0,
) -> list[PreferencePair]:
    """Construct preference pairs from scenario groups.

    For each group, pair every higher-reward trajectory with every
    lower-reward trajectory (where the normalized reward gap exceeds
    *margin*).  In practice, for groups of size G this yields up to
    G*(G-1)/2 pairs.
    """
    pairs: list[PreferencePair] = []

    for group in groups:
        n = len(group.trajectories)
        if n < 2:
            continue

        indexed = list(zip(group.trajectories, group.normalized_rewards))
        indexed.sort(key=lambda x: x[1], reverse=True)

        for i in range(n):
            for j in range(i + 1, n):
                traj_high, norm_high = indexed[i]
                traj_low, norm_low = indexed[j]

                if norm_high - norm_low <= margin:
                    continue

                # Ensure prompts are compatible (should be for same scenario)
                prompt = traj_high.prompt_messages
                chosen = traj_high.completion
                rejected = traj_low.completion

                if not chosen or not rejected:
                    continue

                pair = PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    chosen_reward=traj_high.reward,
                    rejected_reward=traj_low.reward,
                    scenario_id=group.scenario_id,
                    chosen_id=traj_high.trajectory_id,
                    rejected_id=traj_low.trajectory_id,
                )
                pairs.append(pair)

    logger.info("Constructed %d preference pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Export preference dataset
# ---------------------------------------------------------------------------

def export_preference_dataset(
    pairs: list[PreferencePair],
    output_dir: Path,
) -> Path:
    """Write preference pairs as JSONL for DPO/GRPO training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "preferences.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            record = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "chosen_reward": pair.chosen_reward,
                "rejected_reward": pair.rejected_reward,
                "_meta": {
                    "scenario_id": pair.scenario_id,
                    "chosen_id": pair.chosen_id,
                    "rejected_id": pair.rejected_id,
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d preference pairs to %s", len(pairs), out_path)

    # Also write a stats file
    stats = {
        "num_pairs": len(pairs),
        "num_scenarios": len({p.scenario_id for p in pairs}),
        "avg_chosen_reward": (
            sum(p.chosen_reward for p in pairs) / len(pairs) if pairs else 0
        ),
        "avg_rejected_reward": (
            sum(p.rejected_reward for p in pairs) / len(pairs) if pairs else 0
        ),
    }
    with open(output_dir / "preference_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Local training (transformers + trl)
# ---------------------------------------------------------------------------

def run_dpo_training(
    preference_path: Path,
    output_dir: Path,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    epochs: int = 3,
    learning_rate: float = 5e-7,
    batch_size: int = 4,
    max_length: int = 2048,
    beta: float = 0.1,
    lora_rank: int = 16,
    lora_alpha: int = 32,
) -> Path:
    """Run DPO training using transformers + trl + peft.

    This performs LoRA-based DPO fine-tuning on the preference dataset.
    Requires: transformers, trl, peft, torch, datasets.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency for local training: {exc}. "
            "Install with: pip install transformers trl peft torch datasets"
        )

    logger.info("Loading preference dataset from %s", preference_path)
    records: list[dict[str, Any]] = []
    with open(preference_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("No preference pairs found.")

    # Convert to the format expected by DPOTrainer
    def format_prompt(messages: list[dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
        return "\n".join(parts) + "\n<|assistant|>\n"

    dataset_dict = {
        "prompt": [format_prompt(r["prompt"]) for r in records],
        "chosen": [r["chosen"] for r in records],
        "rejected": [r["rejected"] for r in records],
    }
    dataset = Dataset.from_dict(dataset_dict)

    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # DPO training config
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
    )

    logger.info("Starting DPO training for %d epochs", epochs)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Model saved to %s", final_path)

    return final_path


# ---------------------------------------------------------------------------
# API-based fine-tuning helpers
# ---------------------------------------------------------------------------

def export_for_openai_dpo(
    pairs: list[PreferencePair],
    output_dir: Path,
) -> Path:
    """Export preference pairs in OpenAI comparison format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "openai_comparisons.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            # OpenAI fine-tuning expects a specific comparisons format
            prompt_text = "\n".join(
                f"[{m['role']}]: {m['content']}" for m in pair.prompt
            )
            record = {
                "input": prompt_text,
                "preferred_output": pair.chosen,
                "non_preferred_output": pair.rejected,
            }
            fh.write(json.dumps(record) + "\n")

    logger.info("Wrote %d OpenAI comparison pairs to %s", len(pairs), out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GRPO trainer for AiNex planner improvement.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to fine-tuning dataset directory (from prepare_dataset).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for model / preference data.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model for fine-tuning.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (default: 0.1).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Minimum normalized reward gap for preference pairs (default: 0.0).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export preference pairs; do not train.",
    )
    parser.add_argument(
        "--export-openai",
        action="store_true",
        help="Also export in OpenAI comparison format.",
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

    # Load
    entries = load_finetune_data(Path(args.input))
    if not entries:
        logger.error("No entries loaded.")
        return

    # Group and normalize
    groups = group_by_scenario(entries)
    if not groups:
        logger.error(
            "No scenario groups with >= 2 trajectories. "
            "GRPO requires multiple trajectories per scenario. "
            "Consider collecting more data or relaxing grouping criteria."
        )
        return

    normalize_rewards(groups)

    # Build preference pairs
    pairs = build_preference_pairs(groups, margin=args.margin)
    if not pairs:
        logger.error("No preference pairs constructed.")
        return

    output_dir = Path(args.output)
    pref_path = export_preference_dataset(pairs, output_dir)

    if args.export_openai:
        export_for_openai_dpo(pairs, output_dir)

    if args.export_only:
        print(f"Preference dataset written to {pref_path}")
        return

    # Train
    try:
        model_path = run_dpo_training(
            preference_path=pref_path,
            output_dir=output_dir,
            model_name=args.model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            beta=args.beta,
            lora_rank=args.lora_rank,
        )
        print(f"Training complete. Model saved to {model_path}")
    except RuntimeError as exc:
        logger.error("Training failed: %s", exc)
        print(f"Preference dataset is available at {pref_path}")
        print(f"Training error: {exc}")


if __name__ == "__main__":
    main()
