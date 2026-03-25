#!/usr/bin/env python3
"""Collect RPG agent trajectories from Hyperscape at scale.

Coordinator script that manages Hyperscape + ElizaOS agent sessions to
collect semantic planning trajectories across all 11 goal types. The
Hyperscape server must be running separately (this script coordinates
trajectory collection, it does not start the game server itself).

Architecture:
    1. Hyperscape server runs the 3D game world (bun run dev:ai)
    2. ElizaOS agents connect as players via plugin-hyperscape
    3. plugin-trajectory-logger captures all agent decisions
    4. This script monitors progress and cycles through goal types
    5. When complete, exports to JSONL/ART/GRPO formats

Prerequisites:
    - Hyperscape server running: cd hyperscape && bun run dev:ai
    - PostgreSQL with trajectory tables (plugin-trajectory-logger schema)
    - ElizaOS agents configured with HYPERSCAPE_AUTH_TOKEN

Usage:
    # Basic collection (default 10,000 episodes)
    python collect_rpg_trajectories.py --episodes 10000 --output trajectories/

    # Targeted collection for specific goal types
    python collect_rpg_trajectories.py --episodes 500 --goals explore,gather,fight

    # Resume interrupted collection
    python collect_rpg_trajectories.py --episodes 10000 --output trajectories/ --resume

    # Export only (no new collection, just convert existing DB data)
    python collect_rpg_trajectories.py --export-only --output trajectories/

    # With custom database URL
    python collect_rpg_trajectories.py --episodes 1000 --db-url postgresql://...
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("collect_rpg_trajectories")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent.parent.parent
HYPERSCAPE_DIR = WORKSPACE / "hyperscape"

# All 11 goal types supported by the Hyperscape RPG environment
GOAL_TYPES = [
    "explore",
    "gather",
    "craft",
    "trade",
    "fight",
    "quest",
    "social",
    "navigate",
    "build",
    "survive",
    "strategize",
]

# Default Hyperscape server endpoints
DEFAULT_GAME_URL = "http://localhost:5555"
DEFAULT_ELIZA_URL = "http://localhost:4001"
DEFAULT_DB_URL = "postgresql://localhost:5432/hyperscape"

# Collection timing
POLL_INTERVAL_SECONDS = 10
GOAL_ROTATION_EPISODES = 50  # Switch goal type every N episodes
HEALTH_CHECK_INTERVAL = 60   # Seconds between server health checks
MAX_CONSECUTIVE_FAILURES = 10  # Abort after this many consecutive errors

# ART/GRPO export settings
GRPO_GROUP_SIZE = 4  # Trajectories per GRPO comparison group


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CollectionProgress:
    """Track progress of trajectory collection."""
    target_episodes: int
    episodes_per_goal: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_episodes: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_episode_time: float = field(default_factory=time.time)

    @property
    def episodes_remaining(self) -> int:
        return max(0, self.target_episodes - self.total_episodes)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def episodes_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed < 1:
            return 0.0
        return self.total_episodes / elapsed

    @property
    def estimated_remaining_seconds(self) -> float:
        eps = self.episodes_per_second
        if eps < 0.001:
            return float("inf")
        return self.episodes_remaining / eps

    def goal_with_fewest_episodes(self, goals: list[str]) -> str:
        """Return the goal type with the fewest collected episodes."""
        return min(goals, key=lambda g: self.episodes_per_goal.get(g, 0))

    def summary(self) -> str:
        elapsed = self.elapsed_seconds
        lines = [
            f"Progress: {self.total_episodes}/{self.target_episodes} episodes "
            f"({self.total_episodes/max(self.target_episodes,1)*100:.1f}%)",
            f"  Total steps: {self.total_steps:,}",
            f"  Mean reward: {self.total_reward/max(self.total_episodes,1):.3f}",
            f"  Errors: {self.errors}",
            f"  Elapsed: {elapsed/60:.1f} min",
            f"  Rate: {self.episodes_per_second:.2f} ep/s",
            f"  ETA: {self.estimated_remaining_seconds/60:.1f} min",
            f"  Episodes per goal:",
        ]
        for goal in GOAL_TYPES:
            count = self.episodes_per_goal.get(goal, 0)
            lines.append(f"    {goal:>12}: {count}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "target_episodes": self.target_episodes,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "errors": self.errors,
            "elapsed_seconds": self.elapsed_seconds,
            "episodes_per_goal": dict(self.episodes_per_goal),
        }


# ---------------------------------------------------------------------------
# Database access (trajectory logger tables)
# ---------------------------------------------------------------------------

class TrajectoryDB:
    """Access trajectory data from the plugin-trajectory-logger PostgreSQL tables."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = None

    def connect(self):
        """Establish database connection."""
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.db_url)
            logger.info("Connected to trajectory database")
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise
        except Exception as exc:
            logger.error(f"Failed to connect to database: {exc}")
            raise

    def close(self):
        if self._conn:
            self._conn.close()

    def count_trajectories(self, since: Optional[float] = None) -> int:
        """Count total trajectories, optionally since a timestamp."""
        with self._conn.cursor() as cur:
            if since:
                cur.execute(
                    "SELECT COUNT(*) FROM trajectories WHERE created_at >= to_timestamp(%s)",
                    (since,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM trajectories")
            return cur.fetchone()[0]

    def count_by_goal(self, since: Optional[float] = None) -> dict[str, int]:
        """Count trajectories grouped by goal type."""
        with self._conn.cursor() as cur:
            query = """
                SELECT
                    COALESCE(metadata_json::json->>'goalDescription', 'unknown') as goal,
                    COUNT(*)
                FROM trajectories
            """
            params: list = []
            if since:
                query += " WHERE created_at >= to_timestamp(%s)"
                params.append(since)
            query += " GROUP BY goal ORDER BY COUNT(*) DESC"
            cur.execute(query, params)
            rows = cur.fetchall()

        result: dict[str, int] = {}
        for goal_desc, count in rows:
            # Map goal descriptions to goal types
            goal_type = _classify_goal_description(goal_desc)
            result[goal_type] = result.get(goal_type, 0) + count
        return result

    def fetch_trajectories(
        self,
        limit: int = 1000,
        offset: int = 0,
        since: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Fetch trajectory records from the database."""
        with self._conn.cursor() as cur:
            query = """
                SELECT
                    trajectory_id, agent_id,
                    start_time, end_time, duration_ms,
                    episode_id, scenario_id, batch_id,
                    steps_json, reward_components_json,
                    metrics_json, metadata_json,
                    total_reward, episode_length,
                    final_status
                FROM trajectories
            """
            params: list = []
            if since:
                query += " WHERE created_at >= to_timestamp(%s)"
                params.append(since)
            query += " ORDER BY created_at ASC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        trajectories = []
        for row in rows:
            record = dict(zip(columns, row))
            # Parse JSON fields
            for json_field in ["steps_json", "reward_components_json",
                               "metrics_json", "metadata_json"]:
                raw = record.pop(json_field, "{}")
                key = json_field.replace("_json", "")
                try:
                    record[key] = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    record[key] = {}
            trajectories.append(record)

        return trajectories

    def get_new_trajectories_since(self, last_id: Optional[str] = None,
                                    limit: int = 100) -> list[dict]:
        """Fetch trajectories newer than last_id for incremental polling."""
        with self._conn.cursor() as cur:
            if last_id:
                cur.execute(
                    """SELECT trajectory_id, agent_id, total_reward, episode_length,
                              final_status, metadata_json, created_at
                       FROM trajectories
                       WHERE trajectory_id > %s
                       ORDER BY created_at ASC LIMIT %s""",
                    (last_id, limit),
                )
            else:
                cur.execute(
                    """SELECT trajectory_id, agent_id, total_reward, episode_length,
                              final_status, metadata_json, created_at
                       FROM trajectories
                       ORDER BY created_at DESC LIMIT %s""",
                    (limit,),
                )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def _classify_goal_description(desc: str) -> str:
    """Classify a free-text goal description into one of the 11 goal types."""
    desc_lower = desc.lower()
    keyword_map = {
        "explore": ["explore", "discover", "scout", "survey", "investigate"],
        "gather": ["gather", "collect", "harvest", "mine", "fish", "chop", "forage"],
        "craft": ["craft", "create", "forge", "brew", "smith", "cook"],
        "trade": ["trade", "buy", "sell", "market", "merchant", "exchange"],
        "fight": ["fight", "attack", "defeat", "kill", "combat", "slay", "battle"],
        "quest": ["quest", "mission", "task", "complete", "deliver"],
        "social": ["talk", "speak", "chat", "greet", "ask", "interact", "social"],
        "navigate": ["walk", "go to", "navigate", "move to", "travel", "head to"],
        "build": ["build", "construct", "place", "erect"],
        "survive": ["survive", "food", "eat", "heal", "rest", "starving"],
        "strategize": ["plan", "prepare", "strategy", "equip", "organize"],
    }
    for goal_type, keywords in keyword_map.items():
        for kw in keywords:
            if kw in desc_lower:
                return goal_type
    return "unknown"


# ---------------------------------------------------------------------------
# Goal injection (configures agents to pursue specific goals)
# ---------------------------------------------------------------------------

class GoalInjector:
    """Inject goal configurations into running ElizaOS agents.

    Communicates with the ElizaOS agent's REST API to set the current
    goal/scenario, cycling through all 11 goal types to ensure balanced
    collection.
    """

    def __init__(self, eliza_url: str = DEFAULT_ELIZA_URL):
        self.eliza_url = eliza_url.rstrip("/")

    def set_goal(self, goal_type: str, episode_id: Optional[str] = None) -> bool:
        """Set the active goal type for the agent."""
        import urllib.request
        import urllib.error

        goal_config = _build_goal_config(goal_type, episode_id)
        payload = json.dumps(goal_config).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.eliza_url}/api/goal",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logger.debug(f"Set goal: {goal_type} (episode: {episode_id})")
                    return True
                else:
                    logger.warning(f"Goal injection returned status {resp.status}")
                    return False
        except urllib.error.URLError as exc:
            logger.warning(f"Failed to inject goal '{goal_type}': {exc}")
            return False
        except Exception as exc:
            logger.warning(f"Goal injection error: {exc}")
            return False

    def health_check(self) -> bool:
        """Check if the ElizaOS agent is reachable."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(
                f"{self.eliza_url}/api/health",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


def _build_goal_config(goal_type: str, episode_id: Optional[str] = None) -> dict:
    """Build a goal configuration payload for the agent."""
    # Goal templates per type
    goal_templates = {
        "explore": [
            "Explore the northern forest and discover what lies beyond",
            "Scout the eastern mountain pass for resources",
            "Investigate the abandoned mine near town",
            "Survey the river delta for fishing spots",
        ],
        "gather": [
            "Gather 10 iron ore from the mines",
            "Collect herbs from the meadow",
            "Chop 20 logs from the forest",
            "Fish at the river until you have 5 fish",
        ],
        "craft": [
            "Craft an iron sword at the forge",
            "Brew a healing potion",
            "Cook fish over a campfire",
            "Smith a set of iron armor",
        ],
        "trade": [
            "Sell your lumber at the market for the best price",
            "Buy a better weapon from the merchant",
            "Trade herbs for potions at the alchemist",
            "Exchange ore for coins at the smith",
        ],
        "fight": [
            "Defeat the goblin camp to the east",
            "Clear the skeleton dungeon",
            "Defeat the bandit leader",
            "Slay the cave spiders",
        ],
        "quest": [
            "Complete the blacksmith's quest to forge a legendary sword",
            "Deliver the merchant's package to the village elder",
            "Help the guard investigate the missing patrol",
            "Complete the herbalist's gathering quest",
        ],
        "social": [
            "Talk to the village elder about the town history",
            "Greet the new merchant and learn what they sell",
            "Ask the guard about recent monster sightings",
            "Chat with other players about forming a party",
        ],
        "navigate": [
            "Walk to the bank in the center of town",
            "Navigate to the dungeon entrance",
            "Travel to the northern outpost",
            "Find and reach the hidden shrine",
        ],
        "build": [
            "Build a fence around the farm",
            "Construct a watchtower on the hill",
            "Place torches along the cave entrance",
            "Build a bridge across the stream",
        ],
        "survive": [
            "Find food before your stamina runs out",
            "Locate shelter before nightfall",
            "Heal your wounds using available resources",
            "Find clean water in the desert",
        ],
        "strategize": [
            "Prepare equipment and supplies for the dungeon raid",
            "Plan the most efficient route to gather all quest items",
            "Organize your inventory for the long expedition",
            "Develop a trading strategy to maximize profit",
        ],
    }

    import random
    templates = goal_templates.get(goal_type, ["Explore the world"])
    goal_description = random.choice(templates)

    return {
        "goalType": goal_type,
        "goalDescription": goal_description,
        "episodeId": episode_id or str(uuid.uuid4()),
        "scenarioId": f"{goal_type}_{uuid.uuid4().hex[:8]}",
        "maxSteps": 200,
        "timeoutSeconds": 600,
        "collectTrajectory": True,
    }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_jsonl(trajectories: list[dict], output_path: Path):
    """Export trajectories to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for traj in trajectories:
            # Ensure JSON serializable (convert datetime objects)
            clean = _json_serialize(traj)
            f.write(json.dumps(clean, separators=(",", ":")) + "\n")
            count += 1
    logger.info(f"Exported {count} trajectories to {output_path}")
    return count


def export_art(trajectories: list[dict], output_path: Path):
    """Export trajectories to ART (Agent Reinforcement Training) format.

    Converts each trajectory into a chat-message sequence with a scalar
    reward, following the ART specification used by OpenPipe and RULER.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for traj in trajectories:
            art_traj = _trajectory_to_art(traj)
            if art_traj:
                f.write(json.dumps(art_traj, separators=(",", ":")) + "\n")
                count += 1
    logger.info(f"Exported {count} ART trajectories to {output_path}")
    return count


def export_grpo_groups(
    trajectories: list[dict],
    output_path: Path,
    group_size: int = GRPO_GROUP_SIZE,
):
    """Export trajectories as GRPO comparison groups.

    Groups trajectories by scenario/goal type, creating groups of
    `group_size` trajectories each with relative rankings computed
    from reward.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by scenario
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for traj in trajectories:
        scenario = traj.get("metadata", {}).get("goalDescription", "unknown")
        goal_type = _classify_goal_description(scenario)
        by_scenario[goal_type].append(traj)

    groups = []
    for goal_type, trajs in by_scenario.items():
        # Create groups of group_size
        for i in range(0, len(trajs) - group_size + 1, group_size):
            group_trajs = trajs[i:i + group_size]
            rewards = [t.get("total_reward", 0.0) for t in group_trajs]

            # Compute rankings (1 = best)
            sorted_indices = sorted(range(len(rewards)), key=lambda k: rewards[k], reverse=True)
            rankings = [0] * len(rewards)
            for rank, idx in enumerate(sorted_indices, 1):
                rankings[idx] = rank

            # Normalize rewards to [-1, 1] within group
            min_r, max_r = min(rewards), max(rewards)
            if max_r > min_r:
                normalized = [2.0 * (r - min_r) / (max_r - min_r) - 1.0 for r in rewards]
            else:
                normalized = [0.0] * len(rewards)

            group = {
                "groupId": f"grp_{uuid.uuid4().hex[:12]}",
                "scenarioId": goal_type,
                "trajectories": [_trajectory_to_art(t) for t in group_trajs],
                "rankings": rankings,
                "normalizedRewards": normalized,
                "createdAt": time.time(),
            }
            groups.append(group)

    count = 0
    with open(output_path, "w") as f:
        for group in groups:
            f.write(json.dumps(group, separators=(",", ":")) + "\n")
            count += 1
    logger.info(f"Exported {count} GRPO groups ({count * group_size} trajectories) to {output_path}")
    return count


def _trajectory_to_art(traj: dict) -> Optional[dict]:
    """Convert a single trajectory to ART message format."""
    steps = traj.get("steps", [])
    if not steps:
        return None

    messages = []

    # System message
    goal = traj.get("metadata", {}).get("goalDescription", "Complete the given task")
    agent_name = traj.get("metadata", {}).get("agentName", "RPG Agent")
    messages.append({
        "role": "system",
        "content": (
            f"You are {agent_name}, an AI agent playing Hyperscape, "
            f"an MMORPG game world. Your current goal: {goal}. "
            f"Observe the environment, reason about your situation, "
            f"and take actions to accomplish your goal efficiently."
        ),
    })

    # Convert each step to user/assistant turns
    for step in steps:
        # User message: environment observation
        obs = step.get("observation", {})
        env_state = step.get("environmentState", {})
        user_content = _format_observation(obs, env_state, step.get("stepNumber", 0))
        if user_content:
            messages.append({"role": "user", "content": user_content})

        # Assistant message: reasoning + action
        action = step.get("action", {})
        reasoning = step.get("reasoning", "")
        assistant_content = _format_action(action, reasoning)
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

    reward = float(traj.get("total_reward", 0.0))
    metadata = {
        "trajectoryId": str(traj.get("trajectory_id", "")),
        "agentId": str(traj.get("agent_id", "")),
        "scenarioId": traj.get("scenario_id", ""),
        "goalType": _classify_goal_description(goal),
        "episodeLength": len(steps),
        "finalStatus": traj.get("metrics", {}).get("finalStatus", "unknown"),
    }

    return {
        "messages": messages,
        "reward": reward,
        "metadata": metadata,
    }


def _format_observation(obs: dict, env_state: dict, step_num: int) -> str:
    """Format an observation dict as a readable user message."""
    parts = [f"[Step {step_num}]"]

    if "position" in obs or "agentPosition" in obs:
        pos = obs.get("position", obs.get("agentPosition", {}))
        if isinstance(pos, dict):
            parts.append(f"Position: ({pos.get('x',0):.1f}, {pos.get('y',0):.1f}, {pos.get('z',0):.1f})")
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            parts.append(f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    if "entities" in obs:
        entities = obs["entities"]
        if entities:
            parts.append(f"Nearby entities ({len(entities)}):")
            for ent in entities[:10]:  # Cap at 10
                label = ent.get("label", ent.get("name", "?"))
                etype = ent.get("entityType", ent.get("type", "?"))
                dist = ent.get("distance_to_agent", ent.get("distance", "?"))
                parts.append(f"  - {label} ({etype}) dist={dist}")

    if "inventory" in obs:
        inv = obs["inventory"]
        if inv:
            items_str = ", ".join(f"{k}: {v}" for k, v in list(inv.items())[:8])
            parts.append(f"Inventory: {items_str}")

    health = env_state.get("health", obs.get("health"))
    if health is not None:
        parts.append(f"Health: {health}")

    return "\n".join(parts)


def _format_action(action: dict, reasoning: str) -> str:
    """Format an action dict as an assistant response."""
    parts = []
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")

    action_type = action.get("actionType", action.get("type", "UNKNOWN"))
    action_name = action.get("actionName", action.get("name", ""))
    params = action.get("parameters", {})

    action_str = f"ACTION: {action_type}"
    if action_name and action_name != action_type:
        action_str += f" {action_name}"
    if params:
        param_str = " ".join(f"{k}={v}" for k, v in params.items())
        action_str += f" [{param_str}]"

    parts.append(action_str)

    result = action.get("result", {})
    if result:
        success = action.get("success", False)
        parts.append(f"Result: {'success' if success else 'failed'}")

    return "\n".join(parts)


def _json_serialize(obj: Any) -> Any:
    """Make an object JSON serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialize(v) for v in obj]
    return obj


def export_statistics(
    progress: CollectionProgress,
    trajectories: list[dict],
    output_dir: Path,
):
    """Export collection statistics and metadata."""
    # Goal distribution
    goal_dist = {}
    for traj in trajectories:
        goal = traj.get("metadata", {}).get("goalDescription", "unknown")
        goal_type = _classify_goal_description(goal)
        goal_dist[goal_type] = goal_dist.get(goal_type, 0) + 1

    stats = {
        "collection_timestamp": datetime.now().isoformat(),
        "total_episodes": len(trajectories),
        "total_steps": sum(
            len(t.get("steps", []))
            for t in trajectories
        ),
        "mean_episode_length": (
            sum(len(t.get("steps", [])) for t in trajectories) / max(len(trajectories), 1)
        ),
        "mean_reward": (
            sum(t.get("total_reward", 0.0) for t in trajectories) / max(len(trajectories), 1)
        ),
        "reward_range": [
            min((t.get("total_reward", 0.0) for t in trajectories), default=0.0),
            max((t.get("total_reward", 0.0) for t in trajectories), default=0.0),
        ],
        "goal_type_distribution": goal_dist,
        "collection_duration_seconds": progress.elapsed_seconds,
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Exported statistics to {stats_path}")

    goal_dist_path = output_dir / "goal_distribution.json"
    with open(goal_dist_path, "w") as f:
        json.dump(goal_dist, f, indent=2)


# ---------------------------------------------------------------------------
# Server health checks
# ---------------------------------------------------------------------------

def check_game_server(game_url: str) -> bool:
    """Check if the Hyperscape game server is reachable."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{game_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(
    episodes: int,
    output_dir: Path,
    goals: list[str],
    db_url: str,
    game_url: str,
    eliza_url: str,
    resume: bool = False,
    export_only: bool = False,
):
    """Run the main trajectory collection loop."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save collection config
    config = {
        "target_episodes": episodes,
        "goals": goals,
        "db_url": db_url.split("@")[-1] if "@" in db_url else db_url,  # strip credentials
        "game_url": game_url,
        "eliza_url": eliza_url,
        "started_at": datetime.now().isoformat(),
        "grpo_group_size": GRPO_GROUP_SIZE,
    }
    with open(output_dir / "collection_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Connect to database
    db = TrajectoryDB(db_url)
    try:
        db.connect()
    except Exception as exc:
        logger.error(f"Cannot connect to trajectory database: {exc}")
        logger.error("Ensure PostgreSQL is running and the trajectory logger schema exists.")
        logger.error("Database URL format: postgresql://user:pass@host:port/dbname")
        sys.exit(1)

    progress = CollectionProgress(target_episodes=episodes)
    collection_start = time.time()

    if export_only:
        logger.info("Export-only mode: fetching existing trajectories from database")
        _export_all(db, output_dir, progress, since=None)
        db.close()
        return

    # Check initial state
    if resume:
        existing = db.count_trajectories()
        logger.info(f"Resume mode: found {existing} existing trajectories in database")
    else:
        collection_start = time.time()

    # Set up goal injector
    goal_injector = GoalInjector(eliza_url)

    # Check servers
    logger.info("Checking server health...")
    if not check_game_server(game_url):
        logger.warning(
            f"Hyperscape game server not reachable at {game_url}. "
            f"Start with: cd hyperscape && bun run dev:ai"
        )
    if not goal_injector.health_check():
        logger.warning(
            f"ElizaOS agent not reachable at {eliza_url}. "
            f"Agents may not respond to goal injection."
        )

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        logger.info("Shutdown requested, finishing current episode...")
        shutdown_requested = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main collection loop
    last_trajectory_id = None
    consecutive_failures = 0
    last_health_check = time.time()
    current_goal_idx = 0
    episodes_since_rotation = 0

    logger.info(f"Starting collection: {episodes} episodes across {len(goals)} goal types")
    logger.info(f"Output directory: {output_dir}")

    while progress.total_episodes < episodes and not shutdown_requested:
        try:
            # Periodic health check
            now = time.time()
            if now - last_health_check > HEALTH_CHECK_INTERVAL:
                if not check_game_server(game_url):
                    logger.warning("Game server unreachable, waiting...")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    last_health_check = now
                    continue
                last_health_check = now

            # Rotate goal type
            if episodes_since_rotation >= GOAL_ROTATION_EPISODES:
                current_goal_idx = (current_goal_idx + 1) % len(goals)
                episodes_since_rotation = 0

            # Pick the goal type with fewest episodes (balanced collection)
            current_goal = progress.goal_with_fewest_episodes(goals)
            episode_id = f"ep_{uuid.uuid4().hex[:12]}"
            goal_injector.set_goal(current_goal, episode_id)

            # Poll for new trajectories
            new_trajs = db.get_new_trajectories_since(last_trajectory_id, limit=50)

            if new_trajs:
                consecutive_failures = 0
                for traj_summary in new_trajs:
                    tid = traj_summary.get("trajectory_id")
                    if tid:
                        last_trajectory_id = tid

                    # Update progress
                    meta_raw = traj_summary.get("metadata_json", "{}")
                    if isinstance(meta_raw, str):
                        try:
                            meta = json.loads(meta_raw)
                        except json.JSONDecodeError:
                            meta = {}
                    else:
                        meta = meta_raw or {}

                    goal_desc = meta.get("goalDescription", "unknown")
                    goal_type = _classify_goal_description(goal_desc)

                    progress.total_episodes += 1
                    progress.episodes_per_goal[goal_type] += 1
                    progress.total_reward += float(traj_summary.get("total_reward", 0.0))
                    progress.total_steps += int(traj_summary.get("episode_length", 0))
                    progress.last_episode_time = time.time()
                    episodes_since_rotation += 1

                # Log progress periodically
                if progress.total_episodes % 100 == 0 or progress.total_episodes == episodes:
                    logger.info(f"\n{progress.summary()}")
            else:
                # No new trajectories, wait and retry
                consecutive_failures += 1
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        f"No new trajectories after {MAX_CONSECUTIVE_FAILURES} polls. "
                        f"Check that agents are running and producing episodes."
                    )
                    consecutive_failures = 0

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as exc:
            logger.error(f"Error in collection loop: {exc}")
            progress.errors += 1
            time.sleep(POLL_INTERVAL_SECONDS)

    # Export collected data
    logger.info(f"\nCollection complete. Final stats:\n{progress.summary()}")
    _export_all(db, output_dir, progress, since=collection_start)
    db.close()


def _export_all(db: TrajectoryDB, output_dir: Path, progress: CollectionProgress,
                since: Optional[float]):
    """Fetch all trajectories and export to all formats."""
    logger.info("Fetching trajectories from database for export...")

    all_trajectories = []
    offset = 0
    batch_size = 500
    while True:
        batch = db.fetch_trajectories(limit=batch_size, offset=offset, since=since)
        if not batch:
            break
        all_trajectories.extend(batch)
        offset += batch_size
        logger.info(f"  Fetched {len(all_trajectories)} trajectories...")

    if not all_trajectories:
        logger.warning("No trajectories to export")
        return

    logger.info(f"Exporting {len(all_trajectories)} trajectories...")

    # JSONL (raw)
    export_jsonl(all_trajectories, output_dir / "trajectories.jsonl")

    # ART format
    art_dir = output_dir / "art"
    art_dir.mkdir(parents=True, exist_ok=True)
    export_art(all_trajectories, art_dir / "trajectories_art.jsonl")

    # GRPO groups
    grpo_dir = output_dir / "grpo"
    grpo_dir.mkdir(parents=True, exist_ok=True)
    export_grpo_groups(all_trajectories, grpo_dir / "grpo_groups.jsonl")

    # Statistics
    export_statistics(progress, all_trajectories, output_dir)

    # Compress JSONL
    jsonl_path = output_dir / "trajectories.jsonl"
    if jsonl_path.exists():
        gz_path = output_dir / "trajectories.jsonl.gz"
        with open(jsonl_path, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                f_out.writelines(f_in)
        logger.info(f"Compressed to {gz_path}")

    logger.info(f"Export complete. Files in {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect RPG agent trajectories from Hyperscape",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
    1. Start Hyperscape: cd hyperscape && bun run dev:ai
    2. Ensure PostgreSQL is running with trajectory logger schema
    3. Configure ElizaOS agents with HYPERSCAPE_AUTH_TOKEN

Examples:
    python collect_rpg_trajectories.py --episodes 10000 --output trajectories/
    python collect_rpg_trajectories.py --episodes 500 --goals explore,gather,fight
    python collect_rpg_trajectories.py --export-only --output trajectories/
        """,
    )
    parser.add_argument(
        "--episodes", type=int, default=10000,
        help="Number of episodes to collect (default: 10000)",
    )
    parser.add_argument(
        "--output", type=str, default="trajectories/",
        help="Output directory for exported data (default: trajectories/)",
    )
    parser.add_argument(
        "--goals", type=str, default=None,
        help="Comma-separated goal types to collect (default: all 11 types)",
    )
    parser.add_argument(
        "--db-url", type=str,
        default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL),
        help="PostgreSQL connection URL (default: from DATABASE_URL env or localhost)",
    )
    parser.add_argument(
        "--game-url", type=str, default=DEFAULT_GAME_URL,
        help=f"Hyperscape game server URL (default: {DEFAULT_GAME_URL})",
    )
    parser.add_argument(
        "--eliza-url", type=str, default=DEFAULT_ELIZA_URL,
        help=f"ElizaOS agent REST API URL (default: {DEFAULT_ELIZA_URL})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted collection (count existing DB trajectories)",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Skip collection, only export existing DB data to files",
    )

    args = parser.parse_args()

    # Resolve goal types
    if args.goals:
        goals = [g.strip() for g in args.goals.split(",")]
        invalid = [g for g in goals if g not in GOAL_TYPES]
        if invalid:
            parser.error(f"Unknown goal types: {invalid}. Valid: {GOAL_TYPES}")
    else:
        goals = list(GOAL_TYPES)

    output_dir = Path(args.output).resolve()

    collect(
        episodes=args.episodes,
        output_dir=output_dir,
        goals=goals,
        db_url=args.db_url,
        game_url=args.game_url,
        eliza_url=args.eliza_url,
        resume=args.resume,
        export_only=args.export_only,
    )


if __name__ == "__main__":
    main()
