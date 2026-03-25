"""Unified trajectory database for Hyperscape robot training.

Bridges the TypeScript ElizaOS trajectory logger and the Python training pipeline
with a local SQLite store that uses the same column names as the production
PostgreSQL schema.

Usage::

    from training.trajectory_db import TrajectoryDB

    db = TrajectoryDB("my_trajectories.db")
    db.initialize()
    tid = db.insert_trajectory({...})
    db.close()
"""

from training.trajectory_db.db import TrajectoryDB
from training.trajectory_db.models import (
    ActionAttemptRecord,
    ControlFrame,
    EmbodiedContext,
    LLMCallPurpose,
    LLMCallRecord,
    ProviderAccessRecord,
    TrajectoryRecord,
    TrajectoryStatus,
    TrajectoryStepRecord,
)

__all__ = [
    "TrajectoryDB",
    "ActionAttemptRecord",
    "ControlFrame",
    "EmbodiedContext",
    "LLMCallPurpose",
    "LLMCallRecord",
    "ProviderAccessRecord",
    "TrajectoryRecord",
    "TrajectoryStatus",
    "TrajectoryStepRecord",
]
