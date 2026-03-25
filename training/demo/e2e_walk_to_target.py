"""End-to-end demo: user instruction → planner grounding → executor → bridge.

Usage:
    python -m training.demo.e2e_walk_to_target \
        --instruction "walk to the red ball and wave" \
        --bridge-uri ws://127.0.0.1:9103 \
        [--planner-url http://localhost:5555] \
        [--target-label "red ball"] \
        [--emote wave]

This demonstrates the full pipeline:
1. Parse user instruction into a canonical intent
2. (Optionally) query Hyperscape planner snapshot for entity grounding
3. Start the AinexExecutionService against the bridge
4. Run navigation + optional emote
5. Report result
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_demo")


def fetch_planner_context(planner_url: str, agent_id: str) -> dict | None:
    """Fetch canonical planner context from Hyperscape snapshot endpoint."""
    try:
        import urllib.request
        url = f"{planner_url}/hyperscape/snapshot/{agent_id}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("success"):
                return data.get("canonicalPlannerContext")
    except Exception as exc:
        logger.warning("Could not fetch planner context: %s", exc)
    return None


def ground_entity_from_context(
    planner_context: dict | None,
    target_label: str,
    instruction: str,
) -> tuple[str, str, list[float] | None]:
    """Resolve a target entity from the canonical planner context.

    Returns (entity_id, label, position) or ("", target_label, None).
    """
    if planner_context is None:
        return "", target_label, None

    entities = planner_context.get("entities", [])
    instruction_lower = instruction.lower()
    target_lower = target_label.lower()

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        label = entity.get("label", "")
        if not isinstance(label, str):
            continue
        label_lower = label.lower()

        # Match by target label or instruction keywords
        if target_lower and target_lower in label_lower:
            return (
                entity.get("entityId", ""),
                label,
                entity.get("position"),
            )
        if "ball" in instruction_lower and "ball" in label_lower:
            return (
                entity.get("entityId", ""),
                label,
                entity.get("position"),
            )

    # No match — return best guess
    return "", target_label, None


def infer_canonical_action(instruction: str) -> str:
    """Infer a canonical action type from the instruction text."""
    lower = instruction.lower()
    if "pick up" in lower or "pickup" in lower or "grab" in lower:
        return "PICKUP_ENTITY"
    if "emote" in lower or "wave" in lower or "bow" in lower:
        # If instruction is ONLY about emoting, use EMOTE
        if not any(w in lower for w in ["walk", "go", "move", "navigate"]):
            return "EMOTE"
    if "stop" in lower or "idle" in lower or "abort" in lower:
        return "IDLE"
    return "NAVIGATE_TO_ENTITY"


def infer_emote(instruction: str) -> str:
    """Infer emote name from instruction."""
    lower = instruction.lower()
    if "bow" in lower:
        return "bow"
    if "wave" in lower or "emote" in lower:
        return "wave"
    return ""


async def run_demo(
    instruction: str,
    bridge_uri: str,
    planner_url: str | None = None,
    planner_agent_id: str = "",
    target_label: str = "",
    emote: str = "",
    max_steps: int = 200,
    hz: float = 20.0,
    trace_path: str = "",
) -> dict:
    """Run the end-to-end demo."""

    trace_id = f"e2e-demo-{uuid.uuid4()}"
    planner_step_id = f"planner-{uuid.uuid4()}"
    start_time = time.time()

    logger.info("=== E2E Demo: %s ===", instruction)
    logger.info("Bridge: %s", bridge_uri)
    logger.info("Trace ID: %s", trace_id)

    # Step 1: Fetch planner context (optional)
    planner_context = None
    if planner_url:
        logger.info("Fetching planner context from %s ...", planner_url)
        planner_context = fetch_planner_context(planner_url, planner_agent_id)
        if planner_context:
            entity_count = len(planner_context.get("entities", []))
            logger.info("Planner context: %d entities", entity_count)
        else:
            logger.info("No planner context available (will use instruction-only grounding)")

    # Step 2: Ground target entity
    entity_id, entity_label, entity_position = ground_entity_from_context(
        planner_context, target_label or "red ball", instruction
    )
    canonical_action = infer_canonical_action(instruction)
    emote_name = emote or infer_emote(instruction)

    logger.info("Grounded: entity_id=%s label=%s pos=%s", entity_id, entity_label, entity_position)
    logger.info("Canonical action: %s", canonical_action)
    if emote_name:
        logger.info("Emote after navigation: %s", emote_name)

    # Step 3: Connect to bridge and run execution
    # Use the real AiNex bridge client from the Eliza plugin
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eliza" / "packages" / "python"))
    from elizaos_plugin_ainex.bridge_client import AiNexBridgeClient

    client = AiNexBridgeClient(url=bridge_uri)
    try:
        await client.connect()
    except Exception as exc:
        logger.error("Failed to connect to bridge at %s: %s", bridge_uri, exc)
        return {"success": False, "reason": "bridge_connection_failed"}

    logger.info("Connected to bridge")

    # Step 4: Start policy mode
    start_resp = await client.policy_start(
        task=instruction,
        hz=hz,
        max_steps=max_steps,
        trace_id=trace_id,
        planner_step_id=planner_step_id,
        canonical_action=canonical_action,
        target_entity_id=entity_id,
        target_label=entity_label,
    )
    if not start_resp.ok:
        logger.error("policy_start failed: %s", start_resp.message)
        await client.disconnect()
        return {"success": False, "reason": f"policy_start_failed:{start_resp.message}"}

    logger.info("Policy started, running %d steps at %.1f Hz", max_steps, hz)

    # Step 5: Run control loop (simple forward walk for demo)
    trace_records = []
    steps_completed = 0
    try:
        interval = 1.0 / max(hz, 1.0)
        for step in range(max_steps):
            steps_completed = step + 1

            # Build walk command steering toward the grounded entity
            import math
            if entity_position and len(entity_position) >= 2:
                # entity_position is in world/game frame; treat as body-relative
                dx = float(entity_position[0])
                dy = float(entity_position[1]) if len(entity_position) > 1 else 0.0
                dist = math.sqrt(dx * dx + dy * dy)
                bearing = math.atan2(dy, dx)
                walk_x = min(0.02, dist * 0.01)  # proportional forward
                walk_yaw = max(-5.0, min(5.0, bearing * 3.0))  # steer toward target
            else:
                walk_x = 0.02  # default forward
                walk_yaw = 0.0

            action_dict = {
                "walk_x": walk_x,
                "walk_y": 0.0,
                "walk_yaw": walk_yaw,
                "walk_height": 0.035,
                "walk_speed": 2,
                "head_pan": 0.0,
                "head_tilt": 0.0,
            }

            tick_resp = await client.policy_tick(action_dict)
            if not tick_resp.ok:
                logger.warning("policy_tick failed at step %d: %s", step, tick_resp.message)
                break

            trace_records.append({
                "step": step,
                "timestamp": time.time(),
                "action": action_dict,
                "trace_id": trace_id,
            })

            if step % 20 == 0:
                logger.info("Step %d/%d", step, max_steps)

            await asyncio.sleep(interval)

    finally:
        await client.policy_stop(reason="demo_complete")
        logger.info("Policy stopped after %d steps", steps_completed)

    # Step 6: Emote (if requested)
    if emote_name:
        logger.info("Playing emote: %s", emote_name)
        emote_resp = await client.action_play(emote_name)
        if emote_resp.ok:
            logger.info("Emote played successfully")
        else:
            logger.warning("Emote failed: %s", emote_resp.message)

    await client.disconnect()
    elapsed = time.time() - start_time

    # Step 7: Save trace
    result = {
        "success": True,
        "instruction": instruction,
        "canonical_action": canonical_action,
        "target_entity_id": entity_id,
        "target_label": entity_label,
        "target_position": entity_position,
        "emote": emote_name,
        "trace_id": trace_id,
        "planner_step_id": planner_step_id,
        "steps_completed": steps_completed,
        "elapsed_seconds": round(elapsed, 2),
        "planner_context_available": planner_context is not None,
    }

    if trace_path:
        output = Path(trace_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump({
                "result": result,
                "trace": trace_records,
            }, f, indent=2)
        logger.info("Trace saved to %s", trace_path)

    logger.info("=== Demo complete: %s ===", "SUCCESS" if result["success"] else "FAILED")
    logger.info("Result: %s", json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end walk-to-target demo")
    parser.add_argument("--instruction", type=str, default="walk to the red ball and wave")
    parser.add_argument("--bridge-uri", type=str, default="ws://127.0.0.1:9103")
    parser.add_argument("--planner-url", type=str, default="")
    parser.add_argument("--planner-agent-id", type=str, default="")
    parser.add_argument("--target-label", type=str, default="red ball")
    parser.add_argument("--emote", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--trace-path", type=str,
                        default="/home/shaw/Documents/hyperscape-robot-workspace/end_to_end_outputs/traces/e2e_demo_trace.json")

    args = parser.parse_args()

    result = asyncio.run(run_demo(
        instruction=args.instruction,
        bridge_uri=args.bridge_uri,
        planner_url=args.planner_url or None,
        planner_agent_id=args.planner_agent_id,
        target_label=args.target_label,
        emote=args.emote,
        max_steps=args.max_steps,
        hz=args.hz,
        trace_path=args.trace_path,
    ))

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
