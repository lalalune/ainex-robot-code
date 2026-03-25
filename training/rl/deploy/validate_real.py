"""Validation harness for real robot deployment.

Runs a sequence of progressively more aggressive tests against the
real robot, recording commanded vs actual joint positions at each stage.

Usage:
    python -m training.rl.deploy.validate_real --bridge ws://localhost:9100
    python -m training.rl.deploy.validate_real --bridge ws://localhost:9100 --stage walk
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np

from training.mujoco.ainex_constants import ALL_JOINT_NAMES, LEG_JOINT_NAMES
from bridge.isaaclab.joint_map import joint_name_to_servo_id, radians_to_pulse


STAGES = {
    "servo_ping": "Read servo positions (no commands sent)",
    "stand": "Send standing pose for 5 seconds",
    "sway": "Small 0.05 rad hip sway for 5 seconds",
    "walk": "Walk forward at 0.1 m/s for 5 seconds",
}


async def run_validation(bridge_url: str, stage: str, duration: float = 5.0) -> dict:
    """Run a single validation stage and return recorded telemetry."""
    import websockets

    log: list[dict] = []
    print(f"\n--- Stage: {stage} ({STAGES[stage]}) ---")
    print(f"Duration: {duration}s")

    async with websockets.connect(bridge_url) as ws:
        # Start policy mode
        await _send(ws, "policy.start", {"task": f"validate_{stage}", "hz": 20})
        resp = await _recv(ws)
        if not resp.get("ok"):
            print(f"  Failed to start policy: {resp.get('message')}")
            return {"stage": stage, "success": False, "error": resp.get("message"), "log": []}

        start = time.monotonic()
        step = 0
        try:
            while time.monotonic() - start < duration:
                step += 1
                elapsed = time.monotonic() - start

                # Compute command based on stage
                if stage == "servo_ping":
                    # Don't send any commands, just read telemetry
                    await _send(ws, "status.get", {})
                elif stage == "stand":
                    cmd = {name: 0.0 for name in LEG_JOINT_NAMES}
                    await _send(ws, "policy.tick", {"joint_positions": cmd, "duration": 50})
                elif stage == "sway":
                    angle = 0.05 * np.sin(elapsed * 2.0 * np.pi * 0.5)
                    cmd = {name: 0.0 for name in LEG_JOINT_NAMES}
                    cmd["r_hip_roll"] = float(angle)
                    cmd["l_hip_roll"] = float(angle)
                    await _send(ws, "policy.tick", {"joint_positions": cmd, "duration": 50})
                elif stage == "walk":
                    # Use BraxWalkSkill or simple walk command
                    await _send(ws, "walk.set", {
                        "x": 0.01, "y": 0.0, "yaw": 0.0,
                        "height": 0.036, "speed": 1,
                    })

                resp = await _recv(ws)
                telemetry = resp.get("data", {})
                telemetry["step"] = step
                telemetry["elapsed"] = elapsed
                telemetry["stage"] = stage
                log.append(telemetry)

                # Status every 1 second
                if step % 20 == 0:
                    imu_r = telemetry.get("imu_roll", 0)
                    imu_p = telemetry.get("imu_pitch", 0)
                    print(f"  t={elapsed:.1f}s step={step} imu=({imu_r:.3f}, {imu_p:.3f})")

                await asyncio.sleep(0.05)  # 20Hz

        except KeyboardInterrupt:
            print("  Interrupted!")
        finally:
            # Stop and return to standing
            await _send(ws, "policy.stop", {})
            try:
                await _recv(ws)
            except Exception:
                pass

    print(f"  Completed: {len(log)} frames recorded")
    return {"stage": stage, "success": True, "log": log}


async def _send(ws, command: str, payload: dict) -> None:
    msg = {
        "type": "command",
        "request_id": f"validate-{time.monotonic():.3f}",
        "command": command,
        "payload": payload,
    }
    await ws.send(json.dumps(msg))


async def _recv(ws, timeout: float = 2.0) -> dict:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(raw)
    except asyncio.TimeoutError:
        return {"ok": False, "message": "timeout"}
    except Exception as exc:
        return {"ok": False, "message": f"recv error: {exc}"}


def analyze_log(log: list[dict]) -> dict:
    """Analyze recorded telemetry for sim-to-real gap metrics."""
    if not log:
        return {}

    imu_rolls = [f.get("imu_roll", 0) for f in log]
    imu_pitches = [f.get("imu_pitch", 0) for f in log]

    return {
        "frames": len(log),
        "duration_s": log[-1].get("elapsed", 0) if log else 0,
        "imu_roll_mean": float(np.mean(imu_rolls)),
        "imu_roll_std": float(np.std(imu_rolls)),
        "imu_roll_max": float(np.max(np.abs(imu_rolls))),
        "imu_pitch_mean": float(np.mean(imu_pitches)),
        "imu_pitch_std": float(np.std(imu_pitches)),
        "imu_pitch_max": float(np.max(np.abs(imu_pitches))),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate real robot deployment")
    parser.add_argument("--bridge", default="ws://localhost:9100")
    parser.add_argument("--stage", default="servo_ping", choices=list(STAGES.keys()),
                        help="Validation stage to run")
    parser.add_argument("--all", action="store_true", help="Run all stages in sequence")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration per stage")
    parser.add_argument("--output", default=None, help="Save telemetry log to JSON file")
    args = parser.parse_args()

    stages = list(STAGES.keys()) if args.all else [args.stage]
    all_results = []

    for stage in stages:
        result = asyncio.run(run_validation(args.bridge, stage, args.duration))
        analysis = analyze_log(result.get("log", []))
        result["analysis"] = analysis
        all_results.append(result)

        if analysis:
            print(f"\n  Analysis:")
            print(f"    IMU roll:  mean={analysis['imu_roll_mean']:.4f} "
                  f"std={analysis['imu_roll_std']:.4f} max={analysis['imu_roll_max']:.4f}")
            print(f"    IMU pitch: mean={analysis['imu_pitch_mean']:.4f} "
                  f"std={analysis['imu_pitch_std']:.4f} max={analysis['imu_pitch_max']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nTelemetry saved to {args.output}")

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    for r in all_results:
        status = "PASS" if r.get("success") else "FAIL"
        frames = len(r.get("log", []))
        print(f"  {r['stage']:15s}: {status} ({frames} frames)")


if __name__ == "__main__":
    main()
