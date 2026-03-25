#!/usr/bin/env bash
# Train AiNex walking policy on Nebius cloud GPU.
#
# Usage (on the Nebius VM):
#   bash train_on_nebius.sh
#
# This script:
#   1. Installs dependencies (JAX + CUDA, MuJoCo, etc)
#   2. Runs training with correct physics (Kp=200, real init pose, STL feet)
#   3. Evaluates the checkpoint
#   4. Runs standing validation
#   5. Packages artifacts for download

set -euo pipefail

CHECKPOINT_DIR="$HOME/training_output/mujoco_walk_v1"
EVAL_DIR="$HOME/training_output/eval"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_EVALS="${NUM_EVALS:-20}"

echo "============================================================"
echo "AiNex MuJoCo Training — Nebius Cloud"
echo "  Steps: $NUM_TIMESTEPS"
echo "  Envs:  $NUM_ENVS"
echo "  Output: $CHECKPOINT_DIR"
echo "============================================================"

# --- Step 0: Install dependencies ---
echo "[0/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet "jax[cuda12]" mujoco mujoco-mjx brax ml-collections numpy pillow

# Clone the repo if not already present
if [ ! -d "$HOME/ainex-robot-code" ]; then
    echo "ERROR: ainex-robot-code not found. Upload it first:"
    echo "  scp -r ainex-robot-code/ user@VM_IP:~/ainex-robot-code/"
    exit 1
fi

cd "$HOME/ainex-robot-code"

# --- Step 1: Pre-flight standing test ---
echo ""
echo "[1/5] Pre-flight standing test..."
python3 -c "
import mujoco, numpy as np
from training.mujoco.joystick import Joystick
env = Joystick()
model = env.mj_model
data = mujoco.MjData(model)
data.qpos[:] = np.array(env._init_q)
data.qvel[:] = 0
for i in range(min(model.nu, len(env._default_pose))):
    data.ctrl[i] = float(env._default_pose[i])
mujoco.mj_forward(model, data)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'body_link')
for _ in range(1250):
    mujoco.mj_step(model, data)
z = data.xpos[body_id, 2]
kp = env.mj_model.actuator_gainprm[0, 0]
print(f'Standing test: z={z:.4f}m kp={kp:.0f}')
assert z > 0.10, f'FAILED: robot fell (z={z:.4f}). Fix physics before training.'
print('PASSED')
"
echo "Pre-flight: PASSED"

# --- Step 2: Train ---
echo ""
echo "[2/5] Training ($NUM_TIMESTEPS steps, $NUM_ENVS envs)..."
python3 -m training.mujoco.train \
    --num-timesteps "$NUM_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --num-evals "$NUM_EVALS" \
    --checkpoint-dir "$CHECKPOINT_DIR"

echo "Training complete."
cat "$CHECKPOINT_DIR/metrics.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if isinstance(d, list):
    for e in d[-3:]:
        print(f'  Step {e[\"steps\"]:>12,} | Reward: {e[\"reward\"]:>8.2f}')
elif isinstance(d, dict):
    print(f'  Best reward: {d.get(\"best_reward\", \"?\")}')"

# --- Step 3: Evaluate ---
echo ""
echo "[3/5] Evaluating checkpoint..."
mkdir -p "$EVAL_DIR"
python3 -m training.mujoco.eval_policy \
    --checkpoint "$CHECKPOINT_DIR" \
    --n-steps 2000 \
    --output-dir "$EVAL_DIR" \
    --forward-cmd 0.6 \
    --export-trace

# --- Step 4: Validate ---
echo ""
echo "[4/5] Validating rollout..."
python3 -m training.mujoco.validate_rollout \
    --trace-json "$EVAL_DIR/ainex_trace_forward.json" \
    --min-distance 5.0 \
    --output "$EVAL_DIR/validation.json"

# --- Step 5: Package ---
echo ""
echo "[5/5] Packaging artifacts..."
ARTIFACT_DIR="$HOME/training_output"
tar czf "$HOME/ainex_training_artifacts.tar.gz" \
    -C "$ARTIFACT_DIR" \
    mujoco_walk_v1/ \
    eval/

echo ""
echo "============================================================"
echo "DONE. Artifacts at: $HOME/ainex_training_artifacts.tar.gz"
echo ""
echo "Download with:"
echo "  scp user@VM_IP:~/ainex_training_artifacts.tar.gz ."
echo ""
echo "Then shut down the VM to stop billing:"
echo "  nebius compute instance delete --name <instance-name>"
echo "============================================================"
