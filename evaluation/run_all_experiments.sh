#!/usr/bin/env bash
#
# RPG2Robot: Full Experiment Pipeline
#
# Master script that runs all phases of the RPG2Robot experiment:
#   Phase 1: Data Collection (RPG trajectories + MuJoCo rollouts)
#   Phase 2: Training (MuJoCo policies)
#   Phase 3: Evaluation (benchmarks, bridge tests)
#   Phase 4: Export & Upload (HuggingFace packaging)
#
# Prerequisites:
#   - NVIDIA GPU with CUDA (for MuJoCo MJX / JAX)
#   - PostgreSQL running (for trajectory logger)
#   - Python environment with: jax, brax, mujoco, psycopg2-binary
#   - Node.js / Bun (for Hyperscape server)
#   - huggingface_hub (for upload)
#
# Usage:
#   # Run all phases
#   ./run_all_experiments.sh
#
#   # Run specific phase
#   ./run_all_experiments.sh --phase 1
#   ./run_all_experiments.sh --phase 2
#   ./run_all_experiments.sh --phase 3
#   ./run_all_experiments.sh --phase 4
#
#   # Dry run (show what would execute)
#   ./run_all_experiments.sh --dry-run
#
#   # With custom settings
#   ./run_all_experiments.sh --mujoco-episodes 2000 --rpg-episodes 5000
#
# Estimated times (A100 GPU):
#   Phase 1a (MuJoCo rollouts):  ~30 min (10k episodes, 4 tasks)
#   Phase 1b (RPG trajectories): ~8-24 hours (10k episodes, depends on LLM speed)
#   Phase 2  (Training):         ~2 hours (4 policies, 100M steps each)
#   Phase 3  (Evaluation):       ~30 min
#   Phase 4  (Export/Upload):    ~15 min
#   Total:                       ~3-25 hours (Phase 1b dominates)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AINEX_CODE="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="$(cd "${AINEX_CODE}/.." && pwd)"
HYPERSCAPE_DIR="${WORKSPACE}/hyperscape"
HUGGINGFACE_DIR="${WORKSPACE}/huggingface"

# Output directories
ROLLOUT_DIR="${WORKSPACE}/rollouts"
TRAJECTORY_DIR="${WORKSPACE}/trajectories"
CHECKPOINT_BASE="${AINEX_CODE}/checkpoints"
EVAL_DIR="${AINEX_CODE}/evaluation"
LOG_DIR="${WORKSPACE}/experiment_logs"

# Default episode counts
MUJOCO_EPISODES="${MUJOCO_EPISODES:-1000}"
RPG_EPISODES="${RPG_EPISODES:-10000}"
TRAINING_STEPS="${TRAINING_STEPS:-100000000}"
NUM_ENVS="${NUM_ENVS:-4096}"
EVAL_EPISODES="${EVAL_EPISODES:-500}"

# Server URLs
GAME_URL="${GAME_URL:-http://localhost:5555}"
ELIZA_URL="${ELIZA_URL:-http://localhost:4001}"
DB_URL="${DATABASE_URL:-postgresql://localhost:5432/hyperscape}"

# Phase control
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-false}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

log_phase() {
    echo ""
    echo -e "${BOLD}${BLUE}================================================================${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}================================================================${NC}"
    echo ""
}

log_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_cmd() {
    local desc="$1"
    shift
    log_step "$desc"
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "  ${YELLOW}[DRY RUN]${NC} $*"
        return 0
    fi
    echo "  > $*"
    "$@" 2>&1 | tee -a "${LOG_DIR}/experiment.log"
    local status=${PIPESTATUS[0]}
    if [ $status -ne 0 ]; then
        log_error "Command failed with exit code $status: $*"
        return $status
    fi
    return 0
}

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    local diff=$((now - start))
    printf '%dh %dm %ds' $((diff/3600)) $((diff%3600/60)) $((diff%60))
}

check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    else
        log_warn "No NVIDIA GPU detected. MuJoCo MJX training will be slow on CPU."
    fi
}

check_python_deps() {
    local missing=()
    for pkg in jax mujoco brax numpy; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing Python packages: ${missing[*]}"
        log_error "Install with: pip install ${missing[*]}"
        return 1
    fi
    log_info "Python dependencies OK"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --mujoco-episodes)
            MUJOCO_EPISODES="$2"
            shift 2
            ;;
        --rpg-episodes)
            RPG_EPISODES="$2"
            shift 2
            ;;
        --training-steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase NUM          Run specific phase (1-4) or 'all' (default: all)"
            echo "  --dry-run            Show commands without executing"
            echo "  --mujoco-episodes N  MuJoCo rollout episodes per task (default: 1000)"
            echo "  --rpg-episodes N     RPG trajectory episodes (default: 10000)"
            echo "  --training-steps N   PPO training timesteps (default: 100000000)"
            echo "  --num-envs N         Parallel environments for training (default: 4096)"
            echo "  --eval-episodes N    Evaluation episodes per task (default: 500)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR" "$ROLLOUT_DIR" "$TRAJECTORY_DIR"

EXPERIMENT_START=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "" | tee "${LOG_DIR}/experiment.log"
log_phase "RPG2Robot Full Experiment Pipeline"
log_info "Timestamp:       $TIMESTAMP"
log_info "Phase:           $PHASE"
log_info "Dry run:         $DRY_RUN"
log_info "Workspace:       $WORKSPACE"
log_info "MuJoCo episodes: $MUJOCO_EPISODES per task"
log_info "RPG episodes:    $RPG_EPISODES"
log_info "Training steps:  $TRAINING_STEPS"
log_info "Parallel envs:   $NUM_ENVS"
echo ""

check_gpu
check_python_deps || exit 1

# ---------------------------------------------------------------------------
# Phase 1: Data Collection
# ---------------------------------------------------------------------------

if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
    PHASE1_START=$(date +%s)
    log_phase "Phase 1: Data Collection"

    # ---- Phase 1a: MuJoCo rollouts ----
    log_phase "Phase 1a: Collect MuJoCo Control Trajectories"
    log_info "Collecting ${MUJOCO_EPISODES} episodes for each task"
    log_info "Tasks: locomotion, target_reaching, wave, compositional"
    log_info "Estimated time: ~30 min (A100 GPU)"

    # Locomotion rollouts
    run_cmd "Collect locomotion rollouts" \
        python3 -m evaluation.collect_mujoco_rollouts \
            --task locomotion \
            --episodes "$MUJOCO_EPISODES" \
            --checkpoint mujoco_locomotion_v13_flat_feet \
            --output "$ROLLOUT_DIR" \
            --seed 42

    # Target reaching rollouts
    run_cmd "Collect target reaching rollouts" \
        python3 -m evaluation.collect_mujoco_rollouts \
            --task target_reaching \
            --episodes "$MUJOCO_EPISODES" \
            --checkpoint walk_to_target \
            --output "$ROLLOUT_DIR" \
            --seed 42

    # Wave rollouts
    run_cmd "Collect wave rollouts" \
        python3 -m evaluation.collect_mujoco_rollouts \
            --task wave \
            --episodes "$MUJOCO_EPISODES" \
            --checkpoint mujoco_wave_v2 \
            --output "$ROLLOUT_DIR" \
            --seed 42

    # Compositional rollouts
    run_cmd "Collect compositional rollouts" \
        python3 -m evaluation.collect_mujoco_rollouts \
            --task compositional \
            --episodes "$MUJOCO_EPISODES" \
            --checkpoint multi_task \
            --output "$ROLLOUT_DIR" \
            --seed 42

    # Convert to LeRobot and RLDS formats
    run_cmd "Convert rollouts to LeRobot format" \
        python3 -m evaluation.collect_mujoco_rollouts \
            --convert-lerobot \
            --output "$ROLLOUT_DIR"

    run_cmd "Convert rollouts to RLDS format" \
        python3 -m evaluation.collect_mujoco_rollouts \
            --convert-rlds \
            --output "$ROLLOUT_DIR"

    # ---- Phase 1b: RPG trajectories ----
    log_phase "Phase 1b: Collect RPG Agent Trajectories"
    log_info "Collecting ${RPG_EPISODES} episodes across 11 goal types"
    log_info "Requires: Hyperscape server (bun run dev:ai) + PostgreSQL"
    log_info "Estimated time: 8-24 hours (depends on LLM API speed)"
    log_info ""
    log_info "NOTE: Start Hyperscape first in a separate terminal:"
    log_info "  cd ${HYPERSCAPE_DIR} && bun run dev:ai"
    log_info ""

    run_cmd "Collect RPG trajectories" \
        python3 -m evaluation.collect_rpg_trajectories \
            --episodes "$RPG_EPISODES" \
            --output "$TRAJECTORY_DIR" \
            --db-url "$DB_URL" \
            --game-url "$GAME_URL" \
            --eliza-url "$ELIZA_URL"

    log_info "Phase 1 complete. Elapsed: $(elapsed_since $PHASE1_START)"
fi

# ---------------------------------------------------------------------------
# Phase 2: Training
# ---------------------------------------------------------------------------

if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ]; then
    PHASE2_START=$(date +%s)
    log_phase "Phase 2: Policy Training"
    log_info "Training PPO policies in MuJoCo Playground"
    log_info "Steps: ${TRAINING_STEPS}, Envs: ${NUM_ENVS}"
    log_info "Estimated time: ~2 hours total (4 policies x ~30 min each)"

    # Train locomotion policy (joystick velocity tracking)
    run_cmd "Train locomotion policy (v13 with domain rand + flat feet)" \
        python3 -m training.mujoco.train \
            --num-timesteps "$TRAINING_STEPS" \
            --num-envs "$NUM_ENVS" \
            --checkpoint-dir "${CHECKPOINT_BASE}/mujoco_locomotion_v13_flat_feet" \
            --seed 0

    # Train target reaching policy
    run_cmd "Train target reaching policy" \
        python3 -m training.mujoco.train \
            --target \
            --num-timesteps "$TRAINING_STEPS" \
            --num-envs "$NUM_ENVS" \
            --checkpoint-dir "${CHECKPOINT_BASE}/walk_to_target" \
            --seed 0

    # Train wave policy (requires frozen walking policy from above)
    run_cmd "Train wave-while-walking policy" \
        python3 -m training.mujoco.train_upper \
            --task wave \
            --num-timesteps "$((TRAINING_STEPS / 2))" \
            --num-envs "$NUM_ENVS" \
            --checkpoint-dir "${CHECKPOINT_BASE}/mujoco_wave_v2" \
            --walking-checkpoint "${CHECKPOINT_BASE}/mujoco_locomotion_v13_flat_feet" \
            --seed 0

    # Train bridge/multi-task policy
    run_cmd "Train multi-task bridge policy" \
        python3 -m training.train_bridge_policy \
            --num-timesteps "$((TRAINING_STEPS / 2))" \
            --checkpoint-dir "${CHECKPOINT_BASE}/multi_task" \
            --seed 0

    log_info "Phase 2 complete. Elapsed: $(elapsed_since $PHASE2_START)"
fi

# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------

if [ "$PHASE" = "all" ] || [ "$PHASE" = "3" ]; then
    PHASE3_START=$(date +%s)
    log_phase "Phase 3: Evaluation"
    log_info "Running evaluation benchmarks"
    log_info "Estimated time: ~30 min"

    # Evaluate locomotion policy
    run_cmd "Evaluate locomotion policy" \
        python3 -m training.mujoco.eval_policy \
            --checkpoint "${CHECKPOINT_BASE}/mujoco_locomotion_v13_flat_feet" \
            --output-dir "${AINEX_CODE}/training/videos" \
            --n-steps 500

    # Evaluate target reaching policy
    run_cmd "Evaluate target reaching policy" \
        python3 -m training.mujoco.eval_policy \
            --checkpoint "${CHECKPOINT_BASE}/walk_to_target" \
            --output-dir "${AINEX_CODE}/training/videos" \
            --n-steps 500

    # Validate rollout quality
    run_cmd "Validate locomotion rollouts" \
        python3 -m training.mujoco.validate_rollout \
            --checkpoint "${CHECKPOINT_BASE}/mujoco_locomotion_v13_flat_feet"

    # Run evaluation suite (if available)
    if python3 -c "from evaluation import Evaluator" 2>/dev/null; then
        run_cmd "Run full evaluation suite" \
            python3 -m evaluation.run_eval \
                --planner rpg2robot \
                --tasks all \
                --episodes "$EVAL_EPISODES"
    else
        log_warn "Evaluation suite not fully configured, skipping full eval"
    fi

    log_info "Phase 3 complete. Elapsed: $(elapsed_since $PHASE3_START)"
fi

# ---------------------------------------------------------------------------
# Phase 4: Export & Upload
# ---------------------------------------------------------------------------

if [ "$PHASE" = "all" ] || [ "$PHASE" = "4" ]; then
    PHASE4_START=$(date +%s)
    log_phase "Phase 4: Export & Upload to HuggingFace"
    log_info "Packaging artifacts for HuggingFace Hub"
    log_info "Estimated time: ~15 min"

    # Verify data exists
    log_step "Checking data artifacts..."
    ARTIFACTS_READY=true

    if [ ! -d "$ROLLOUT_DIR" ] || [ -z "$(ls -A "$ROLLOUT_DIR" 2>/dev/null)" ]; then
        log_warn "No MuJoCo rollouts found at $ROLLOUT_DIR"
        ARTIFACTS_READY=false
    else
        ROLLOUT_COUNT=$(find "$ROLLOUT_DIR" -name "*.jsonl" | wc -l)
        log_info "Found $ROLLOUT_COUNT rollout files"
    fi

    if [ ! -d "$TRAJECTORY_DIR" ] || [ -z "$(ls -A "$TRAJECTORY_DIR" 2>/dev/null)" ]; then
        log_warn "No RPG trajectories found at $TRAJECTORY_DIR"
        ARTIFACTS_READY=false
    else
        TRAJ_COUNT=$(find "$TRAJECTORY_DIR" -name "*.jsonl" | wc -l)
        log_info "Found $TRAJ_COUNT trajectory files"
    fi

    CKPT_COUNT=0
    for ckpt in mujoco_locomotion_v13_flat_feet walk_to_target mujoco_wave_v2 multi_task; do
        if [ -f "${CHECKPOINT_BASE}/${ckpt}/final_params" ]; then
            CKPT_COUNT=$((CKPT_COUNT + 1))
        fi
    done
    log_info "Found $CKPT_COUNT/4 policy checkpoints"

    if [ "$ARTIFACTS_READY" = "false" ]; then
        log_warn "Some artifacts are missing. Upload will skip missing items."
    fi

    # Dry run first to show what would be uploaded
    run_cmd "Preview upload (dry run)" \
        python3 "${HUGGINGFACE_DIR}/upload.py" \
            --artifact all \
            --dry-run

    # Actual upload (requires HF_TOKEN)
    if [ -n "${HF_TOKEN:-}" ]; then
        run_cmd "Upload all artifacts to HuggingFace" \
            python3 "${HUGGINGFACE_DIR}/upload.py" \
                --artifact all \
                --token "$HF_TOKEN"
    else
        log_warn "HF_TOKEN not set. Skipping actual upload."
        log_info "To upload, set HF_TOKEN and re-run Phase 4:"
        log_info "  HF_TOKEN=hf_xxx ./run_all_experiments.sh --phase 4"
    fi

    log_info "Phase 4 complete. Elapsed: $(elapsed_since $PHASE4_START)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log_phase "Experiment Complete"
log_info "Total elapsed: $(elapsed_since $EXPERIMENT_START)"
log_info ""
log_info "Output locations:"
log_info "  MuJoCo rollouts:    $ROLLOUT_DIR"
log_info "  RPG trajectories:   $TRAJECTORY_DIR"
log_info "  Policy checkpoints: $CHECKPOINT_BASE"
log_info "  Evaluation videos:  ${AINEX_CODE}/training/videos"
log_info "  Experiment logs:    $LOG_DIR"
log_info ""
log_info "HuggingFace repos (after upload):"
log_info "  Model:        https://huggingface.co/rpg2robot/ainex-mujoco-model"
log_info "  Policies:     https://huggingface.co/rpg2robot/rpg2robot-policies"
log_info "  Trajectories: https://huggingface.co/datasets/rpg2robot/hyperscape-trajectories"
log_info "  Control:      https://huggingface.co/datasets/rpg2robot/ainex-control-trajectories"
