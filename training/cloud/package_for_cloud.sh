#!/usr/bin/env bash
# Package the minimal files needed for cloud training.
# Run from ainex-robot-code/
set -euo pipefail

OUT="$HOME/ainex_cloud_upload.tar.gz"

tar czf "$OUT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='checkpoints' \
    --exclude='end_to_end_outputs' \
    --exclude='training/videos' \
    --exclude='training/rl' \
    --exclude='training/runtime' \
    --exclude='training/models' \
    --exclude='training/datasets' \
    --exclude='training/eval' \
    --exclude='training/configs' \
    --exclude='training/tests' \
    --exclude='.git' \
    training/mujoco/ \
    training/schema/ \
    training/__init__.py \
    training/train_bridge_policy.py \
    training/cloud/ \
    perception/entity_slots/

SIZE=$(du -h "$OUT" | cut -f1)
echo "Packaged: $OUT ($SIZE)"
echo ""
echo "Upload to VM:"
echo "  scp $OUT user@VM_IP:~/"
echo "  ssh user@VM_IP 'mkdir -p ainex-robot-code && cd ainex-robot-code && tar xzf ~/ainex_cloud_upload.tar.gz'"
echo "  ssh user@VM_IP 'cd ainex-robot-code && bash training/cloud/train_on_nebius.sh'"
