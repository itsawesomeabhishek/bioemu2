#!/usr/bin/env bash
# Local 4-GPU CP=2x2 sampling using release-package inputs.
# Run from repo root:
# bash src/bioemu2/context_parallel/scripts/sample.sh
set -eo pipefail

# Editable local inputs.
SYSTEM_YAML="/path/to/system.yaml"
CHECKPOINT_PATH="/path/to/checkpoint.ckpt"
OUT_DIR="/tmp/cp22-sample"
NUM_SAMPLES=5
BATCH_SIZE=2
PRECISION="bf16-mixed"  # 32-true | bf16-mixed

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

source "$REPO_ROOT/src/bioemu2/context_parallel/scripts/cp_env.sh"

mkdir -p "$OUT_DIR"

uv run torchrun --nproc-per-node=4 --master-port=29562 \
  -m bioemu2.cp predict-structure \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --system "$SYSTEM_YAML" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --precision "$PRECISION" \
  --out "$OUT_DIR"