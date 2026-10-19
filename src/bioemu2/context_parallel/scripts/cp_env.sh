#!/usr/bin/env bash
# Sourceable env-setup for CP entry points such as sample.sh.
# Usage:    source .../cp_env.sh [NNODES]
#             NNODES  positive int — nodes in the run (default 1)
# Reads:    EMU_CP_VERBOSITY  (default: info)  warning|info|debug|trace
#           EMU_CP_NNODES     (default: 1)     fallback for NNODES if $1 omitted
#           EMU_CP_GPU_COUNT  (optional)       override auto-detected per-node GPU count
#                                              (bypasses nvidia-smi; used by tests)
# Sets:     PYTORCH_CUDA_ALLOC_CONF, HYDRA_FULL_ERROR (setdefault, caller wins)
#           NCCL_DEBUG, TORCH_DISTRIBUTED_DEBUG — must be set here in the shell because
#             libnccl / c10d read them when torchrun bootstraps the process, before
#             init_cp runs. Force-overridden at warning|info|debug so a platform-set
#             NCCL_DEBUG=INFO (Singularity / AzureML) doesn't leak the ~1k-line NCCL
#             topo dump into the "info" log; setdefault only at `trace`, where a user's
#             `NCCL_DEBUG=VERSION`/`TRACE` should still win.
# Derives (skipped if EMU_CP_GRID or CUDA_VISIBLE_DEVICES is already set):
#           EMU_CP_GRID='S,S' where S = sqrt(per_node * NNODES) — errors if not square
#           CUDA_VISIBLE_DEVICES=0,..,per_node-1
#
# EMU_CP_VERBOSITY also sets the bioemu2.context_parallel logger level in Python
# (init_cp -> _apply_cp_verbosity). Just `export EMU_CP_VERBOSITY=debug` before the run.

: "${EMU_CP_VERBOSITY=info}"
case "$(printf '%s' "$EMU_CP_VERBOSITY" | tr '[:upper:]' '[:lower:]')" in
    warning|info|debug)  _cp_nccl=WARN; _cp_tdd=OFF ;;
    trace)               _cp_nccl=INFO; _cp_tdd=DETAIL ;;
    *)
        echo "cp_env.sh: EMU_CP_VERBOSITY='$EMU_CP_VERBOSITY' must be warning|info|debug|trace" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
export EMU_CP_VERBOSITY=$EMU_CP_VERBOSITY
# At trace, respect caller-set NCCL_DEBUG (e.g. VERSION/TRACE for deeper debug);
# at warning|info|debug, force our value so a platform-set NCCL_DEBUG=INFO
# on Singularity/AzureML can't flood the log with the NCCL topo dump.
case "$EMU_CP_VERBOSITY" in
    trace)
        export NCCL_DEBUG="${NCCL_DEBUG:-$_cp_nccl}"
        export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-$_cp_tdd}"
        ;;
    *)
        export NCCL_DEBUG="$_cp_nccl"
        export TORCH_DISTRIBUTED_DEBUG="$_cp_tdd"
        ;;
esac
unset _cp_nccl _cp_tdd

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# Auto-derive EMU_CP_GRID + CUDA_VISIBLE_DEVICES from detected per-node GPU count
# and the requested nnodes. Escape hatch: pre-set either var to skip derivation.
if [ -z "${EMU_CP_GRID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _cp_out=$(EMU_CP_NNODES_ARG="${1:-${EMU_CP_NNODES:-1}}" python3 - <<'PY'
import math, os, shutil, subprocess, sys

def die(msg): sys.exit(f"cp_env.sh: {msg}")

nnodes_s = os.environ["EMU_CP_NNODES_ARG"]
if not nnodes_s.isdigit() or int(nnodes_s) < 1:
    die(f"nnodes={nnodes_s!r} must be a positive int")
nnodes = int(nnodes_s)

override = os.environ.get("EMU_CP_GPU_COUNT")
if override is not None:
    if not override.isdigit() or int(override) < 1:
        die(f"EMU_CP_GPU_COUNT={override!r} must be a positive int")
    per_node = int(override)
elif shutil.which("nvidia-smi"):
    per_node = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines())
    if per_node < 1:
        die("nvidia-smi reported 0 GPUs")
else:
    die("nvidia-smi not found; set EMU_CP_GPU_COUNT to override")

total = per_node * nnodes
side = math.isqrt(total)
if side * side != total:
    die(f"total GPUs {total} = {per_node}/node x {nnodes} nodes is not a perfect square")

devices = ",".join(str(i) for i in range(per_node))
print(f"export EMU_CP_GRID={side},{side}")
print(f"export CUDA_VISIBLE_DEVICES={devices}")
PY
) || return 1 2>/dev/null || exit 1
    eval "$_cp_out"
    unset _cp_out
fi
