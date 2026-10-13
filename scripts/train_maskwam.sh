#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MASKWAM_ROOT="${MASKWAM_ROOT:-$ROOT}"
export MASKWAM_DATA_ROOT="${MASKWAM_DATA_ROOT:-$ROOT/realdata}"
export MASKWAM_PRETRAINED_ROOT="${MASKWAM_PRETRAINED_ROOT:-$ROOT/pretrained_models}"
export MASKWAM_WAN_PATH="${MASKWAM_WAN_PATH:-$MASKWAM_PRETRAINED_ROOT/Wan2.2-TI2V-5B}"
export MASKWAM_CHECKPOINT_ROOT="${MASKWAM_CHECKPOINT_ROOT:-$ROOT/ckpt}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PLATFORM="${1:-xtrainer}"
shift || true

case "$PLATFORM" in
  libero|xtrainer) ;;
  *)
    echo "Usage: $0 {libero|xtrainer} [Hydra overrides...]" >&2
    exit 2
    ;;
esac

fail() { echo "[launcher] ERROR: $*" >&2; exit 2; }
PYTHON_BIN="${MASKWAM_PYTHON:-python3}"

NUM_MACHINES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPU_NUM:-1}}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

[[ "$NUM_MACHINES" =~ ^[1-9][0-9]*$ ]] || fail "Invalid machine count: $NUM_MACHINES"
[[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || fail "Invalid GPUs per machine: $GPUS_PER_NODE"
[[ "$MACHINE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || fail "Invalid machine rank: $MACHINE_RANK"
(( MACHINE_RANK < NUM_MACHINES )) || fail "Machine rank $MACHINE_RANK must be less than $NUM_MACHINES"
[[ "$MAIN_PROCESS_PORT" =~ ^[1-9][0-9]*$ ]] || fail "Invalid main process port: $MAIN_PROCESS_PORT"
(( MAIN_PROCESS_PORT <= 65535 )) || fail "Main process port must be <= 65535"

TOTAL_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))
export MASKWAM_EXPECTED_WORLD_SIZE="$TOTAL_PROCESSES"
export MASKWAM_MACHINE_RANK="$MACHINE_RANK"
echo "[launcher] machines=$NUM_MACHINES gpus_per_machine=$GPUS_PER_NODE total_processes=$TOTAL_PROCESSES machine_rank=$MACHINE_RANK master=$MAIN_PROCESS_IP:$MAIN_PROCESS_PORT"

LAUNCH_COMMAND=(
  "$PYTHON_BIN" -m accelerate.commands.launch
  --num_processes "$TOTAL_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_ip "$MAIN_PROCESS_IP"
  --main_process_port "$MAIN_PROCESS_PORT"
  "$ROOT/scripts/train_maskwam.py" "dataset=$PLATFORM" "$@"
)
if (( TOTAL_PROCESSES > 1 )); then
  LAUNCH_COMMAND=("${LAUNCH_COMMAND[@]:0:3}" --multi_gpu "${LAUNCH_COMMAND[@]:3}")
fi
if [[ "${MASKWAM_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${LAUNCH_COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${LAUNCH_COMMAND[@]}"
