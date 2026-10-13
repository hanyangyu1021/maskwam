#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MASKWAM_ROOT="${MASKWAM_ROOT:-$ROOT}"
export MASKWAM_DATA_ROOT="${MASKWAM_DATA_ROOT:-$ROOT/realdata}"
export MASKWAM_PRETRAINED_ROOT="${MASKWAM_PRETRAINED_ROOT:-$ROOT/pretrained_models}"
export MASKWAM_WAN_PATH="${MASKWAM_WAN_PATH:-$MASKWAM_PRETRAINED_ROOT/Wan2.2-TI2V-5B}"
export MASKWAM_CHECKPOINT_ROOT="${MASKWAM_CHECKPOINT_ROOT:-$ROOT/ckpt}"

python3 -m playground.app "$@"
