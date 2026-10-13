#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${CONVERTER:-${SCRIPT_DIR}/convert_xtrainer_to_episode_folders.py}"

if (( $# < 2 )); then
  echo "Usage: $0 <input_root> <output_root> [converter arguments...]" >&2
  exit 2
fi

INPUT_ROOT="$1"
OUTPUT_ROOT="$2"

# Extra arguments after input/output roots are forwarded to the Python converter.
# Example: bash convert_all_context_xtrainer.sh <input_root> <output_root> --skip-existing
EXTRA_ARGS=("${@:3}")

CAMERA_VIEWS=(cam_high cam_left_wrist cam_right_wrist)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONVERTER}" ]]; then
  echo "[error] Converter not found: ${CONVERTER}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "[error] Input root does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[setup] python=${PYTHON_BIN}"
echo "[setup] converter=${CONVERTER}"
echo "[setup] input_root=${INPUT_ROOT}"
echo "[setup] output_root=${OUTPUT_ROOT}"
echo "[setup] camera_views=${CAMERA_VIEWS[*]}"

converted=0
skipped=0

for task_dir in "${INPUT_ROOT}"/*; do
  [[ -d "${task_dir}" ]] || continue

  shopt -s nullglob
  hdf5_files=("${task_dir}"/*.hdf5)
  shopt -u nullglob

  if (( ${#hdf5_files[@]} == 0 )); then
    echo "[skip] no .hdf5 files: ${task_dir}"
    ((skipped += 1))
    continue
  fi

  task_name="$(basename "${task_dir}")"
  output_dir="${OUTPUT_ROOT}/${task_name}"

  echo
  echo "[run] ${task_name}: ${#hdf5_files[@]} hdf5 files"
  echo "      input=${task_dir}"
  echo "      output=${output_dir}"

  "${PYTHON_BIN}" "${CONVERTER}" \
    --input-dir "${task_dir}" \
    --output-dir "${output_dir}" \
    --camera-views "${CAMERA_VIEWS[@]}" \
    "${EXTRA_ARGS[@]}"

  ((converted += 1))
done

echo
echo "[done] converted_tasks=${converted} skipped_tasks=${skipped}"
