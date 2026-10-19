#!/usr/bin/env bash
# Start ONLY the tiny infrastructure smoke task. No training is launched.
set -euo pipefail
cd "$(dirname "$0")/.."
scaf_python="${SCAF_PYTHON:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log_root="${SCAF_JOB_LOG_ROOT:-$PWD/runs/launcher}"
mkdir -p "$log_root"
job_dir="$(mktemp -d "$log_root/job_XXXXXXXX")"
nohup "$scaf_python" -u -m scaf.cli smoke "$@" > "$job_dir/console.log" 2>&1 < /dev/null &
echo "$!" > "$job_dir/pid"
echo "$job_dir"
