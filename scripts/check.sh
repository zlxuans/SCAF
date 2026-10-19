#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scaf_python="${SCAF_PYTHON:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$scaf_python" -m pytest -q
"$scaf_python" -m scaf.cli smoke --device cpu
