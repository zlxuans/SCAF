#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scaf_python="${SCAF_PYTHON:-$PWD/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
scaf_audit="${SCAF_AUDIT_MANIFEST:?Set the completed stage-2 audit manifest path}"
scaf_geometry="${SCAF_GEOMETRY_CHECKPOINT:?Set the verified stage-4 geometry-final.pt path}"
scaf_assets="${SCAF_ASSET_MANIFEST:-$PWD/weights/baselines/manifest.json}"
export SCAF_BASELINE_ASSET_MANIFEST="$scaf_assets"
scaf_data="${SCAF_DATA_ROOT:-$PWD/data}"
scaf_output="${SCAF_STAGE5_OUTPUT:-$PWD/runs/stage5-$(date -u +%Y%m%dT%H%M%S)-$$}"
mkdir -p "$(dirname "$scaf_output")"
mkdir "$scaf_output"
"$scaf_python" -m pytest -q | tee "$scaf_output/pytest.log"
nice -n 10 ionice -c 3 "$scaf_python" -m scaf.stage5 \
    --data-root "$scaf_data" --audit-manifest "$scaf_audit" --asset-manifest "$scaf_assets" \
    --geometry-checkpoint "$scaf_geometry" --output "$scaf_output/experiment" | tee "$scaf_output/experiment.log"
"$scaf_python" scripts/verify_stage5.py "$scaf_output/experiment" --output "$scaf_output/artifact-verification.json"
printf '%s\n' "$scaf_output"
