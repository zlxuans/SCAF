#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scaf_python="${SCAF_PYTHON:-$PWD/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
scaf_output="runs/stage2-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$scaf_output"
"$scaf_python" -m pytest -q | tee "$scaf_output/pytest.log"
nice -n 10 ionice -c 3 "$scaf_python" -m scaf.data.cli audit --output "$scaf_output/audit" | tee "$scaf_output/audit.log"
for scaf_sequence in tum_fr1_xyz tum_fr2_desk tum_fr3_walking_xyz; do
    nice -n 10 ionice -c 3 "$scaf_python" -m scaf.data.cli preview \
        --sequence "$scaf_sequence" --audit-manifest "$scaf_output/audit/manifest.json" \
        --output "$scaf_output/$scaf_sequence" | tee "$scaf_output/$scaf_sequence.log"
done
printf '%s\n' "$scaf_output"
