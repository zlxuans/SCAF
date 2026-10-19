#!/usr/bin/env bash
# Usage: bash scripts/snapshot.sh ABSOLUTE_DESTINATION [COMMIT]
set -euo pipefail
cd "$(dirname "$0")/.."
destination="${1:?provide a new absolute path outside the source checkout}"
revision="${2:-HEAD}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo 'Commit or resolve source changes before creating a run snapshot.' >&2
    exit 1
fi
case "$destination" in /*) ;; *) echo 'Destination must be absolute.' >&2; exit 1;; esac
python_for_paths="${SCAF_PYTHON:-$PWD/.venv/bin/python}"
"$python_for_paths" -c 'from pathlib import Path; import sys; root=Path.cwd().resolve(); dest=Path(sys.argv[1]).resolve(); sys.exit("Destination must be outside the checkout" if dest == root or root in dest.parents else 0)' "$destination"
git worktree add --detach "$destination" "$revision"
