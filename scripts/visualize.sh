#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VK_DRIVER_FILES="/home/lu/code/REMUS100_HoloOcean/vendor/radeon_icd.json"
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$project_dir"
exec /home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.visualize "$@"
