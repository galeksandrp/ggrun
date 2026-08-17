#!/usr/bin/env bash
# CUDA host-compiler selection must follow nvcc's GCC cap.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(
    awk '
        /^cuda_max_host_gcc_for\(\)/ {keep=1}
        keep {print}
        keep && /^}/ {exit}
    ' "$ROOT/install.sh"
)"

[[ "$(cuda_max_host_gcc_for 12.3)" == 12 ]]
[[ "$(cuda_max_host_gcc_for 12.4)" == 13 ]]
[[ "$(cuda_max_host_gcc_for 11.8)" == 11 ]]
[[ "$(cuda_max_host_gcc_for 13.2)" == 14 ]]
echo "  ✓ CUDA version maps to the GCC cap nvcc actually enforces"
