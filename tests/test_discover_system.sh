#!/usr/bin/env bash
# System scan must find ik_llama.cpp, llama.cpp, forks, and CUDA outside PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t ggrun-discover.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

HOME="$TMP/home"
export HOME
mkdir -p "$HOME/ik_llama.cpp/build/bin" \
         "$HOME/llama.cpp/build-vulkan/bin" \
         "$HOME/src/fork-llama.cpp-hy3/build-cuda/bin" \
         "$HOME/cuda/bin" \
         "$TMP/bin"

cat >"$HOME/ik_llama.cpp/build/bin/llama-server" <<'EOF'
#!/usr/bin/env bash
echo ik
EOF
cat >"$HOME/llama.cpp/build-vulkan/bin/llama-server" <<'EOF'
#!/usr/bin/env bash
echo vulkan
EOF
cat >"$HOME/src/fork-llama.cpp-hy3/build-cuda/bin/llama-server" <<'EOF'
#!/usr/bin/env bash
echo fork
EOF
cat >"$HOME/cuda/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
echo "Cuda compilation tools, release 12.0"
EOF
chmod +x "$HOME/ik_llama.cpp/build/bin/llama-server" \
         "$HOME/llama.cpp/build-vulkan/bin/llama-server" \
         "$HOME/src/fork-llama.cpp-hy3/build-cuda/bin/llama-server" \
         "$HOME/cuda/bin/nvcc"

# Hide the real machine's tools so the test only sees the fake tree.
PATH="$TMP/bin:/usr/bin:/bin"
export PATH
export CUDA_HOME="$HOME/cuda"
unset CUDA_PATH CUDACXX

out="$(
    LLM_INSTALL_NONINTERACTIVE=1 \
    LLM_INSTALL_SCAN_ROOTS="$HOME" \
    "$ROOT/install.sh" --discover
)"

printf '%s\n' "$out" | grep -q "ik_llama=$HOME/ik_llama.cpp/build/bin/llama-server" \
    || { echo "  ✗ did not find ik_llama.cpp"; printf '%s\n' "$out" | sed 's/^/    /'; exit 1; }
printf '%s\n' "$out" | grep -q "llama_vulkan=$HOME/llama.cpp/build-vulkan/bin/llama-server" \
    || { echo "  ✗ did not find vulkan llama.cpp"; printf '%s\n' "$out" | sed 's/^/    /'; exit 1; }
printf '%s\n' "$out" | grep -q "fork=$HOME/src/fork-llama.cpp-hy3/build-cuda/bin/llama-server" \
    || { echo "  ✗ did not find fork"; printf '%s\n' "$out" | sed 's/^/    /'; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
    printf '%s\n' "$out" | grep -q "nvcc=$HOME/cuda/bin/nvcc" \
        || { echo "  ✗ did not find CUDA toolkit outside PATH"; printf '%s\n' "$out" | sed 's/^/    /'; exit 1; }
fi

echo "  ✓ discover finds ik_llama.cpp, llama.cpp, forks, and CUDA"

# Adopt those hits into an empty prefix without downloading.
INSTALL_DIR="$TMP/prefix"
mkdir -p "$INSTALL_DIR"
eval "$(
    awk '
        /^say\(\)|^ok\(\)|^warn\(\)/ {print; next}
        /^_discover_timeout\(\)|^_discover_abspath\(\)|^_discover_skip_path\(\)|^_discover_locate\(\)/ {keep=1}
        /^find_nvidia_smi\(\)|^cuda_nvcc_path\(\)/ {keep=1}
        /^FOUND_NVIDIA_SMI=/ {keep=1}
        /^_path_score\(\)|^_better_server\(\)|^classify_llama_bin\(\)|^_collect_llama_bins\(\)/ {keep=1}
        /^scan_system_installs\(\)|^link_existing_backend\(\)|^adopt_system_backends\(\)|^link_default_llama_server\(\)/ {keep=1}
        keep {print}
        keep && /^}/ {keep=0}
    ' "$ROOT/install.sh"
)"
scan_system_installs
adopt_system_backends || true
test -L "$INSTALL_DIR/ik_llama-server-cuda"
test -L "$INSTALL_DIR/llama-server-vulkan"
echo "  ✓ adopt links existing ik_llama.cpp and llama.cpp"
