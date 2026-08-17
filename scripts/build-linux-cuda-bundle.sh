#!/usr/bin/env bash
# Build a prebuilt Linux x86_64 CUDA release bundle (ik_llama.cpp) so that
# `install.sh` serves a CUDA GPU backend with NO toolchain on users' machines.
#
# Upstream llama.cpp publishes CUDA prebuilts for Windows only, and ik_llama.cpp
# (ggrun's fast CUDA backend) ships no binaries at all — so without this bundle a
# Linux NVIDIA user either needs a full CUDA toolkit to compile, or falls back to
# the slower prebuilt Vulkan backend. The release workflow runs this script in a
# CUDA development container. It can also be run manually when a new pinned
# backend revision needs testing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IK_REPO="${IK_LLAMA_REPO:-https://github.com/ikawrakow/ik_llama.cpp.git}"
IK_REF="${IK_LLAMA_REF:-1fddd12ba861c4815a8633f14d9c5670692099cc}"
IK_DIR="${IK_LLAMA_DIR:-$ROOT_DIR/.ik_llama.cpp}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
ASSET="ggrun-linux-x86_64-cuda.tar.gz"

for tool in nvcc cmake go git; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Error: '$tool' not found on PATH" >&2; exit 1; }
done
# ik_llama's ggml target asks for CUDA20. Ubuntu 22.04's CMake 3.22 cannot.
cmake_ver="$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)"
cmake_maj="${cmake_ver%%.*}"
cmake_min="${cmake_ver#*.}"
if [ "${cmake_maj:-0}" -lt 3 ] || { [ "${cmake_maj}" -eq 3 ] && [ "${cmake_min%%.*}" -lt 25 ]; }; then
    echo "Error: CMake >= 3.25 is required (have ${cmake_ver:-unknown})" >&2
    exit 1
fi

# 1. Go launcher — package-release.sh bundles $ROOT_DIR/go/ggrun when present.
echo "==> Building ggrun (Go launcher)"
version="${GITHUB_REF_NAME:-}"
ldflags="-s -w"
case "$version" in
    v*) ldflags="$ldflags -X github.com/raketenkater/ggrun/pkg/update.currentVersion=$version" ;;
    *) ;;
esac
( cd "$ROOT_DIR/go" && go build -buildvcs=false -trimpath -ldflags="$ldflags" -o ggrun ./cmd/ggrun )

# 2. ik_llama.cpp CUDA llama-server.
if [[ -d "$IK_DIR/.git" ]]; then
    echo "==> Updating ik_llama.cpp ($IK_REF)"
    git -C "$IK_DIR" fetch --depth 1 origin "$IK_REF"
    git -C "$IK_DIR" checkout FETCH_HEAD
else
    echo "==> Cloning ik_llama.cpp ($IK_REF)"
    git init "$IK_DIR"
    git -C "$IK_DIR" remote add origin "$IK_REPO"
    git -C "$IK_DIR" fetch --depth 1 origin "$IK_REF"
    git -C "$IK_DIR" checkout --detach FETCH_HEAD
fi
echo "==> Configuring + building llama-server (CUDA)"
# Devel images have nvcc but no driver. The stub satisfies link-time
# libcuda.so.1; the real driver is loaded on the user's machine.
stub_dir=""
for d in /usr/local/cuda/lib64/stubs /usr/local/cuda/targets/x86_64-linux/lib/stubs; do
    [[ -e "$d/libcuda.so" ]] && stub_dir="$d" && break
done
cmake_link=()
if [[ -n "$stub_dir" ]]; then
    [[ -e "$stub_dir/libcuda.so.1" ]] || ln -sfn libcuda.so "$stub_dir/libcuda.so.1"
    export LIBRARY_PATH="$stub_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
    cmake_link=(-DCMAKE_EXE_LINKER_FLAGS="-L${stub_dir}" -DCMAKE_SHARED_LINKER_FLAGS="-L${stub_dir}")
    echo "==> Linking against CUDA driver stub in $stub_dir"
fi
cmake -S "$IK_DIR" -B "$IK_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89" \
    "${cmake_link[@]}"
cmake --build "$IK_DIR/build" --config Release -j"$(nproc 2>/dev/null || echo 4)" -t llama-server

SERVER="$IK_DIR/build/bin/llama-server"
[[ -x "$SERVER" ]] || { echo "Error: build did not produce $SERVER" >&2; exit 1; }

# 3. Package the relocatable bundle and record its checksum.
echo "==> Packaging $ASSET"
mkdir -p "$OUT_DIR"
"$ROOT_DIR/scripts/package-release.sh" "$ASSET" "$SERVER" "$OUT_DIR"
( cd "$OUT_DIR" && sha256sum "$ASSET" >> SHA256SUMS )

echo
echo "Built $OUT_DIR/$ASSET"
