#!/usr/bin/env bash
# place_isolated_backend must keep a landed binary even when --version fails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d -t ggrun-place-backend.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# install.sh runs on source; pull only the helpers this test needs.
eval "$(
    awk '
        /^say\(\)/ {print; next}
        /^ok\(\)/ {print; next}
        /^warn\(\)/ {print; next}
        /^place_isolated_backend\(\)/ {keep=1}
        keep {print}
        keep && /^}/ {exit}
    ' "$ROOT/install.sh"
)"

INSTALL_DIR="$TMP/bin"
mkdir -p "$INSTALL_DIR" "$TMP/src"

cat >"$TMP/src/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "missing vulkan icd" >&2
exit 1
EOF
chmod +x "$TMP/src/llama-server"
printf 'dummy' >"$TMP/src/libggml.so"

if ! place_isolated_backend "$TMP/src/llama-server" llama-server-vulkan; then
    echo "  ✗ place_isolated_backend returned failure after --version failed"
    exit 1
fi

test -x "$INSTALL_DIR/backends/llama-server-vulkan/llama-server"
test -e "$INSTALL_DIR/llama-server-vulkan"
test -f "$INSTALL_DIR/backends/llama-server-vulkan/libggml.so"

echo "  ✓ keep vulkan backend when --version fails"

# Versioned bundle libs need a SONAME symlink.
printf 'dummy' >"$TMP/src/libllama-common.so.0.0.1"
if ! place_isolated_backend "$TMP/src/llama-server" llama-server-cpu; then
    echo "  ✗ place failed after adding versioned library"
    exit 1
fi
test -L "$INSTALL_DIR/backends/llama-server-cpu/libllama-common.so.0"
test -f "$INSTALL_DIR/backends/llama-server-cpu/libllama-common.so.0.0.1"
echo "  ✓ create SONAME symlink for libllama-common.so.0"

# Missing project libraries are a failed place, not "GPU runtime missing".
cat >"$TMP/src/llama-server" <<'EOF'
#!/usr/bin/env bash
echo "/tmp/llama-server: error while loading shared libraries: libllama-common.so.0: cannot open shared object file: No such file or directory" >&2
exit 127
EOF
chmod +x "$TMP/src/llama-server"
rm -f "$TMP/src"/lib*.so*
if place_isolated_backend "$TMP/src/llama-server" llama-server-broken; then
    echo "  ✗ place succeeded for a binary missing libllama"
    exit 1
fi
test ! -e "$INSTALL_DIR/llama-server-broken"
echo "  ✓ reject a bundle that cannot load its own libllama"

