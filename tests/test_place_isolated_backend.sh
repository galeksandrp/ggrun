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
