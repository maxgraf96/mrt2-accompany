#!/usr/bin/env bash
# Apply the required local patch to the sibling magenta-realtime checkout.
#
# WHY: the released spectrostream_encoder.mlxfn is traced at 28 s (1,344,000
# samples), but this magenta-realtime checkout's core/src/mlx_engine.cpp hardcodes
# kEncoderInputSamples = 2,880,000 (60 s). Without this patch every prefill_state
# call throws a shape-mismatch and all MRT2-Accompany prefill is broken. The 60 s
# re-export is impossible (raw mrt2_base.safetensors checkpoint is not publicly
# fetchable). 28 s >> the model's ~19.7 s receptive field, so quality is unaffected.
#
# Idempotent: skips if the patch is already applied. Run after cloning/resetting
# magenta-realtime. See PATCHES.md.
set -euo pipefail
MAGENTA_RT_DIR="${MAGENTA_RT_DIR:-$HOME/Code/magenta-realtime}"
PATCH="$(cd "$(dirname "$0")/.." && pwd)/patches/magenta-encoder-28s.patch"
TARGET="$MAGENTA_RT_DIR/core/src/mlx_engine.cpp"

[ -f "$TARGET" ] || { echo "error: $TARGET not found (set MAGENTA_RT_DIR)"; exit 1; }

if grep -q "mrt2-accompany local patch" "$TARGET"; then
    echo "already patched: $TARGET"; exit 0
fi
if git -C "$MAGENTA_RT_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$MAGENTA_RT_DIR" apply "$PATCH"
    echo "applied $PATCH -> $TARGET"
else
    echo "error: patch does not apply cleanly (magenta-realtime may have changed)."
    echo "       Manually set kEncoderInputSamples = 1344000 at both sites in:"
    echo "       $TARGET"
    exit 1
fi
