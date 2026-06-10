#!/usr/bin/env bash
# Apply the required local patches to the sibling magenta-realtime checkout.
# Idempotent per patch: each is skipped if its marker is already present.
# Run after cloning/resetting magenta-realtime. See PATCHES.md for the WHY of
# each patch.
set -euo pipefail
MAGENTA_RT_DIR="${MAGENTA_RT_DIR:-$HOME/Code/magenta-realtime}"
PATCH_DIR="$(cd "$(dirname "$0")/.." && pwd)/patches"
TARGET="$MAGENTA_RT_DIR/core/src/mlx_engine.cpp"

[ -f "$TARGET" ] || { echo "error: $TARGET not found (set MAGENTA_RT_DIR)"; exit 1; }

apply_one() {  # <patch-file> <already-applied-marker> <manual-hint>
    local patch="$PATCH_DIR/$1" marker="$2" hint="$3"
    if grep -q "$marker" "$TARGET"; then
        echo "already applied: $1"; return 0
    fi
    if git -C "$MAGENTA_RT_DIR" apply --check "$patch" 2>/dev/null; then
        git -C "$MAGENTA_RT_DIR" apply "$patch"
        echo "applied $1 -> $TARGET"
    else
        echo "error: $1 does not apply cleanly (magenta-realtime may have changed)."
        echo "       $hint"
        exit 1
    fi
}

# 1) Encoder length: released spectrostream_encoder.mlxfn is traced at 28 s
#    (1,344,000 samples); the code hardcodes 60 s and breaks all prefill.
apply_one magenta-encoder-28s.patch "1344000" \
    "Manually set kEncoderInputSamples = 1344000 at both sites in $TARGET"

# 2) MusicCoCa worker thread lifetime: was detached with a raw `this` capture;
#    destroying the engine mid-encode (host removes the plugin) crashed the
#    host (destroyed mutex / freed TFLite interpreter use). Join in ~Impl.
apply_one magenta-musiccoca-thread-join.patch "musiccoca_thread_" \
    "Make the MusicCoCa worker joinable and join it in ~Impl before unload()."
