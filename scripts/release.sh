#!/usr/bin/env bash
# release.sh — build, Developer-ID-sign, notarize, staple, and package the
# VST3 + Standalone app into a distributable DMG.
#
# Usage:
#   scripts/release.sh [--no-notarize] [--skip-build]
#
# Credentials are read from `.env` at the repo root if present, otherwise
# from the SA3 repo's plugin/.env (same Apple account for both projects).
# Required keys:
#   APPLE_ID="you@example.com"
#   APPLE_TEAM_ID="ABCDE12345"
#   APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"   (appleid.apple.com app password)
#   CODESIGN_IDENTITY="Developer ID Application: Your Name (ABCDE12345)"
#     (SA3_CODESIGN_IDENTITY is accepted as an alias)
#
# Mirrors SA3-Variations' release workflow (stable-audio-3/plugin/scripts),
# minus the dylib vendoring: MRT2-Accompany links MLX / TFLite /
# SentencePiece statically, so the only non-code payload is mlx.metallib.
# Ship a DMG, not a ZIP: the metallib is not Mach-O, its code signature
# lives in extended attributes, and most unzip tools strip xattrs —
# breaking Gatekeeper on the user's machine. A DMG preserves them.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# ── credentials ──────────────────────────────────────────────────────────────
ENV_FILE=""
for f in "$REPO/.env" "$HOME/Code/stable-audio-3/plugin/.env"; do
    if [[ -f "$f" ]]; then ENV_FILE="$f"; break; fi
done
if [[ -n "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    echo "==> credentials: $ENV_FILE"
fi
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:-${SA3_CODESIGN_IDENTITY:-}}"

if [[ -z "$CODESIGN_IDENTITY" || "$CODESIGN_IDENTITY" == "-" ]]; then
    echo "error: CODESIGN_IDENTITY not set — a Developer ID Application identity" >&2
    echo "       is required for a distributable build (see header of this script)" >&2
    exit 1
fi

notarize=1 build=1
for arg in "$@"; do
    case "$arg" in
        --no-notarize) notarize=0 ;;
        --skip-build)  build=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done
for v in APPLE_ID APPLE_TEAM_ID APPLE_APP_PASSWORD; do
    if [[ -z "${!v:-}" ]] || [[ "${!v}" == "FILL_ME_IN"* ]] \
                          || [[ "${!v}" == "xxxx-"* ]]; then
        notarize=0
    fi
done

# ── build ────────────────────────────────────────────────────────────────────
if [[ "$build" == "1" ]]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target MRT2Accompany_VST3 MRT2Accompany_Standalone \
        -j "$(sysctl -n hw.ncpu)"
fi

ART="$REPO/build/MRT2Accompany_artefacts/Release"
APP="$ART/Standalone/MRT2-Accompany.app"
VST3="$ART/VST3/MRT2-Accompany.vst3"
ENTITLEMENTS="$REPO/scripts/entitlements.plist"
for b in "$APP" "$VST3"; do
    [[ -d "$b" ]] || { echo "error: missing artefact $b" >&2; exit 1; }
done

# ── sign ─────────────────────────────────────────────────────────────────────
# Inside-out: the metallib first (non-Mach-O — signature lands in xattrs,
# and codesign refuses --options=runtime for it), then the bundle, whose
# resource seal then covers the signed metallib. The main executable is
# signed as part of the bundle signature; there are no nested dylibs.
for b in "$APP" "$VST3"; do
    echo "==> signing $b"
    codesign --force --sign "$CODESIGN_IDENTITY" --timestamp \
        "$b/Contents/MacOS/mlx.metallib"
    codesign --force --sign "$CODESIGN_IDENTITY" --options=runtime --timestamp \
        --entitlements "$ENTITLEMENTS" "$b"
    codesign --verify --deep --strict --verbose=2 "$b"
done

# ── notarize + staple ────────────────────────────────────────────────────────
if [[ "$notarize" == "1" ]]; then
    for b in "$APP" "$VST3"; do
        echo
        echo "############ notarizing $b ############"
        # Sanity: hardened runtime must be on the inner binary or Gatekeeper
        # rejects the bundle at load time in hardened hosts even if the
        # notary accepts it. (Capture instead of piping to grep -q: grep's
        # early exit + pipefail turns a match into a spurious failure.)
        inner_bin=$(find "$b/Contents/MacOS" -maxdepth 1 -type f ! -name "*.metallib" | head -1)
        cs_out=$(codesign -dvvv "$inner_bin" 2>&1 || true)
        if [[ "$cs_out" != *"flags=0x10000(runtime)"* ]]; then
            echo "error: hardened runtime NOT set on $inner_bin" >&2
            exit 1
        fi

        tmp=$(mktemp -d)
        zip_path="$tmp/$(basename "$b").zip"
        # ditto -c -k --keepParent = the archive layout Apple's notary
        # expects (preserves resource forks + xattrs).
        ditto -c -k --keepParent "$b" "$zip_path"
        echo "==> submitting (Apple ID: $APPLE_ID, team: $APPLE_TEAM_ID) — blocks until Apple replies"
        xcrun notarytool submit "$zip_path" \
            --apple-id "$APPLE_ID" \
            --team-id  "$APPLE_TEAM_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --wait
        rm -rf "$tmp"
        echo "==> stapling"
        xcrun stapler staple "$b"
        spctl -a -vv --type install "$b" 2>&1 || spctl -a -vv "$b" || true
    done
else
    echo "(skipping notarization — credentials unset or --no-notarize passed)"
fi

# ── package ──────────────────────────────────────────────────────────────────
OUT="$REPO/build/release_assets"
rm -rf "$OUT" && mkdir -p "$OUT"
TMP=$(mktemp -d)
cp -R "$APP"  "$TMP/"
cp -R "$VST3" "$TMP/"
cat > "$TMP/README.txt" <<'EOF'
MRT2-Accompany — install
========================

1. Drag the bundles to the standard locations:
     MRT2-Accompany.app    ->  /Applications/
     MRT2-Accompany.vst3   ->  ~/Library/Audio/Plug-Ins/VST3/

2. First run: the plugin offers a one-click model download (~3 GB from
   HuggingFace) into ~/Documents/Magenta/magenta-rt-v2/. Nothing else to
   install.

Requirements: Apple Silicon Mac (M-series Pro/Max recommended for
real-time generation), macOS 14+.
EOF
DMG="$OUT/MRT2-Accompany.dmg"
hdiutil create -srcfolder "$TMP" -volname "MRT2-Accompany" \
    -format UDZO -fs HFS+ -quiet "$DMG"
rm -rf "$TMP"
codesign --force --sign "$CODESIGN_IDENTITY" --timestamp "$DMG"
echo
ls -lh "$DMG"
echo "release: ok — upload $DMG"
