#!/bin/bash
# Fetch the Meta-World V2 MuJoCo assets (meshes, textures, scene XMLs) from
# upstream Metaworld at a pinned commit. These are unmodified binary assets
# (~38M) that we do not ship in this repository to keep clones small.
#
# Two asset files are modified in this fork and ship with the repo; this
# script never overwrites existing files, so they are preserved.
#
# Usage: bash scripts/fetch_metaworld_assets.sh
set -euo pipefail

UPSTREAM=https://github.com/Farama-Foundation/Metaworld
SHA=19411bd64337d25070c3a24b969cb23847968888
SUBDIR=metaworld/envs/assets_v2
EXPECTED_FILES=325

cd "$(dirname "$0")/.."
DEST="$SUBDIR"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Fetching $SUBDIR from $UPSTREAM @ ${SHA:0:10} ..."
git clone --quiet --filter=blob:none --no-checkout --sparse "$UPSTREAM" "$TMP/mw"
git -C "$TMP/mw" sparse-checkout set --no-cone "/$SUBDIR" >/dev/null
git -C "$TMP/mw" checkout --quiet "$SHA"

mkdir -p "$DEST"
# Copy fetched files, never overwriting the locally modified asset files
# shipped in this repo.
(cd "$TMP/mw/$SUBDIR" && find . -type f) | while IFS= read -r f; do
    if [ ! -e "$DEST/$f" ]; then
        mkdir -p "$DEST/$(dirname "$f")"
        cp "$TMP/mw/$SUBDIR/$f" "$DEST/$f"
    fi
done

COUNT=$(find "$DEST" -type f | wc -l)
if [ "$COUNT" -ne "$EXPECTED_FILES" ]; then
    echo "ERROR: expected $EXPECTED_FILES files in $DEST, found $COUNT" >&2
    exit 1
fi
echo "Done: $COUNT asset files in $DEST"
