#!/bin/bash
set -e

# This script fetches upstream genie_sim at a specific commit,
# and keeps ONLY the "source/" directory.

COMMIT_ID="6321d714a25d6b33031338a0f9fcc1553e43066f"

echo "[1/3] Fetching upstream genie_sim"

git clone https://github.com/AgibotTech/genie_sim.git --no-checkout --filter=blob:none
cd genie_sim

git sparse-checkout init --cone 2>/dev/null || true
git sparse-checkout set source

git fetch --depth 1 origin $COMMIT_ID
git checkout $COMMIT_ID

echo "[2/3] Fetched genie_sim source at $COMMIT_ID"

git am ../patches/*.patch

echo "[3/3] Applied patches to genie_sim"
