#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Config – adjust only if you changed these in the Rust code
##############################################################################
BIN_NAME="wordle_solver"            # ← crate‑/binary‑name in Cargo.toml
SHARD_DIR="./train"                 # ← where each worker writes its CSV
OUT_FILE="$SHARD_DIR/training_data.csv"
##############################################################################


# Ensure the shard directory exists
mkdir -p "$SHARD_DIR"

echo "▶ Building release binary …"
cargo build --release

echo "▶ Starting training parent process …"
# The parent will spawn N workers and block until they all finish
"./target/release/$BIN_NAME" train

echo "▶ Merging shard files → $OUT_FILE"
# Guard: make sure shards exist
shopt -s nullglob
shards=("$SHARD_DIR"/training_data.*.csv)
if [ ${#shards[@]} -eq 0 ]; then
  echo "❌  No shard files found in $SHARD_DIR"
  exit 1
fi

# Header from the first shard
head -n1   "${shards[0]}" >  "$OUT_FILE"
# All rows except header from every shard
for f in "${shards[@]}"; do
  tail -n +2 "$f" >> "$OUT_FILE"
done

echo "✅  Done! Combined CSV has $(($(wc -l <"$OUT_FILE") - 1)) data rows."
