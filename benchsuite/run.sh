#!/usr/bin/env bash
set -euo pipefail

# sift Benchmark Suite
#
# End-to-end CLI benchmarks using hyperfine.
# Requires: hyperfine (https://github.com/sharkdp/hyperfine)
#
# Usage:
#   cargo build --release
#   bash benchsuite/run.sh
#
# Environment variables:
#   SIFT_BENCH_CORPUS  Path to test corpus (default: /tmp/sift-bench-corpus)
#   SIFT_BIN           Path to sift binary (default: ./target/release/sift)

CORPUS="${SIFT_BENCH_CORPUS:-/tmp/sift-bench-corpus}"
RESULTS_DIR="benchsuite/results"
SIFT="${SIFT_BIN:-./target/release/sift}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$RESULTS_DIR"

echo "=== sift Benchmark Suite ==="
echo "Corpus: $CORPUS"
echo "Binary: $SIFT"
echo ""

# Check for hyperfine
if ! command -v hyperfine &> /dev/null; then
    echo "Error: hyperfine is required but not installed."
    echo "Install: cargo install hyperfine"
    exit 1
fi

# Check for sift binary
if [ ! -x "$SIFT" ]; then
    echo "Error: sift binary not found at $SIFT"
    echo "Run: cargo build --release"
    exit 1
fi

# Ensure corpus exists
if [ ! -d "$CORPUS" ]; then
    echo "Creating test corpus at $CORPUS..."
    bash "$SCRIPT_DIR/create_corpus.sh" "$CORPUS"
fi

# Clean any existing bench indexes
rm -rf ~/.sift/indexes/bench-* 2>/dev/null || true

echo "--- Benchmark 1: Fresh scan (cold) ---"
hyperfine --warmup 0 --runs 3 \
    --export-json "$RESULTS_DIR/fresh_scan.json" \
    "SIFT_INDEX=bench-fresh $SIFT scan $CORPUS"

echo ""
echo "--- Benchmark 2: Incremental scan (all cached) ---"
SIFT_INDEX=bench-incr "$SIFT" scan "$CORPUS" > /dev/null 2>&1
hyperfine --warmup 1 --runs 5 \
    --export-json "$RESULTS_DIR/incremental_scan.json" \
    "SIFT_INDEX=bench-incr $SIFT scan $CORPUS"

echo ""
echo "--- Benchmark 3: Keyword search ---"
SIFT_INDEX=bench-search "$SIFT" scan "$CORPUS" > /dev/null 2>&1
hyperfine --warmup 3 --runs 10 \
    --export-json "$RESULTS_DIR/keyword_search.json" \
    "SIFT_INDEX=bench-search $SIFT search --keyword-only 'function error handling'" \
    "SIFT_INDEX=bench-search $SIFT search --keyword-only 'import os sys'" \
    "SIFT_INDEX=bench-search $SIFT search --keyword-only 'database connection pool'"

echo ""
echo "--- Benchmark 4: Dry run (discovery only) ---"
hyperfine --warmup 3 --runs 5 \
    --export-json "$RESULTS_DIR/dry_run.json" \
    "SIFT_INDEX=bench-dry $SIFT scan --dry-run $CORPUS"

echo ""
echo "=== Results saved to $RESULTS_DIR ==="
