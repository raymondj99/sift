#!/usr/bin/env bash
set -euo pipefail

# Generate a synthetic test corpus for sift benchmarks.
# Usage: bash benchsuite/create_corpus.sh [TARGET_DIR]

CORPUS="${1:-/tmp/sift-bench-corpus}"
mkdir -p "$CORPUS"

echo "Generating test corpus at $CORPUS..."

# Generate Rust source files (2000 files across 50 modules)
for i in $(seq 1 2000); do
    dir="$CORPUS/src/module_$(printf '%03d' $((i % 50)))"
    mkdir -p "$dir"
    cat > "$dir/file_$i.rs" << 'RUST'
use std::collections::HashMap;

/// A sample struct for benchmarking.
pub struct Handler {
    config: HashMap<String, String>,
    count: usize,
}

impl Handler {
    pub fn new() -> Self {
        Self {
            config: HashMap::new(),
            count: 0,
        }
    }

    pub fn process(&mut self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.count += 1;
        let result = format!("processed: {} (count: {})", input, self.count);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process() {
        let mut h = Handler::new();
        assert!(h.process("test").is_ok());
    }
}
RUST
done

# Generate Markdown documentation files (1000 files across 20 sections)
for i in $(seq 1 1000); do
    dir="$CORPUS/docs/section_$(printf '%02d' $((i % 20)))"
    mkdir -p "$dir"
    cat > "$dir/doc_$i.md" << 'MD'
# Performance Optimization Guide

This document covers various performance optimization techniques
for building high-performance CLI tools in Rust.

## Key Principles

1. Minimize allocations in hot paths
2. Use SIMD for data-parallel operations
3. Batch I/O operations
4. Leverage parallel processing with rayon

## Database Optimization

When using SQLite, always enable WAL mode and batch writes
within transactions for maximum throughput.

## Error Handling

Use proper error types with thiserror for zero-cost abstractions.
Avoid dynamic dispatch in performance-critical paths.
MD
done

# Generate JSON data files (500 files)
mkdir -p "$CORPUS/data"
for i in $(seq 1 500); do
    echo '{"id":'$i',"name":"item_'$i'","value":'$((RANDOM % 1000))'}' > "$CORPUS/data/item_$i.json"
done

echo "Created corpus:"
find "$CORPUS" -type f | wc -l | xargs echo "  Total files:"
du -sh "$CORPUS" | awk '{print "  Total size:", $1}'
