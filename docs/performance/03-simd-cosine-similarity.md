# 03: SIMD-Accelerated Cosine Similarity

## Problem

In `sift-store/src/flat.rs:422-443`, the cosine similarity function uses a naive scalar loop:

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}
```

With 768-dimensional vectors (nomic-embed-text-v2) and O(n) brute-force search over all entries, this is the hottest loop in vector search. SIMD can provide 4-8x speedup on this inner loop.

## Solution

Two-pronged approach:

### A) Help LLVM auto-vectorize (minimal changes)

Restructure the loop to enable LLVM auto-vectorization by using iterators and `zip`:

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (dot, norm_a, norm_b) = a.iter().zip(b.iter()).fold(
        (0.0f32, 0.0f32, 0.0f32),
        |(dot, na, nb), (&ai, &bi)| {
            (dot + ai * bi, na + ai * ai, nb + bi * bi)
        },
    );

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}
```

This is more auto-vectorization-friendly because LLVM can see the independent accumulations.

### B) Use SimSIMD for maximum performance (recommended)

Add the `simsimd` crate which provides hand-tuned SIMD kernels for x86 AVX2/AVX-512 and ARM NEON/SVE:

**File: `crates/sift-store/Cargo.toml`**

```toml
[dependencies]
simsimd = { version = "6", optional = true }

[features]
simd = ["dep:simsimd"]
```

**File: `crates/sift-store/src/flat.rs`**

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    #[cfg(feature = "simd")]
    {
        use simsimd::SpatialSimilarity;
        // simsimd returns cosine *distance* (1 - similarity), convert back
        match f32::cos(a, b) {
            Some(distance) => 1.0 - distance as f32,
            None => fallback_cosine(a, b),
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        fallback_cosine(a, b)
    }
}

/// Scalar fallback for when SIMD is not available.
fn fallback_cosine(a: &[f32], b: &[f32]) -> f32 {
    let (dot, norm_a, norm_b) = a.iter().zip(b.iter()).fold(
        (0.0f32, 0.0f32, 0.0f32),
        |(dot, na, nb), (&ai, &bi)| {
            (dot + ai * bi, na + ai * ai, nb + bi * bi)
        },
    );

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}
```

### C) Precompute norms for stored vectors

Since embedding vectors don't change after insertion, precompute and store the L2 norm. This reduces cosine to a simple dot product divided by two precomputed scalars:

```rust
struct StoredEntry {
    uri: String,
    text: String,
    vector: Vec<f32>,
    norm: f32,  // precomputed L2 norm
    // ... other fields
}

// On insert:
let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

// On search:
fn cosine_with_precomputed_norm(query: &[f32], query_norm: f32, entry: &StoredEntry) -> f32 {
    let dot: f32 = query.iter().zip(entry.vector.iter()).map(|(a, b)| a * b).sum();
    let denom = query_norm * entry.norm;
    if denom == 0.0 { 0.0 } else { dot / denom }
}
```

### Feature Gate Integration

**File: `crates/sift-cli/Cargo.toml`**

```toml
[features]
simd = ["sift-store/simd"]
standard = ["ast", "pdf", ..., "simd"]  # Include in standard profile
full = ["standard", ...]
```

## Tests

```rust
#[cfg(test)]
mod cosine_tests {
    use super::*;

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let score = cosine_similarity(&v, &v);
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_768_dimensions() {
        // Realistic embedding dimension
        let a: Vec<f32> = (0..768).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32).cos()).collect();
        let score = cosine_similarity(&a, &b);
        assert!(score > -1.0 && score < 1.0);
    }

    #[test]
    fn test_cosine_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_zero_vectors() {
        let z = vec![0.0; 768];
        assert_eq!(cosine_similarity(&z, &z), 0.0);
    }

    #[test]
    fn test_cosine_mismatched_lengths() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_fallback() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).cos()).collect();

        let simd_result = cosine_similarity(&a, &b);
        let fallback_result = fallback_cosine(&a, &b);

        // Allow small floating-point difference
        assert!((simd_result - fallback_result).abs() < 1e-4,
            "SIMD={}, fallback={}", simd_result, fallback_result);
    }
}
```

## Evaluation Metric

**Benchmark: Vector search over N entries with 768-dim vectors**

```rust
// Add to benches/vector_search.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_cosine_similarity(c: &mut Criterion) {
    let dims = [128, 256, 512, 768, 1024];
    let mut group = c.benchmark_group("cosine_similarity");

    for &dim in &dims {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| cosine_similarity(&a, &b));
        });
    }
    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 50_000, 100_000];
    let dim = 768;
    let mut group = c.benchmark_group("vector_search");

    for &n in &sizes {
        let store = FlatVectorIndex::new();
        // ... populate with n random vectors ...

        let query: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();

        group.bench_with_input(BenchmarkId::new("brute_force", n), &n, |bench, _| {
            bench.iter(|| store.search(&query, 10));
        });
    }
    group.finish();
}
```

Expected improvement:
- **Auto-vectorized**: ~2x faster than naive loop
- **SimSIMD**: ~4-8x faster than naive loop (hardware dependent)
- **With precomputed norms**: additional ~30% on top of SIMD
