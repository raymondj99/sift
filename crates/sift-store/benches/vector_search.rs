use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sift_core::{Chunk, ContentType, EmbeddedChunk};
use sift_store::{FlatVectorIndex, VectorStore};

fn make_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f32 * 0.618 + i as f32 * 0.317).sin())
        .collect()
}

fn make_embedded_chunk(uri: &str, dim: usize, seed: u64) -> EmbeddedChunk {
    EmbeddedChunk {
        chunk: Chunk {
            text: format!("chunk text for {uri}"),
            source_uri: uri.to_string(),
            chunk_index: 0,
            content_type: ContentType::Text,
            file_type: "txt".to_string(),
            title: None,
            language: None,
            byte_range: None,
        },
        vector: make_random_vector(dim, seed),
    }
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 256, 512, 768, 1024] {
        let a: Vec<f32> = make_random_vector(dim, 42);
        let b: Vec<f32> = make_random_vector(dim, 99);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                let (dot, norm_a, norm_b) = a
                    .iter()
                    .zip(b.iter())
                    .fold((0.0f32, 0.0f32, 0.0f32), |(dot, na, nb), (&ai, &bi)| {
                        (dot + ai * bi, na + ai * ai, nb + bi * bi)
                    });
                let denom = (norm_a * norm_b).sqrt();
                if denom == 0.0 {
                    0.0f32
                } else {
                    dot / denom
                }
            });
        });
    }
    group.finish();
}

fn bench_flat_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_search");
    let dim = 768;

    for n in [100, 1_000, 10_000] {
        let store = FlatVectorIndex::new();
        let chunks: Vec<EmbeddedChunk> = (0..n)
            .map(|i| make_embedded_chunk(&format!("file:///{i}.txt"), dim, i as u64))
            .collect();
        store.insert(&chunks).unwrap();

        let query = make_random_vector(dim, 12345);

        group.bench_with_input(BenchmarkId::new("entries", n), &n, |bench, _| {
            bench.iter(|| store.search(&query, 10).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "hnsw")]
fn bench_hnsw_search(c: &mut Criterion) {
    use sift_store::HnswIndex;

    let mut group = c.benchmark_group("hnsw_search");
    let dim = 768;

    for n in [100, 1_000, 10_000] {
        let store = HnswIndex::new();
        let chunks: Vec<EmbeddedChunk> = (0..n)
            .map(|i| make_embedded_chunk(&format!("file:///{i}.txt"), dim, i as u64))
            .collect();
        store.insert(&chunks).unwrap();

        let query = make_random_vector(dim, 12345);

        group.bench_with_input(BenchmarkId::new("entries", n), &n, |bench, _| {
            bench.iter(|| store.search(&query, 10).unwrap());
        });
    }
    group.finish();
}

fn bench_vector_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_insert");
    let dim = 768;

    for batch_size in [1, 10, 100] {
        let chunks: Vec<EmbeddedChunk> = (0..batch_size)
            .map(|i| make_embedded_chunk(&format!("file:///{i}.txt"), dim, i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |bench, _| {
                let store = FlatVectorIndex::new();
                bench.iter(|| store.insert(&chunks).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(all(not(feature = "fulltext"), feature = "fts5"))]
fn bench_fts5_search(c: &mut Criterion) {
    use sift_store::{Fts5Store, FullTextStore};

    let mut group = c.benchmark_group("fts5_search");

    let sentences = [
        "the quick brown fox jumps over the lazy dog",
        "rust programming language systems performance",
        "database connection pool management retry",
        "error handling middleware request processing",
        "kubernetes deployment configuration yaml helm",
        "machine learning model training inference pipeline",
        "distributed systems consensus algorithm raft paxos",
        "authentication authorization jwt token refresh",
    ];

    for n in [100, 1_000, 10_000] {
        let store = Fts5Store::open_in_memory().unwrap();
        let chunks: Vec<EmbeddedChunk> = (0..n)
            .map(|i| {
                let text = sentences[i % sentences.len()];
                let full_text = format!("{text} document number {i}");
                EmbeddedChunk {
                    chunk: Chunk {
                        text: full_text,
                        source_uri: format!("file:///{i}.txt"),
                        chunk_index: 0,
                        content_type: ContentType::Text,
                        file_type: "txt".to_string(),
                        title: None,
                        language: None,
                        byte_range: None,
                    },
                    vector: vec![],
                }
            })
            .collect();
        FullTextStore::insert(&store, &chunks).unwrap();

        group.bench_with_input(BenchmarkId::new("entries", n), &n, |bench, _| {
            bench.iter(|| FullTextStore::search(&store, "error handling retry", 10).unwrap());
        });
    }
    group.finish();
}

// Pick the right benchmark group based on available features.
#[cfg(all(feature = "hnsw", not(feature = "fulltext"), feature = "fts5"))]
criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_flat_search,
    bench_hnsw_search,
    bench_vector_insert,
    bench_fts5_search,
);

#[cfg(all(feature = "hnsw", any(feature = "fulltext", not(feature = "fts5"))))]
criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_flat_search,
    bench_hnsw_search,
    bench_vector_insert,
);

#[cfg(not(feature = "hnsw"))]
criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_flat_search,
    bench_vector_insert,
);

criterion_main!(benches);
