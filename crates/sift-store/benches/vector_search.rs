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
            text: format!("chunk text for {}", uri),
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

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    let dim = 768;

    for n in [100, 1_000, 10_000] {
        let store = FlatVectorIndex::new();
        let chunks: Vec<EmbeddedChunk> = (0..n)
            .map(|i| make_embedded_chunk(&format!("file:///{}.txt", i), dim, i as u64))
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
            .map(|i| make_embedded_chunk(&format!("file:///{}.txt", i), dim, i as u64))
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

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_vector_search,
    bench_vector_insert,
);
criterion_main!(benches);
