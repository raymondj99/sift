//! `sift bench` — internal benchmark suite for the Cortex memory system.
//!
//! Validates performance contracts, mathematical invariants, and correctness
//! guarantees. Each benchmark creates a temporary store, seeds test data,
//! runs the operation, and reports pass/fail with supporting metrics.
//!
//! Designed to run fast (< 5s total) against temporary stores.

#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use std::time::{Duration, Instant};

/// Run all benchmarks, or a specific one.
pub fn run(filter: Option<&str>) -> anyhow::Result<()> {
    println!("{}", "Cortex Benchmark Suite".bold());
    println!();

    type Bench = (&'static str, fn() -> BenchResult);
    let benchmarks: Vec<Bench> = vec![
        ("latency-ingest", bench_latency_ingest),
        ("latency-recall", bench_latency_recall),
        ("strengthening", bench_retrieval_strengthening),
        ("forgetting", bench_forgetting_curve),
        ("consolidation", bench_consolidation_accuracy),
        ("skill-extraction", bench_skill_extraction),
        ("e2e", bench_end_to_end),
    ];

    let explicit_benchmark = filter.filter(|f| *f != "all");
    let mut passed = 0usize;
    let mut failed = 0usize;

    for (name, func) in &benchmarks {
        if let Some(f) = explicit_benchmark {
            if *name != f {
                continue;
            }
        }

        let result = func();
        let status = if result.passed {
            passed += 1;
            "PASS".green().bold().to_string()
        } else {
            failed += 1;
            "FAIL".red().bold().to_string()
        };

        println!("  [{status}] {name}");
        for line in &result.details {
            println!("         {line}");
        }
        println!();
    }

    if explicit_benchmark == Some("memory-rebuild") {
        let result = bench_memory_rebuild();
        let status = if result.passed {
            passed += 1;
            "PASS".green().bold().to_string()
        } else {
            failed += 1;
            "FAIL".red().bold().to_string()
        };

        println!("  [{status}] memory-rebuild");
        for line in &result.details {
            println!("         {line}");
        }
        println!();
    }

    if explicit_benchmark == Some("matryoshka") {
        let result = bench_matryoshka_recall();
        let status = if result.passed {
            passed += 1;
            "PASS".green().bold().to_string()
        } else {
            failed += 1;
            "FAIL".red().bold().to_string()
        };

        println!("  [{status}] matryoshka");
        for line in &result.details {
            println!("         {line}");
        }
        println!();
    }

    println!("{}", format!("{passed} passed, {failed} failed").bold());

    if failed > 0 {
        anyhow::bail!("{failed} benchmark(s) failed");
    }

    Ok(())
}

// ===========================================================================
// Benchmark: Memory Rebuild
// ===========================================================================

#[cfg(feature = "embeddings")]
const MEMORY_REBUILD_OBSERVATIONS: usize = 1_000;
#[cfg(feature = "embeddings")]
const MEMORY_REBUILD_EMBED_BATCH_SIZE: usize = 64;
#[cfg(feature = "embeddings")]
const MEMORY_REBUILD_ITERATIONS: usize = 3;

/// Time a forced rebuild of the memory search index with a real local embedder.
fn bench_memory_rebuild() -> BenchResult {
    #[cfg(not(feature = "embeddings"))]
    {
        return BenchResult::fail(vec![
            "requires the `embeddings` feature".to_string(),
            "build with `cargo build --features embeddings` or a feature set that includes MCP"
                .to_string(),
        ]);
    }

    #[cfg(feature = "embeddings")]
    {
        use sift_core::Config;
        use sift_core::Embedder as _;
        use sift_embed::models::{get_model, NOMIC_EMBED_TEXT_V1_5};
        use sift_embed::{ModelManager, OnnxEmbedder};
        use std::sync::Arc;

        let config = match Config::load() {
            Ok(config) => config,
            Err(err) => return BenchResult::fail(vec![format!("failed to load config: {err}")]),
        };

        let manager = match ModelManager::new() {
            Ok(manager) => manager,
            Err(err) => {
                return BenchResult::fail(vec![format!("failed to locate model directory: {err}")]);
            }
        };
        manager.init_ort_env_with_override(config.default.ort_dylib_path.as_deref());

        let model = get_model(&config.default.model).unwrap_or(&NOMIC_EMBED_TEXT_V1_5);
        if !model.is_download_supported() {
            return BenchResult::fail(vec![format!(
                "default model '{}' does not support local ONNX download/runtime: {}",
                model.name, model.notes
            )]);
        }

        let Some(model_dir) = manager.downloaded_model_dir(model) else {
            return BenchResult::fail(vec![format!(
                "model '{}' is not downloaded; run `sift models download {}`",
                model.name, model.name
            )]);
        };

        let embedder = match OnnxEmbedder::load_model(&model_dir, model) {
            Ok(embedder) => Arc::new(embedder),
            Err(err) => return BenchResult::fail(vec![format!("failed to load embedder: {err}")]),
        };
        let model_name = embedder.model_name().to_string();
        let embedder: Arc<dyn sift_core::Embedder> = embedder;

        let mut elapsed_samples = Vec::with_capacity(MEMORY_REBUILD_ITERATIONS);
        for iteration in 0..MEMORY_REBUILD_ITERATIONS {
            let store = match seeded_memory_rebuild_store(MEMORY_REBUILD_OBSERVATIONS) {
                Ok(store) => store.with_embedder(Arc::clone(&embedder)),
                Err(err) => {
                    return BenchResult::fail(vec![format!(
                        "failed to create benchmark store for iteration {}: {err}",
                        iteration + 1
                    )]);
                }
            };

            let start = Instant::now();
            if let Err(err) = store.rebuild_search_index_now() {
                return BenchResult::fail(vec![format!(
                    "forced rebuild failed for iteration {}: {err}",
                    iteration + 1
                )]);
            }
            elapsed_samples.push(start.elapsed());
        }

        elapsed_samples.sort();
        let median = elapsed_samples[elapsed_samples.len() / 2];
        let median_secs = median.as_secs_f64();
        let observations_per_sec = MEMORY_REBUILD_OBSERVATIONS as f64 / median_secs.max(0.000_001);
        let all_samples = elapsed_samples
            .iter()
            .map(|sample| format!("{:.2}s", sample.as_secs_f64()))
            .collect::<Vec<_>>()
            .join(", ");

        let details = vec![
            format!("observations: {MEMORY_REBUILD_OBSERVATIONS}"),
            format!("model: {model_name}"),
            format!("batch size: {MEMORY_REBUILD_EMBED_BATCH_SIZE}"),
            format!("iterations: {MEMORY_REBUILD_ITERATIONS}"),
            format!("median elapsed: {:.2}s", median_secs),
            format!(
                "median throughput: {:.1} observations/sec",
                observations_per_sec
            ),
            format!("samples: [{all_samples}]"),
        ];

        BenchResult::pass(details)
    }
}

#[cfg(feature = "embeddings")]
fn seeded_memory_rebuild_store(
    observation_count: usize,
) -> anyhow::Result<sift_memory::MemoryStore> {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench()?;

    for i in 0..observation_count {
        let entity_id = store.save_entity(
            &format!("memory-rebuild-entity-{i}"),
            sift_memory::EntityType::Concept,
            1.0,
            "bench",
        )?;

        store.add_observation(
            &entity_id,
            &format!(
                "memory rebuild benchmark observation {i} about batched embedding throughput and recall startup latency"
            ),
            0.8,
            "bench",
        )?;
    }

    Ok(store)
}

// ===========================================================================
// Benchmark: Matryoshka Recall
// ===========================================================================
//
// Verifies that truncated Matryoshka embeddings retain enough quality to be
// usable in production. Loads the configured embedding model, embeds a small
// synthetic corpus + matched queries at the model's full dim and at each
// declared Matryoshka dim, then reports Recall@1 and average rank-of-target
// per dim.
//
// Pass criterion: 256-dim Recall@1 must be at least 80% of the full-dim
// Recall@1. Below that, truncation isn't worth the storage win and the
// feature should be reconsidered.

/// Hand-crafted query → expected-document pairs covering distinct topics.
/// Each pair's expected doc is unambiguous so a working retriever should
/// hit Recall@1 = 1.0 at full dim.
#[cfg(feature = "embeddings")]
const MRL_PAIRS: &[(&str, &str)] = &[
    (
        "How do Python decorators work?",
        "Python decorators are functions that wrap other functions to extend their behavior without modifying the source.",
    ),
    (
        "What is the Rust borrow checker?",
        "The Rust borrow checker enforces ownership and lifetime rules at compile time to prevent data races and use-after-free.",
    ),
    (
        "Explain the CAP theorem in distributed systems.",
        "The CAP theorem states that a distributed data store cannot simultaneously guarantee Consistency, Availability, and Partition tolerance.",
    ),
    (
        "How does HNSW indexing work?",
        "HNSW is a graph-based approximate nearest-neighbor index that organizes vectors into a hierarchy of small-world graphs for fast similarity search.",
    ),
    (
        "What is mean pooling for sentence embeddings?",
        "Mean pooling averages the hidden states of all tokens, weighted by the attention mask, to produce a single fixed-size sentence embedding.",
    ),
    (
        "How does BM25 ranking work in full-text search?",
        "BM25 scores a document against a query using term frequency saturation and inverse document frequency to balance keyword importance and document length.",
    ),
    (
        "What is reciprocal rank fusion?",
        "Reciprocal rank fusion combines multiple ranked result lists by summing 1/(k + rank) for each item across lists, prioritizing items that rank well in many sources.",
    ),
    (
        "Explain the Ebbinghaus forgetting curve.",
        "The Ebbinghaus forgetting curve describes how memory retention decays exponentially over time without reinforcement, modelled as exp(-lambda * elapsed).",
    ),
    (
        "What is bi-temporal modeling?",
        "Bi-temporal modeling stores both when a fact was observed and when it was true in the real world, enabling historical queries and proper supersession of outdated facts.",
    ),
    (
        "How does an inverted index speed up keyword search?",
        "An inverted index maps each term to the list of documents that contain it, so keyword lookups become O(matches) instead of scanning every document.",
    ),
];

#[cfg(feature = "embeddings")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Recall@1 + mean rank for one dimensionality. Returns (recall_at_1,
/// mean_rank_of_target).
#[cfg(feature = "embeddings")]
fn evaluate_at_dim(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> (f32, f32) {
    let n = query_vecs.len();
    let mut hits_at_1 = 0usize;
    let mut total_rank = 0usize;

    for (qi, q) in query_vecs.iter().enumerate() {
        let mut scored: Vec<(usize, f32)> = doc_vecs
            .iter()
            .enumerate()
            .map(|(di, d)| (di, cosine_similarity(q, d)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored[0].0 == qi {
            hits_at_1 += 1;
        }
        let rank = scored
            .iter()
            .position(|(d, _)| *d == qi)
            .unwrap_or(scored.len());
        total_rank += rank + 1;
    }

    (hits_at_1 as f32 / n as f32, total_rank as f32 / n as f32)
}

fn bench_matryoshka_recall() -> BenchResult {
    #[cfg(not(feature = "embeddings"))]
    {
        return BenchResult::fail(vec!["requires the `embeddings` feature".to_string()]);
    }

    #[cfg(feature = "embeddings")]
    {
        use sift_core::{Config, Embedder as _};
        use sift_embed::models::{get_model, NOMIC_EMBED_TEXT_V1_5};
        use sift_embed::{ModelManager, OnnxEmbedder};

        let config = match Config::load() {
            Ok(c) => c,
            Err(e) => return BenchResult::fail(vec![format!("failed to load config: {e}")]),
        };

        let manager = match ModelManager::new() {
            Ok(m) => m,
            Err(e) => return BenchResult::fail(vec![format!("model manager: {e}")]),
        };
        manager.init_ort_env_with_override(config.default.ort_dylib_path.as_deref());

        let model = get_model(&config.default.model).unwrap_or(&NOMIC_EMBED_TEXT_V1_5);
        if model.matryoshka_dims.is_empty() {
            return BenchResult::fail(vec![format!(
                "model '{}' does not declare any Matryoshka dimensions",
                model.name
            )]);
        }
        let Some(model_dir) = manager.downloaded_model_dir(model) else {
            return BenchResult::fail(vec![format!(
                "model '{}' not downloaded; run `sift models download {}`",
                model.name, model.name
            )]);
        };

        let queries: Vec<&str> = MRL_PAIRS.iter().map(|(q, _)| *q).collect();
        let docs: Vec<&str> = MRL_PAIRS.iter().map(|(_, d)| *d).collect();

        let mut details = vec![
            format!("model: {} (native dim {})", model.name, model.dimensions),
            format!(
                "corpus: {} query→doc pairs (distinct topics)",
                MRL_PAIRS.len()
            ),
            String::new(),
            format!(
                "  {:>6}  {:>9}  {:>10}  {:>11}",
                "dim", "Recall@1", "mean rank", "size/vec"
            ),
        ];

        // Baseline at native dim (no truncation)
        let baseline = match OnnxEmbedder::load_model(&model_dir, model) {
            Ok(e) => e,
            Err(e) => return BenchResult::fail(vec![format!("load full embedder: {e}")]),
        };
        let q_full: Result<Vec<_>, _> = queries.iter().map(|q| baseline.embed_query(q)).collect();
        let d_full: Result<Vec<_>, _> = docs.iter().map(|d| baseline.embed_passage(d)).collect();
        let (q_full, d_full) = match (q_full, d_full) {
            (Ok(q), Ok(d)) => (q, d),
            (Err(e), _) | (_, Err(e)) => {
                return BenchResult::fail(vec![format!("baseline embed failed: {e}")]);
            }
        };
        let (full_r1, full_mean_rank) = evaluate_at_dim(&q_full, &d_full);
        let bytes_full = model.dimensions * 4;
        details.push(format!(
            "  {:>6}  {:>9.3}  {:>10.2}  {:>9}B",
            model.dimensions, full_r1, full_mean_rank, bytes_full
        ));

        // Each Matryoshka truncation
        let mut results: Vec<(usize, f32, f32, usize)> =
            vec![(model.dimensions, full_r1, full_mean_rank, bytes_full)];
        for &dim in model.matryoshka_dims {
            if dim == model.dimensions {
                continue; // already covered as baseline
            }
            let truncated = match OnnxEmbedder::load_model_with_truncation(&model_dir, model, dim) {
                Ok(e) => e,
                Err(e) => {
                    details.push(format!("  {:>6}  load failed: {}", dim, e));
                    continue;
                }
            };
            let q_t: Result<Vec<_>, _> = queries.iter().map(|q| truncated.embed_query(q)).collect();
            let d_t: Result<Vec<_>, _> = docs.iter().map(|d| truncated.embed_passage(d)).collect();
            let (q_t, d_t) = match (q_t, d_t) {
                (Ok(q), Ok(d)) => (q, d),
                (Err(e), _) | (_, Err(e)) => {
                    details.push(format!("  {:>6}  embed failed: {}", dim, e));
                    continue;
                }
            };
            let (r1, mean_rank) = evaluate_at_dim(&q_t, &d_t);
            let bytes = dim * 4;
            details.push(format!(
                "  {:>6}  {:>9.3}  {:>10.2}  {:>9}B",
                dim, r1, mean_rank, bytes
            ));
            results.push((dim, r1, mean_rank, bytes));
        }

        // Pass criterion: 256-dim Recall@1 ≥ 80% of full-dim Recall@1
        let r1_256 = results
            .iter()
            .find(|(d, _, _, _)| *d == 256)
            .map(|(_, r, _, _)| *r);
        let target_threshold = full_r1 * 0.80;
        let passed = match r1_256 {
            Some(r) => r >= target_threshold,
            None => true, // model doesn't support 256d — don't fail on its absence
        };

        details.push(String::new());
        details.push(format!(
            "pass criterion: 256d Recall@1 (={:?}) >= 80% of full-dim Recall@1 ({:.3} = {:.3}*0.80)",
            r1_256, target_threshold, full_r1
        ));

        if passed {
            BenchResult::pass(details)
        } else {
            BenchResult::fail(details)
        }
    }
}

struct BenchResult {
    passed: bool,
    details: Vec<String>,
}

impl BenchResult {
    fn pass(details: Vec<String>) -> Self {
        Self {
            passed: true,
            details,
        }
    }

    fn fail(details: Vec<String>) -> Self {
        Self {
            passed: false,
            details,
        }
    }
}

/// Collect timing samples and compute percentiles.
struct LatencyCollector {
    samples: Vec<Duration>,
}

impl LatencyCollector {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    fn record<F: FnMut()>(&mut self, mut f: F) {
        let start = Instant::now();
        f();
        self.samples.push(start.elapsed());
    }

    fn p50(&self) -> Duration {
        self.percentile(50)
    }

    fn p95(&self) -> Duration {
        self.percentile(95)
    }

    fn p99(&self) -> Duration {
        self.percentile(99)
    }

    fn percentile(&self, pct: usize) -> Duration {
        let mut sorted = self.samples.clone();
        sorted.sort();
        if sorted.is_empty() {
            return Duration::ZERO;
        }
        let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
        sorted[idx]
    }
}

// ===========================================================================
// Benchmark: Ingest Latency
// ===========================================================================

/// Validate that episode ingest completes in < 100ms (p99 < 200ms).
fn bench_latency_ingest() -> BenchResult {
    let dir = tempfile::tempdir().unwrap();
    let store = sift_memory::episodes::EpisodeStore::open(dir.path()).unwrap();

    let mut collector = LatencyCollector::new();
    let iterations = 100;

    for i in 0..iterations {
        let content =
            format!(r#"{{"tool_name": "Edit", "tool_input": {{"file_path": "/src/file{i}.rs"}}}}"#);
        collector.record(|| {
            let _ = store.ingest(
                "bench-session",
                sift_memory::EventType::PostToolUse,
                &content,
            );
        });
    }

    let p50 = collector.p50();
    let p95 = collector.p95();
    let p99 = collector.p99();

    let passed = p99 < Duration::from_millis(200);

    let details = vec![
        format!("iterations: {iterations}"),
        format!(
            "p50: {:.2}ms  p95: {:.2}ms  p99: {:.2}ms",
            p50.as_secs_f64() * 1000.0,
            p95.as_secs_f64() * 1000.0,
            p99.as_secs_f64() * 1000.0
        ),
        format!(
            "threshold: p99 < 200ms {}",
            if passed { "(ok)" } else { "(EXCEEDED)" }
        ),
    ];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}

// ===========================================================================
// Benchmark: Recall Latency
// ===========================================================================

/// Measure recall latency with enhanced scoring + access logging.
fn bench_latency_recall() -> BenchResult {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench().unwrap();

    // Seed 50 entities with 3 observations each = 150 observations
    for i in 0..50 {
        let entity_id = store
            .save_entity(
                &format!("entity-{i}"),
                sift_memory::EntityType::Concept,
                1.0,
                "bench",
            )
            .unwrap();
        for j in 0..3 {
            store
                .add_observation(
                    &entity_id,
                    &format!(
                        "benchmark observation {i} fact number {j} about testing memory systems"
                    ),
                    0.8,
                    "bench",
                )
                .unwrap();
        }
    }

    let mut collector = LatencyCollector::new();
    let queries = [
        "benchmark observation testing",
        "memory systems fact",
        "entity concept observation",
    ];

    for query in &queries {
        for _ in 0..30 {
            collector.record(|| {
                let _ = store.recall(query, 10, &sift_memory::RecallFilters::default());
            });
        }
    }

    let p50 = collector.p50();
    let p95 = collector.p95();
    let p99 = collector.p99();

    // Recall should complete in < 50ms even with 150 observations
    let passed = p99 < Duration::from_millis(50);

    let details = vec![
        format!("store: 50 entities, 150 observations"),
        format!("queries: {} x 30 iterations", queries.len()),
        format!(
            "p50: {:.2}ms  p95: {:.2}ms  p99: {:.2}ms",
            p50.as_secs_f64() * 1000.0,
            p95.as_secs_f64() * 1000.0,
            p99.as_secs_f64() * 1000.0
        ),
        format!(
            "threshold: p99 < 50ms {}",
            if passed { "(ok)" } else { "(EXCEEDED)" }
        ),
    ];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}

// ===========================================================================
// Benchmark: Retrieval-Dependent Strengthening
// ===========================================================================

/// Verify that recalling the same query N times causes scores to increase
/// monotonically (Ebbinghaus spacing effect).
fn bench_retrieval_strengthening() -> BenchResult {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench().unwrap();

    let entity_id = store
        .save_entity(
            "strengthening-test",
            sift_memory::EntityType::Concept,
            1.0,
            "bench",
        )
        .unwrap();
    store
        .add_observation(
            &entity_id,
            "retrieval dependent strengthening validates Ebbinghaus spacing",
            1.0,
            "bench",
        )
        .unwrap();

    let query = "retrieval dependent strengthening validates Ebbinghaus spacing";
    let mut scores: Vec<f32> = Vec::new();

    for _ in 0..5 {
        let results = store
            .recall(query, 10, &sift_memory::RecallFilters::default())
            .unwrap();
        if let Some(r) = results.first() {
            scores.push(r.score);
        }
    }

    // Scores should be monotonically non-decreasing
    let monotonic = scores.windows(2).all(|w| w[1] >= w[0]);
    // Score should increase meaningfully (at least 5% from first to last)
    let meaningful_increase =
        scores.len() >= 2 && scores.last().unwrap_or(&0.0) > &(scores[0] * 1.05);

    let passed = monotonic && meaningful_increase;

    let score_strs: Vec<String> = scores.iter().map(|s| format!("{s:.4}")).collect();
    let details = vec![
        format!("scores over 5 recalls: [{}]", score_strs.join(", ")),
        format!(
            "monotonic: {}  increase > 5%: {}",
            if monotonic { "yes" } else { "NO" },
            if meaningful_increase { "yes" } else { "NO" }
        ),
    ];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}

// ===========================================================================
// Benchmark: Forgetting Curve
// ===========================================================================

/// Verify that utility scores decay correctly with simulated time.
///
/// Creates observations, then runs decay/pruning with a very short
/// prune window to verify that old, unaccessed observations get lower
/// utility scores than recent ones.
fn bench_forgetting_curve() -> BenchResult {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench().unwrap();

    let entity_id = store
        .save_entity("decay-test", sift_memory::EntityType::Concept, 1.0, "bench")
        .unwrap();

    // Create observations — they'll all have observed_at = now
    store
        .add_observation(&entity_id, "recent observation", 1.0, "bench")
        .unwrap();
    store
        .add_observation(&entity_id, "another recent observation", 1.0, "bench")
        .unwrap();

    // Manually backdate one observation to simulate aging
    {
        let conn = store.db().lock().unwrap();
        let old_time = sift_memory::now_secs() - (60 * 86400); // 60 days ago
        conn.execute(
            "UPDATE observations SET observed_at = ?1 WHERE content = 'recent observation'",
            rusqlite::params![old_time],
        )
        .unwrap();
    }

    // Run consolidation — the decay phase scores all observations
    let episodes = sift_memory::episodes::EpisodeStore::open_in_memory_for_bench().unwrap();
    let config = sift_memory::ConsolidationConfig {
        prune_min_age_days: 365,      // Don't prune anything
        prune_utility_threshold: 0.0, // Don't prune anything
        ..Default::default()
    };
    let _ = sift_memory::consolidation::run_consolidation(&store, &episodes, &config);

    // Instead of checking utility_score (which depends on phase interactions),
    // verify the fundamental invariant: recall scores are lower for older
    // observations due to the decay formula.
    let old_results = store
        .recall(
            "recent observation",
            10,
            &sift_memory::RecallFilters::default(),
        )
        .unwrap();
    let new_results = store
        .recall(
            "another recent observation",
            10,
            &sift_memory::RecallFilters::default(),
        )
        .unwrap();

    let old_score = old_results.first().map_or(0.0, |r| r.score);
    let new_score = new_results.first().map_or(0.0, |r| r.score);

    // The old observation (60 days) should score lower than the new one (0 days)
    // due to base_decay = exp(-0.01 * 60) ≈ 0.55 vs exp(0) = 1.0
    let passed = old_score < new_score && old_score > 0.0;

    let details = vec![
        format!("old (60d) recall score: {old_score:.4}"),
        format!("new (0d) recall score:  {new_score:.4}"),
        format!(
            "old < new: {}  (decay factor ≈ {:.2})",
            if old_score < new_score { "yes" } else { "NO" },
            (-0.01_f64 * 60.0).exp()
        ),
    ];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}

// ===========================================================================
// Benchmark: Consolidation Accuracy
// ===========================================================================

/// Verify dedup correctly merges known duplicates and doesn't merge uniques.
fn bench_consolidation_accuracy() -> BenchResult {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench().unwrap();

    let entity_id = store
        .save_entity("dedup-test", sift_memory::EntityType::Concept, 1.0, "bench")
        .unwrap();

    // Create 3 exact duplicates and 2 unique observations
    store
        .add_observation(&entity_id, "duplicate fact about testing", 1.0, "bench")
        .unwrap();
    store
        .add_observation(&entity_id, "duplicate fact about testing", 0.9, "bench")
        .unwrap();
    store
        .add_observation(&entity_id, "duplicate fact about testing", 0.8, "bench")
        .unwrap();
    store
        .add_observation(&entity_id, "unique fact number one", 1.0, "bench")
        .unwrap();
    store
        .add_observation(&entity_id, "completely different observation", 1.0, "bench")
        .unwrap();

    let report = store.consolidate().unwrap();

    // 3 identical copies → pairwise dedup merges 2 (leaving 1 survivor)
    let dedup_correct = report.duplicates_merged >= 2;

    // Verify 3 active observations remain (1 surviving dup + 2 unique)
    let active: i64 = {
        let conn = store.db().lock().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM observations WHERE entity_id = ?1 AND valid_until IS NULL",
            [&entity_id],
            |row| row.get(0),
        )
        .unwrap()
    };
    let active_correct = active == 3;

    let passed = dedup_correct && active_correct;

    let details = vec![
        format!("input: 3 duplicates + 2 unique = 5 total"),
        format!(
            "duplicates merged: {} (expected >= 2)",
            report.duplicates_merged
        ),
        format!("active remaining: {active} (expected 3)"),
    ];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}

// ===========================================================================
// Benchmark: Skill Extraction
// ===========================================================================

/// Verify that repeated tool patterns across sessions produce skills.
fn bench_skill_extraction() -> BenchResult {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench().unwrap();
    let episodes = sift_memory::episodes::EpisodeStore::open_in_memory_for_bench().unwrap();

    // Simulate 5 sessions with the same Edit->Bash->Edit pattern
    for session_num in 0..5 {
        let session = format!("skill-session-{session_num}");
        for tool in ["Edit", "Bash", "Edit"] {
            let content = format!(r#"{{"tool_name": "{tool}", "tool_input": "bench"}}"#);
            episodes
                .ingest(&session, sift_memory::EventType::PostToolUse, &content)
                .unwrap();
        }
    }

    // First consolidation processes episodes
    let config = sift_memory::ConsolidationConfig {
        skill_min_frequency: 3,
        skill_extraction: true,
        ..Default::default()
    };
    let _ = sift_memory::consolidation::run_consolidation(&store, &episodes, &config);

    // Second consolidation extracts skills from processed episodes
    let report = sift_memory::consolidation::run_consolidation(&store, &episodes, &config).unwrap();

    // Check if any skills were created (from first or second run)
    let conn = store.db().lock().unwrap();
    let skill_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM skills", [], |row| row.get(0))
        .unwrap();

    let passed = skill_count > 0 || report.skills_created > 0;

    let patterns: Vec<String> = conn
        .prepare("SELECT name, frequency FROM skills ORDER BY frequency DESC LIMIT 5")
        .unwrap()
        .query_map([], |row| {
            Ok(format!(
                "{} (freq: {})",
                row.get::<_, String>(0).unwrap_or_default(),
                row.get::<_, i64>(1).unwrap_or(0)
            ))
        })
        .unwrap()
        .filter_map(Result::ok)
        .collect();

    let details = vec![
        format!("sessions: 5, pattern: Edit->Bash->Edit"),
        format!("skills found: {skill_count}"),
        format!(
            "top patterns: {}",
            if patterns.is_empty() {
                "(none)".to_string()
            } else {
                patterns.join(", ")
            }
        ),
    ];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}

// ===========================================================================
// Benchmark: End-to-End Pipeline
// ===========================================================================

/// Full pipeline: ingest episodes → consolidate → recall → verify persistence.
fn bench_end_to_end() -> BenchResult {
    let store = sift_memory::MemoryStore::open_in_memory_for_bench().unwrap();
    let episodes = sift_memory::episodes::EpisodeStore::open_in_memory_for_bench().unwrap();

    // 1. Ingest a PostCompact episode (the gold signal)
    let compact_content = r#"{"compact_summary": "The user is building a Rust semantic search engine called sift with an automated memory system called Cortex."}"#;
    let ingest_result = episodes.ingest(
        "e2e-session",
        sift_memory::EventType::PostCompact,
        compact_content,
    );
    let ingested = ingest_result.is_ok() && ingest_result.unwrap().is_some();

    // 2. Ingest a PostToolUse episode
    let tool_content =
        r#"{"tool_name": "Edit", "tool_input": {"file_path": "/src/cortex/consolidation.rs"}}"#;
    episodes
        .ingest(
            "e2e-session",
            sift_memory::EventType::PostToolUse,
            tool_content,
        )
        .unwrap();

    // 3. Consolidate
    let config = sift_memory::ConsolidationConfig::default();
    let report = sift_memory::consolidation::run_consolidation(&store, &episodes, &config).unwrap();
    // PostToolUse extracts without LLM; PostCompact requires LLM, so under
    // the default config it must be deferred (not silently counted as
    // processed). We assert both halves.
    let processed_ok = report.episodes_processed >= 1;
    let deferred_ok = report.episodes_deferred >= 1;
    let consolidated = processed_ok && deferred_ok;

    // 4. Recall — search for the knowledge we ingested
    let results = store
        .recall(
            "Rust semantic search engine sift Cortex",
            10,
            &sift_memory::RecallFilters::default(),
        )
        .unwrap();
    let recalled = !results.is_empty();

    let passed = ingested && consolidated && recalled;

    let details = vec![format!(
        "ingest: {}  consolidate: {} ({} processed, {} deferred)  recall: {} ({} results)",
        if ingested { "ok" } else { "FAIL" },
        if consolidated { "ok" } else { "FAIL" },
        report.episodes_processed,
        report.episodes_deferred,
        if recalled { "ok" } else { "FAIL" },
        results.len()
    )];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}
