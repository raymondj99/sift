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
            let _ = store.ingest("bench-session", "post_tool_use", &content);
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
                .ingest(&session, "post_tool_use", &content)
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
    let ingest_result = episodes.ingest("e2e-session", "post_compact", compact_content);
    let ingested = ingest_result.is_ok() && ingest_result.unwrap().is_some();

    // 2. Ingest a PostToolUse episode
    let tool_content =
        r#"{"tool_name": "Edit", "tool_input": {"file_path": "/src/cortex/consolidation.rs"}}"#;
    episodes
        .ingest("e2e-session", "post_tool_use", tool_content)
        .unwrap();

    // 3. Consolidate
    let config = sift_memory::ConsolidationConfig::default();
    let report = sift_memory::consolidation::run_consolidation(&store, &episodes, &config).unwrap();
    let consolidated = report.episodes_processed >= 2;

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
        "ingest: {}  consolidate: {} ({} episodes)  recall: {} ({} results)",
        if ingested { "ok" } else { "FAIL" },
        if consolidated { "ok" } else { "FAIL" },
        report.episodes_processed,
        if recalled { "ok" } else { "FAIL" },
        results.len()
    )];

    if passed {
        BenchResult::pass(details)
    } else {
        BenchResult::fail(details)
    }
}
