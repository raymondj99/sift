//! Single-cell experiment runner.
//!
//! A "cell" is one `(model × matryoshka dim × RRF alpha × corpus)`
//! configuration. The runner indexes the corpus, runs every query, and
//! emits TREC-standard metrics + per-query latency + per-category
//! breakdown.
//!
//! # Architecture
//!
//! Wired through `sift-store::HybridSearchEngine` directly — **not**
//! through `sift-memory`. Driving retrieval through `MemoryStore` would
//! couple measured retrieval quality to the memory tier's decay curve and
//! retrieval-count boost multipliers, conflating two independent
//! variables.
//!
//! Pure in-memory backends (`FlatVectorIndex` + `Bm25Store`) are used so
//! cells are reproducible without on-disk fixtures and feature flags;
//! tantivy/FTS5/HNSW are not in play.
//!
//! # Latency boundary (per `PLAN-retrieval-lab.md` §4.3)
//!
//! - **Inside** the timed region: prefix concatenation, query
//!   tokenization, query embedding, matryoshka truncation, and
//!   `engine.search`.
//! - **Outside** the timed region: corpus embedding, indexing, ONNX
//!   session warm-up (absorbed into `warmup_queries`).
//!
//! [`run_cell`] enforces this boundary by capturing `Instant::now()`
//! before the prefix concat and `elapsed()` after the search returns;
//! corpus embedding happens once at the top of the function and is not
//! repeated.
//!
//! # Determinism
//!
//! ONNX multi-threaded inference is non-associative in float reductions,
//! so two runs produce slightly different scores. The runner enforces a
//! deterministic post-engine sort by `(score desc, doc_id asc)`; for
//! exactly tied scores this gives byte-stable rankings, but ULP-level
//! score differences slip past it. What matters for committed baselines
//! is **aggregate-metric stability**, not byte-exact ranks — Phase 0 E6
//! verifies the property.

use crate::corpus::RetrievalCorpus;
use crate::error::{RetrievalLabError, RetrievalLabResult};
use crate::latency::LatencyCollector;
use crate::metrics::{
    aggregate, average_precision, ndcg_at_k, recall_at_k, reciprocal_rank, Aggregated, Qrels,
};
use crate::repro::Environment;
use serde::{Deserialize, Serialize};
use sift_core::{Chunk, ContentType, EmbeddedChunk, Embedder, SearchMode, SearchResult};
use sift_embed::models::{ModelManager, ModelSpec};
use sift_embed::OnnxEmbedder;
use sift_store::{Bm25Store, FlatVectorIndex, HybridSearchEngine};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const CELL_RESULT_SCHEMA_VERSION: u32 = 1;

/// Default fetch_k multiplier. The engine fetches `top_k * multiplier`
/// candidates from each store before RRF fusion, then truncates to
/// top_k. Matches the hardcoded `top_k * 3` in
/// `sift-store::hybrid::HybridSearchEngine::search` and is kept as a
/// config knob because rerankers (when added) commonly want N >> K.
pub const DEFAULT_FETCH_K_MULTIPLIER: usize = 3;

/// Default warmup. ONNX session JIT, tokenizer JIT, and OS page cache
/// need a few queries to settle before the latency tail represents
/// steady state.
pub const DEFAULT_WARMUP_QUERIES: usize = 5;

#[derive(Clone)]
pub struct CellConfig {
    /// Embedding model. `ModelSpec` lacks `Debug`/`Eq`, so this struct
    /// implements those manually below — the model is identified by its
    /// `name` field, which is stable across the registry.
    pub model: &'static ModelSpec,
    /// Matryoshka truncation dim. Must be ≤ `model.dimensions` and is
    /// validated against `model.matryoshka_dims` (with the base
    /// dimension permitted as a no-op truncation).
    pub dim: usize,
    /// RRF fusion weight for the vector list, `[0.0, 1.0]`.
    pub alpha: f32,
    pub top_k: usize,
    pub fetch_k_multiplier: usize,
    pub warmup_queries: usize,
    pub mode: SearchMode,
}

impl std::fmt::Debug for CellConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CellConfig")
            .field("model", &self.model.name)
            .field("dim", &self.dim)
            .field("alpha", &self.alpha)
            .field("top_k", &self.top_k)
            .field("fetch_k_multiplier", &self.fetch_k_multiplier)
            .field("warmup_queries", &self.warmup_queries)
            .field("mode", &self.mode.as_str())
            .finish()
    }
}

impl CellConfig {
    pub fn validate(&self) -> RetrievalLabResult<()> {
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(RetrievalLabError::InvalidCellConfig(format!(
                "alpha out of range: {}",
                self.alpha
            )));
        }
        if self.top_k == 0 {
            return Err(RetrievalLabError::InvalidCellConfig(
                "top_k must be > 0".into(),
            ));
        }
        if self.fetch_k_multiplier == 0 {
            return Err(RetrievalLabError::InvalidCellConfig(
                "fetch_k_multiplier must be > 0".into(),
            ));
        }
        if self.dim == 0 || self.dim > self.model.dimensions {
            return Err(RetrievalLabError::InvalidCellConfig(format!(
                "dim {} not in (0, {}]",
                self.dim, self.model.dimensions
            )));
        }
        if self.dim != self.model.dimensions && !self.model.matryoshka_dims.contains(&self.dim) {
            return Err(RetrievalLabError::InvalidCellConfig(format!(
                "dim {} not in {}'s matryoshka_dims ({:?})",
                self.dim, self.model.name, self.model.matryoshka_dims
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellConfigJson {
    pub model_name: String,
    pub model_dimensions: usize,
    pub dim_used: usize,
    pub search_prefix: String,
    pub document_prefix: String,
    pub alpha: f32,
    pub top_k: usize,
    pub fetch_k: usize,
    pub warmup_queries: usize,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub name: String,
    pub source_version: String,
    pub n_documents: usize,
    pub n_queries: usize,
    pub n_negative_queries: usize,
    /// Blake3 hex digest of the canonical-JSON encoding of the corpus.
    /// Pinned in every result so a baseline measured against one corpus
    /// snapshot cannot be silently compared to a different snapshot.
    pub corpus_blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub recall_at_k: Aggregated,
    pub ndcg_at_k: Aggregated,
    pub mrr: Aggregated,
    pub map: Aggregated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMetrics {
    pub category: String,
    pub n_queries: usize,
    pub recall_at_k: Aggregated,
    pub ndcg_at_k: Aggregated,
    pub mrr: Aggregated,
    pub map: Aggregated,
}

/// One ranked hit from the search response. Score is the post-fusion RRF
/// score (already normalized to `[0, 1]` by `sift-store::hybrid`);
/// relevance is looked up from the qrels map and `0` if the doc is not
/// in the relevant set for the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedHit {
    pub doc_id: String,
    pub score: f32,
    pub relevance: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryMetrics {
    pub query_id: String,
    pub category: Option<String>,
    pub n_relevant: usize,
    pub recall_at_k: Option<f64>,
    pub ndcg_at_k: Option<f64>,
    pub mrr: Option<f64>,
    pub ap: Option<f64>,
    pub latency_ms: f64,
    pub hits: Vec<RankedHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub n_samples: usize,
    pub n_warmup: usize,
    pub n_outliers: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellResult {
    pub schema_version: u32,
    pub config: CellConfigJson,
    pub corpus: CorpusStats,
    pub aggregate: AggregateMetrics,
    pub per_category: Vec<CategoryMetrics>,
    pub per_query: Vec<PerQueryMetrics>,
    pub latency: LatencyStats,
    pub environment: Environment,
}

/// Run a single cell. See module docs for the latency boundary and
/// determinism contract.
pub fn run_cell(corpus: &RetrievalCorpus, cfg: &CellConfig) -> RetrievalLabResult<CellResult> {
    cfg.validate()?;

    let manager = ModelManager::new().map_err(RetrievalLabError::ModelManager)?;
    manager.init_ort_env_with_override(None);
    let dir = manager.downloaded_model_dir(cfg.model).ok_or_else(|| {
        RetrievalLabError::ModelNotDownloaded {
            model_name: cfg.model.name.into(),
        }
    })?;
    let embedder =
        OnnxEmbedder::load_model(&dir, cfg.model).map_err(|e| RetrievalLabError::EmbedderLoad {
            model_name: cfg.model.name.into(),
            source: e,
        })?;

    // ---- Index build (out of latency boundary) ------------------------
    let chunks = embed_and_chunk(corpus, &embedder, cfg)?;
    let engine = HybridSearchEngine::new(FlatVectorIndex::new(), Bm25Store::new(), cfg.alpha);
    engine.insert(&chunks).map_err(RetrievalLabError::Storage)?;

    let fetch_k = cfg.top_k * cfg.fetch_k_multiplier;

    // ---- Warmup -------------------------------------------------------
    let warmup_n = cfg.warmup_queries.min(corpus.queries.len());
    for q in corpus.queries.iter().take(warmup_n) {
        let _ = embed_and_search(&embedder, &engine, q, cfg, fetch_k)?;
    }

    // ---- Measured loop -----------------------------------------------
    let mut latency = LatencyCollector::new();
    let mut per_query: Vec<PerQueryMetrics> = Vec::with_capacity(corpus.queries.len());

    for q in &corpus.queries {
        let (results, elapsed) = embed_and_search(&embedder, &engine, q, cfg, fetch_k)?;
        latency.record(elapsed);

        let qmap = corpus.qrels_map(&q.id);
        let n_relevant = qmap.values().filter(|&&r| r > 0).count();
        let hits = canonicalize_ranking(results, cfg.top_k, &qmap);
        let retrieved_ids: Vec<String> = hits.iter().map(|h| h.doc_id.clone()).collect();

        per_query.push(PerQueryMetrics {
            query_id: q.id.clone(),
            category: q.category.clone(),
            n_relevant,
            recall_at_k: recall_at_k(&retrieved_ids, &qmap, cfg.top_k),
            ndcg_at_k: ndcg_at_k(&retrieved_ids, &qmap, cfg.top_k),
            mrr: reciprocal_rank(&retrieved_ids, &qmap),
            ap: average_precision(&retrieved_ids, &qmap),
            latency_ms: dur_ms(elapsed),
            hits,
        });
    }

    Ok(CellResult {
        schema_version: CELL_RESULT_SCHEMA_VERSION,
        config: CellConfigJson {
            model_name: cfg.model.name.into(),
            model_dimensions: cfg.model.dimensions,
            dim_used: cfg.dim,
            search_prefix: cfg.model.search_prefix.into(),
            document_prefix: cfg.model.document_prefix.into(),
            alpha: cfg.alpha,
            top_k: cfg.top_k,
            fetch_k,
            warmup_queries: warmup_n,
            mode: cfg.mode.as_str().into(),
        },
        corpus: CorpusStats {
            name: corpus.name.clone(),
            source_version: corpus.source_version.clone(),
            n_documents: corpus.documents.len(),
            n_queries: corpus.queries.len(),
            n_negative_queries: corpus.queries.iter().filter(|q| q.negative).count(),
            corpus_blake3: corpus.content_hash(),
        },
        aggregate: aggregate_metrics(&per_query),
        per_category: per_category_metrics(&per_query),
        per_query,
        latency: latency_stats(latency, warmup_n),
        environment: Environment::capture(),
    })
}

/// Embed all documents with the model's `document_prefix` applied,
/// matryoshka-truncate to `cfg.dim`, and wrap as `EmbeddedChunk`s.
/// Pre-allocates the prefixed strings in one batch to keep the embedder
/// call efficient.
fn embed_and_chunk(
    corpus: &RetrievalCorpus,
    embedder: &OnnxEmbedder,
    cfg: &CellConfig,
) -> RetrievalLabResult<Vec<EmbeddedChunk>> {
    let prefixed: Vec<String> = corpus
        .documents
        .iter()
        .map(|d| format!("{}{}", cfg.model.document_prefix, d.text))
        .collect();
    let prefixed_refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
    let mut vectors = embedder
        .embed_batch(&prefixed_refs)
        .map_err(RetrievalLabError::EmbeddingFailed)?;
    for v in &mut vectors {
        truncate_and_renormalize(v, cfg.dim);
    }
    Ok(corpus
        .documents
        .iter()
        .zip(vectors)
        .map(|(d, v)| EmbeddedChunk {
            chunk: Chunk {
                text: d.text.clone(),
                source_uri: d.id.clone(),
                chunk_index: 0,
                content_type: ContentType::Text,
                file_type: "json".into(),
                title: None,
                language: None,
                byte_range: None,
            },
            vector: v,
        })
        .collect())
}

/// Embed one query (with prefix), matryoshka-truncate, and search the
/// engine. Records the elapsed time covering prefix concat → embed →
/// truncate → search per the latency boundary contract documented at
/// the top of this module.
fn embed_and_search(
    embedder: &OnnxEmbedder,
    engine: &HybridSearchEngine<FlatVectorIndex, Bm25Store>,
    q: &crate::corpus::Query,
    cfg: &CellConfig,
    fetch_k: usize,
) -> RetrievalLabResult<(Vec<SearchResult>, Duration)> {
    let start = Instant::now();
    let qtext = format!("{}{}", cfg.model.search_prefix, q.text);
    let mut qvec = embedder
        .embed(&qtext)
        .map_err(RetrievalLabError::EmbeddingFailed)?;
    truncate_and_renormalize(&mut qvec, cfg.dim);
    let results = engine
        .search(&qvec, &q.text, fetch_k, cfg.mode)
        .map_err(RetrievalLabError::Search)?;
    let elapsed = start.elapsed();
    // Wrap the result so the optimizer cannot elide the call. Wrapping
    // the *result*, not the call, is the correct site (see
    // `crates/sift-cli/src/commands/bench.rs` for the mistaken
    // call-wrapping pattern that lets the optimizer drop the result).
    std::hint::black_box(&results);
    Ok((results, elapsed))
}

/// Re-sort `results` by `(score desc, doc_id asc)` and truncate to
/// `top_k`. Resolves the relevance grade per-doc against `qrels` for
/// downstream analysis.
///
/// Without an explicit tie-break, identical scores produce different
/// rankings across runs (the engine's HashMap iteration order is
/// non-deterministic), making p50 metric drift impossible to
/// distinguish from real signal.
fn canonicalize_ranking(
    mut results: Vec<SearchResult>,
    top_k: usize,
    qrels: &Qrels,
) -> Vec<RankedHit> {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.uri.cmp(&b.uri))
    });
    results
        .into_iter()
        .take(top_k)
        .map(|r| {
            let relevance = qrels.get(r.uri.as_str()).copied().unwrap_or(0);
            RankedHit {
                doc_id: r.uri,
                score: r.score,
                relevance,
            }
        })
        .collect()
}

/// Truncate one embedding to `dim` and re-normalize to unit length.
/// `OnnxEmbedder` L2-normalizes outputs at full dim; truncating the
/// prefix breaks the norm and would silently bias cosine scores.
/// Renormalization is the matryoshka contract.
fn truncate_and_renormalize(v: &mut Vec<f32>, dim: usize) {
    if v.len() > dim {
        v.truncate(dim);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}

fn aggregate_metrics(per_query: &[PerQueryMetrics]) -> AggregateMetrics {
    AggregateMetrics {
        recall_at_k: aggregate(per_query.iter().map(|p| p.recall_at_k)),
        ndcg_at_k: aggregate(per_query.iter().map(|p| p.ndcg_at_k)),
        mrr: aggregate(per_query.iter().map(|p| p.mrr)),
        map: aggregate(per_query.iter().map(|p| p.ap)),
    }
}

/// Bucket per-query metrics by their `category` field and aggregate.
/// Queries with `category == None` are excluded from the per-category
/// breakdown but still counted in the top-level aggregate. This is the
/// principled choice — uncategorized queries can't be assigned a row in
/// the breakdown table.
fn per_category_metrics(per_query: &[PerQueryMetrics]) -> Vec<CategoryMetrics> {
    let mut buckets: HashMap<String, Vec<&PerQueryMetrics>> = HashMap::new();
    for p in per_query {
        if let Some(cat) = &p.category {
            buckets.entry(cat.clone()).or_default().push(p);
        }
    }
    let mut out: Vec<CategoryMetrics> = buckets
        .into_iter()
        .map(|(cat, ps)| CategoryMetrics {
            n_queries: ps.len(),
            recall_at_k: aggregate(ps.iter().map(|p| p.recall_at_k)),
            ndcg_at_k: aggregate(ps.iter().map(|p| p.ndcg_at_k)),
            mrr: aggregate(ps.iter().map(|p| p.mrr)),
            map: aggregate(ps.iter().map(|p| p.ap)),
            category: cat,
        })
        .collect();
    out.sort_by(|a, b| a.category.cmp(&b.category));
    out
}

fn latency_stats(mut latency: LatencyCollector, warmup_n: usize) -> LatencyStats {
    let n_outliers = latency.outlier_count();
    LatencyStats {
        n_samples: latency.len(),
        n_warmup: warmup_n,
        n_outliers,
        mean_ms: latency.mean().map_or(0.0, dur_ms),
        p50_ms: latency.p50().map_or(0.0, dur_ms),
        p95_ms: latency.p95().map_or(0.0, dur_ms),
        p99_ms: latency.p99().map_or(0.0, dur_ms),
    }
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use sift_embed::models::NOMIC_EMBED_TEXT_V1_5;

    #[test]
    fn validate_rejects_alpha_out_of_range() {
        let cfg = base_cfg(1.5, 768);
        let e = cfg.validate().unwrap_err();
        assert!(matches!(e, RetrievalLabError::InvalidCellConfig(_)));
    }

    #[test]
    fn validate_rejects_dim_zero() {
        let cfg = base_cfg(0.5, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_dim_above_model() {
        let cfg = base_cfg(0.5, 99_999);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_dim_not_in_matryoshka_set() {
        // 384 is not a matryoshka dim of v1.5 ([768, 512, 256, 128, 64])
        let cfg = base_cfg(0.5, 384);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_base_dim() {
        let cfg = base_cfg(0.5, NOMIC_EMBED_TEXT_V1_5.dimensions);
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_accepts_matryoshka_dim() {
        let cfg = base_cfg(0.7, 256);
        cfg.validate().unwrap();
    }

    #[test]
    fn truncate_and_renormalize_preserves_unit_length() {
        let mut v = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0]; // already unit
        truncate_and_renormalize(&mut v, 2);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert_eq!(v.len(), 2);

        let mut v = vec![3.0_f32, 4.0, 0.0, 0.0]; // norm 5 at full
        truncate_and_renormalize(&mut v, 2);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "got norm {norm}");
    }

    #[test]
    fn canonicalize_ranking_breaks_ties_by_doc_id() {
        let qrels: Qrels = HashMap::new();
        // Two results with identical scores; out-of-order input.
        let results = vec![
            mk_result("zebra", 0.5),
            mk_result("apple", 0.5),
            mk_result("mango", 0.5),
        ];
        let hits = canonicalize_ranking(results, 3, &qrels);
        let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
        assert_eq!(ids, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn canonicalize_ranking_assigns_relevance_from_qrels() {
        let mut qrels: Qrels = HashMap::new();
        qrels.insert("apple".into(), 2);
        qrels.insert("mango".into(), 1);
        let results = vec![mk_result("apple", 0.9), mk_result("zebra", 0.5)];
        let hits = canonicalize_ranking(results, 5, &qrels);
        assert_eq!(hits[0].relevance, 2);
        assert_eq!(hits[1].relevance, 0);
    }

    fn base_cfg(alpha: f32, dim: usize) -> CellConfig {
        CellConfig {
            model: &NOMIC_EMBED_TEXT_V1_5,
            dim,
            alpha,
            top_k: 10,
            fetch_k_multiplier: DEFAULT_FETCH_K_MULTIPLIER,
            warmup_queries: DEFAULT_WARMUP_QUERIES,
            mode: SearchMode::Hybrid,
        }
    }

    fn mk_result(uri: &str, score: f32) -> SearchResult {
        SearchResult {
            uri: uri.into(),
            text: String::new(),
            score,
            chunk_index: 0,
            content_type: ContentType::Text,
            file_type: "json".into(),
            title: None,
            byte_range: None,
        }
    }
}
