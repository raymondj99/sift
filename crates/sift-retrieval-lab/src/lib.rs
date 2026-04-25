//! Retrieval-lab: a reproducible Rust crate for measuring sift's retrieval
//! quality with TREC-standard IR metrics.
//!
//! # Scope
//!
//! - Single-cell experiment runner that takes
//!   `(model × matryoshka dim × RRF alpha × corpus)` to `(recall@k,
//!   nDCG@k, MRR, MAP, latency p50/95/99)`.
//! - Heuristic transform from `evals/codingmem/scenarios.json` into a
//!   committed retrieval corpus + qrels.
//! - Phase 0 hypothesis-validation suite that falsifies, in order, the
//!   foundational assumptions in `PLAN-retrieval-lab.md` before any
//!   sweep / baseline machinery is built on top.
//!
//! # Non-goals
//!
//! - Not a runtime dependency. Excluded from the workspace
//!   `default-members` to keep the fast default build cheap; build with
//!   `cargo build -p sift-retrieval-lab` to opt in.
//! - Does **not** depend on `sift-memory`. Driving retrieval through
//!   `MemoryStore` would entangle measured retrieval quality with the
//!   memory tier's decay curve and retrieval-count boost multipliers,
//!   making it impossible to isolate the effect of a retrieval change.
//!
//! # Public surface
//!
//! ```ignore
//! use sift_retrieval_lab::{
//!     corpus::RetrievalCorpus,
//!     runner::{run_cell, CellConfig},
//! };
//! use sift_embed::models::NOMIC_EMBED_TEXT_V1_5;
//!
//! let corpus = RetrievalCorpus::load("evals/codingmem/codingmem-retrieval.json".as_ref())?;
//! let cfg = CellConfig {
//!     model: &NOMIC_EMBED_TEXT_V1_5,
//!     dim: 768,
//!     alpha: 0.7,
//!     top_k: 10,
//!     fetch_k_multiplier: sift_retrieval_lab::runner::DEFAULT_FETCH_K_MULTIPLIER,
//!     warmup_queries: sift_retrieval_lab::runner::DEFAULT_WARMUP_QUERIES,
//! };
//! let result = run_cell(&corpus, &cfg)?;
//! println!("recall@10 = {:.4}", result.aggregate.recall_at_k.mean);
//! # Ok::<(), sift_retrieval_lab::error::RetrievalLabError>(())
//! ```

pub mod corpus;
pub mod error;
pub mod latency;
pub mod metrics;
pub mod repro;
pub mod runner;
pub mod transform;

pub use error::{RetrievalLabError, RetrievalLabResult};
