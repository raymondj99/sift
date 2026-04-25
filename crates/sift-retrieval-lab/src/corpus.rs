//! Retrieval corpus types and loader.
//!
//! Schema-versioned: the lab refuses to load a corpus whose
//! `schema_version` it does not understand rather than coercing fields,
//! because silent corruption of qrels would invalidate every committed
//! baseline downstream.
//!
//! Validation runs on every load:
//!
//! - Documents must have unique non-empty ids.
//! - Every qrel must reference a document that exists in `documents`.
//! - The query list and qrels map must agree on query ids.
//! - Negative queries (correct-retrieval-is-empty) must have empty qrels.
//!
//! Content-addressing: [`RetrievalCorpus::content_hash`] returns a Blake3
//! hash of the canonicalized JSON encoding. Used as `corpus_blake3` in
//! every result envelope so a baseline produced against `v0.4.0` cannot
//! be silently compared to a run against `v0.5.0`.

use crate::error::{RetrievalLabError, RetrievalLabResult};
use crate::metrics::Qrels;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const CORPUS_SCHEMA_VERSION: u32 = 1;

/// One indexable item.
///
/// `metadata` is intentionally `serde_json::Value` rather than a typed
/// enum: typed metadata collapses under cross-corpus variance (CodingMem
/// has session/event ids, BEIR has title/url) and the lab never reads
/// metadata from the search path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub category: Option<String>,
    /// `true` for "isolation"-style questions whose correct retrieval is
    /// the empty set ("I don't know", "no information available"). The
    /// runner skips these from MRR/nDCG/MAP per the IR convention but
    /// still records what was retrieved for false-positive analysis.
    #[serde(default)]
    pub negative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qrel {
    pub doc_id: String,
    /// Graded relevance, `0..=4` (TREC convention).
    pub relevance: u8,
}

/// A complete retrieval corpus, ready to be loaded into the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCorpus {
    pub schema_version: u32,
    pub name: String,
    /// Source data version, e.g. CodingMem `0.4.0`. Recorded in the
    /// reproducibility envelope so a baseline gets pinned to the corpus
    /// release it was measured against.
    pub source_version: String,
    pub documents: Vec<Document>,
    pub queries: Vec<Query>,
    /// `query_id` -> per-query qrels (graded `0..=4`).
    pub qrels: HashMap<String, Vec<Qrel>>,
}

impl RetrievalCorpus {
    /// Load a corpus from disk. Validates schema version and structural
    /// integrity before returning; a corpus that fails validation is
    /// useless for retrieval evaluation, so failing here is preferable to
    /// silently producing nonsense metrics later.
    pub fn load(path: &Path) -> RetrievalLabResult<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| RetrievalLabError::io(path, e))?;
        let corpus: Self =
            serde_json::from_str(&raw).map_err(|e| RetrievalLabError::parse(path, e))?;
        if corpus.schema_version != CORPUS_SCHEMA_VERSION {
            return Err(RetrievalLabError::CorpusSchemaMismatch {
                path: path.to_path_buf(),
                found: corpus.schema_version,
                expected: CORPUS_SCHEMA_VERSION,
            });
        }
        corpus
            .validate()
            .map_err(|reason| RetrievalLabError::CorpusInvalid {
                path: path.to_path_buf(),
                reason,
            })?;
        Ok(corpus)
    }

    /// Write the corpus to disk as pretty-printed JSON, creating parent
    /// directories as needed.
    ///
    /// On-disk byte order is deterministic: `qrels` keys are routed
    /// through a `BTreeMap` (alphabetic), each query's `Vec<Qrel>` is
    /// sorted by `doc_id` then `relevance`, and `serde_json::Map`
    /// defaults to `BTreeMap` so all object keys appear in sorted
    /// order. A re-run of the transform on unchanged input therefore
    /// produces a byte-identical file — committing the corpus to git
    /// gives clean diffs.
    pub fn save(&self, path: &Path) -> RetrievalLabResult<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| RetrievalLabError::io(parent, e))?;
        }
        // Canonical compact bytes → Value tree → pretty-print. The
        // round-trip is the cheapest way to inherit the canonicalization
        // contract from `canonical_json` while keeping the file
        // human-readable.
        let canonical_bytes = canonical_json(self);
        let value: serde_json::Value = serde_json::from_slice(&canonical_bytes)
            .expect("canonical JSON must round-trip into a Value tree");
        let pretty = serde_json::to_string_pretty(&value)?;
        std::fs::write(path, pretty).map_err(|e| RetrievalLabError::io(path, e))?;
        Ok(())
    }

    /// Structural validation. Returns `Err(reason)` for the first defect
    /// found; the caller should surface this through `RetrievalLabError`
    /// with the offending path.
    pub fn validate(&self) -> Result<(), String> {
        if self.documents.is_empty() {
            return Err("documents list is empty".into());
        }
        if self.queries.is_empty() {
            return Err("queries list is empty".into());
        }

        let mut seen = HashSet::with_capacity(self.documents.len());
        for d in &self.documents {
            if d.id.is_empty() {
                return Err("document with empty id".into());
            }
            if !seen.insert(&d.id) {
                return Err(format!("duplicate document id: {}", d.id));
            }
        }
        let doc_ids: HashSet<&String> = seen;

        let mut query_ids = HashSet::with_capacity(self.queries.len());
        for q in &self.queries {
            if q.id.is_empty() {
                return Err("query with empty id".into());
            }
            if !query_ids.insert(&q.id) {
                return Err(format!("duplicate query id: {}", q.id));
            }
        }

        for (qid, rels) in &self.qrels {
            if !query_ids.contains(qid) {
                return Err(format!("qrels reference unknown query id: {qid}"));
            }
            for r in rels {
                if !doc_ids.contains(&r.doc_id) {
                    return Err(format!(
                        "qrels for query {qid} reference unknown doc id: {}",
                        r.doc_id
                    ));
                }
            }
        }

        for q in &self.queries {
            if q.negative {
                if let Some(rels) = self.qrels.get(&q.id) {
                    if rels.iter().any(|r| r.relevance > 0) {
                        return Err(format!(
                            "negative query {} has non-empty graded qrels",
                            q.id
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Map `query_id` -> qrels HashMap keyed by doc_id, used by the metric
    /// functions. Empty when the query has no entry in `qrels`.
    pub fn qrels_map(&self, query_id: &str) -> Qrels {
        self.qrels
            .get(query_id)
            .map(|v| v.iter().map(|r| (r.doc_id.clone(), r.relevance)).collect())
            .unwrap_or_default()
    }

    /// `(query_id, n_relevant)` for every query — used by the qrels-signal
    /// hypothesis check (Phase 0 E1) and by the spot-check output of
    /// `codingmem-transform`.
    pub fn n_relevant_histogram(&self) -> Vec<(String, usize)> {
        self.queries
            .iter()
            .map(|q| {
                let n = self
                    .qrels
                    .get(&q.id)
                    .map_or(0, |v| v.iter().filter(|r| r.relevance > 0).count());
                (q.id.clone(), n)
            })
            .collect()
    }

    /// Distinct question categories present in the corpus, sorted. Used
    /// by the per-category metric breakdown.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .queries
            .iter()
            .filter_map(|q| q.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// Blake3 hash of the canonical JSON encoding. Used as the
    /// `corpus_blake3` field in every result envelope so a baseline
    /// produced against one corpus snapshot cannot be silently compared
    /// to a run against another.
    ///
    /// Canonicalization: serde_json encodes the same `RetrievalCorpus`
    /// identically on every call (HashMap iteration order is not
    /// deterministic across processes, but `serde_json` sorts map keys
    /// for the value tree we hand it). We hash the
    /// `to_string`-canonical form rather than the on-disk bytes so a
    /// reformatted-but-equivalent file produces the same hash.
    pub fn content_hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        // Stable encoding: serialize with sorted keys at every level.
        let bytes = canonical_json(self);
        hasher.update(&bytes);
        hasher.finalize().to_hex().to_string()
    }
}

/// Serialize a `RetrievalCorpus` to JSON bytes in a form that hashes
/// identically under any HashMap iteration order or `Vec<Qrel>`
/// permutation.
///
/// Why both: `HashMap` serializes in arbitrary order — fixed by routing
/// the qrels map through a `BTreeMap`. But `Vec<Qrel>` *also* preserves
/// insertion order, so two corpora that differ only by qrel order
/// within a query would hash differently. We sort each query's qrels
/// vector by `doc_id` (with `relevance` as a stable secondary
/// tie-break) before serialization. The hash is then invariant under
/// any semantically-preserving reorder.
fn canonical_json(c: &RetrievalCorpus) -> Vec<u8> {
    use std::collections::BTreeMap;

    #[derive(Serialize)]
    struct Canonical<'a> {
        schema_version: u32,
        name: &'a str,
        source_version: &'a str,
        documents: &'a [Document],
        queries: &'a [Query],
        qrels: BTreeMap<&'a str, Vec<Qrel>>,
    }

    let qrels: BTreeMap<&str, Vec<Qrel>> = c
        .qrels
        .iter()
        .map(|(k, v)| {
            let mut sorted: Vec<Qrel> = v.clone();
            sorted.sort_by(|a, b| {
                a.doc_id
                    .cmp(&b.doc_id)
                    .then_with(|| a.relevance.cmp(&b.relevance))
            });
            (k.as_str(), sorted)
        })
        .collect();

    let canonical = Canonical {
        schema_version: c.schema_version,
        name: &c.name,
        source_version: &c.source_version,
        documents: &c.documents,
        queries: &c.queries,
        qrels,
    };
    serde_json::to_vec(&canonical).expect("canonical serialization cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            text: text.into(),
            metadata: serde_json::Value::Null,
        }
    }
    fn query(id: &str, text: &str) -> Query {
        Query {
            id: id.into(),
            text: text.into(),
            category: None,
            negative: false,
        }
    }
    fn rel(doc_id: &str, rel: u8) -> Qrel {
        Qrel {
            doc_id: doc_id.into(),
            relevance: rel,
        }
    }
    fn corpus_of(
        docs: Vec<Document>,
        queries: Vec<Query>,
        qrels: Vec<(&str, Vec<Qrel>)>,
    ) -> RetrievalCorpus {
        RetrievalCorpus {
            schema_version: CORPUS_SCHEMA_VERSION,
            name: "fixture".into(),
            source_version: "test".into(),
            documents: docs,
            queries,
            qrels: qrels.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    #[test]
    fn validate_passes_well_formed_corpus() {
        let c = corpus_of(
            vec![doc("d1", "alpha"), doc("d2", "beta")],
            vec![query("q1", "?")],
            vec![("q1", vec![rel("d1", 1)])],
        );
        c.validate().unwrap();
    }

    #[test]
    fn validate_rejects_duplicate_doc_ids() {
        let c = corpus_of(
            vec![doc("d1", "x"), doc("d1", "y")],
            vec![query("q1", "?")],
            vec![],
        );
        assert!(c.validate().unwrap_err().contains("duplicate document id"));
    }

    #[test]
    fn validate_rejects_qrel_referencing_missing_doc() {
        let c = corpus_of(
            vec![doc("d1", "x")],
            vec![query("q1", "?")],
            vec![("q1", vec![rel("d2", 1)])],
        );
        assert!(c.validate().unwrap_err().contains("unknown doc id"));
    }

    #[test]
    fn validate_rejects_qrel_for_unknown_query() {
        let c = corpus_of(
            vec![doc("d1", "x")],
            vec![query("q1", "?")],
            vec![("q2", vec![rel("d1", 1)])],
        );
        assert!(c.validate().unwrap_err().contains("unknown query id"));
    }

    #[test]
    fn validate_rejects_negative_query_with_relevant_doc() {
        let mut q = query("q1", "?");
        q.negative = true;
        let c = corpus_of(
            vec![doc("d1", "x")],
            vec![q],
            vec![("q1", vec![rel("d1", 2)])],
        );
        assert!(c.validate().unwrap_err().contains("negative query"));
    }

    #[test]
    fn content_hash_is_stable() {
        let c = corpus_of(
            vec![doc("d1", "x")],
            vec![query("q1", "?")],
            vec![("q1", vec![rel("d1", 1)])],
        );
        let h1 = c.content_hash();
        let h2 = c.content_hash();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // blake3 hex
    }

    #[test]
    fn save_produces_byte_stable_output() {
        let c = corpus_of(
            vec![doc("d1", "x"), doc("d2", "y")],
            vec![query("q1", "?"), query("q2", "??")],
            vec![
                ("q2", vec![rel("d2", 1), rel("d1", 1)]),
                ("q1", vec![rel("d1", 1)]),
            ],
        );
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.json");
        let p2 = dir.path().join("b.json");
        c.save(&p1).unwrap();
        c.save(&p2).unwrap();
        let a = std::fs::read(&p1).unwrap();
        let b = std::fs::read(&p2).unwrap();
        assert_eq!(a, b, "save() must produce byte-identical output");
        // And the file round-trips through load().
        let reloaded = RetrievalCorpus::load(&p1).unwrap();
        assert_eq!(reloaded.documents.len(), c.documents.len());
        assert_eq!(reloaded.qrels.len(), c.qrels.len());
    }

    #[test]
    fn content_hash_invariant_under_qrel_reorder() {
        let mut c1 = corpus_of(
            vec![doc("d1", "x"), doc("d2", "y"), doc("d3", "z")],
            vec![query("q1", "?")],
            vec![("q1", vec![rel("d1", 1), rel("d2", 2), rel("d3", 1)])],
        );
        let h1 = c1.content_hash();
        // Permute the qrels Vec; semantics unchanged.
        c1.qrels.get_mut("q1").unwrap().reverse();
        let h2 = c1.content_hash();
        assert_eq!(
            h1, h2,
            "content hash must be invariant under Vec<Qrel> reorder"
        );
    }

    #[test]
    fn content_hash_changes_on_qrel_change() {
        let mut c = corpus_of(
            vec![doc("d1", "x"), doc("d2", "y")],
            vec![query("q1", "?")],
            vec![("q1", vec![rel("d1", 1)])],
        );
        let h1 = c.content_hash();
        c.qrels.get_mut("q1").unwrap().push(rel("d2", 1));
        let h2 = c.content_hash();
        assert_ne!(h1, h2, "hash must reflect qrel change");
    }
}
