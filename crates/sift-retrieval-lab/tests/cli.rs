//! End-to-end tests for the `retrieval-lab` binary that don't require an
//! ONNX model to be present.
//!
//! Tests that need real embedding (`run`, `phase0`, `research`) live in
//! Phase 0 itself — they require the model and run as a manual gate, not
//! a unit test. These tests cover the cheap surface: corpus validation
//! and Markdown reporting.
//!
//! `Command::cargo_bin` was deprecated in favor of a macro that handles
//! custom cargo build-dirs; we don't use a custom build-dir, and the old
//! API is still the canonical way to address a workspace binary by name.
#![allow(deprecated)]

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;

const SCHEMA_VERSION: u32 = 1;

fn write_corpus(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
    let p = dir.join("corpus.json");
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(body.as_bytes()).unwrap();
    p
}

#[test]
fn validate_accepts_well_formed_corpus() {
    let tmp = tempfile::tempdir().unwrap();
    let body = format!(
        r#"{{
            "schema_version": {SCHEMA_VERSION},
            "name": "test",
            "source_version": "0.0.0",
            "documents": [
                {{"id": "d1", "text": "hello world", "metadata": null}}
            ],
            "queries": [
                {{"id": "q1", "text": "hello", "category": "smoke"}}
            ],
            "qrels": {{
                "q1": [{{"doc_id": "d1", "relevance": 2}}]
            }}
        }}"#
    );
    let path = write_corpus(tmp.path(), &body);
    Command::cargo_bin("retrieval-lab")
        .unwrap()
        .args(["validate", "--corpus"])
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("ok"))
        .stdout(predicate::str::contains("blake3"));
}

#[test]
fn validate_rejects_unknown_schema_version() {
    let tmp = tempfile::tempdir().unwrap();
    let body = r#"{
        "schema_version": 999,
        "name": "test",
        "source_version": "0.0.0",
        "documents": [{"id": "d1", "text": "x", "metadata": null}],
        "queries": [{"id": "q1", "text": "?"}],
        "qrels": {}
    }"#;
    let path = write_corpus(tmp.path(), body);
    Command::cargo_bin("retrieval-lab")
        .unwrap()
        .args(["validate", "--corpus"])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("schema version"));
}

#[test]
fn validate_rejects_dangling_qrels() {
    let tmp = tempfile::tempdir().unwrap();
    let body = format!(
        r#"{{
            "schema_version": {SCHEMA_VERSION},
            "name": "test",
            "source_version": "0.0.0",
            "documents": [{{"id": "d1", "text": "x", "metadata": null}}],
            "queries": [{{"id": "q1", "text": "?"}}],
            "qrels": {{
                "q1": [{{"doc_id": "missing", "relevance": 1}}]
            }}
        }}"#
    );
    let path = write_corpus(tmp.path(), &body);
    Command::cargo_bin("retrieval-lab")
        .unwrap()
        .args(["validate", "--corpus"])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown doc id"));
}

#[test]
fn report_renders_markdown_from_run_json() {
    let tmp = tempfile::tempdir().unwrap();
    let run_json = sample_cell_result_json();
    let p = tmp.path().join("run.json");
    std::fs::write(&p, run_json).unwrap();
    let out = tmp.path().join("report.md");
    Command::cargo_bin("retrieval-lab")
        .unwrap()
        .args(["report"])
        .arg(&p)
        .args(["--out"])
        .arg(&out)
        .assert()
        .success();
    let body = std::fs::read_to_string(&out).unwrap();
    assert!(body.contains("# retrieval-lab cell result"));
    assert!(body.contains("Aggregate metrics"));
    assert!(body.contains("recall@10"));
}

/// Hand-written CellResult JSON matching the v1 schema. Built by reading
/// the schema fields rather than depending on the public types so this
/// test catches accidental schema changes.
fn sample_cell_result_json() -> &'static str {
    r#"{
  "schema_version": 1,
  "config": {
    "model_name": "nomic-embed-text-v1.5",
    "model_dimensions": 768,
    "dim_used": 768,
    "search_prefix": "search_query: ",
    "document_prefix": "search_document: ",
    "alpha": 0.7,
    "top_k": 10,
    "fetch_k": 30,
    "warmup_queries": 5,
    "mode": "hybrid"
  },
  "corpus": {
    "name": "fixture",
    "source_version": "0.0.0",
    "n_documents": 1,
    "n_queries": 1,
    "n_negative_queries": 0,
    "corpus_blake3": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "aggregate": {
    "recall_at_k": {"mean": 0.9, "n_evaluated": 1, "n_skipped": 0},
    "ndcg_at_k":   {"mean": 0.8, "n_evaluated": 1, "n_skipped": 0},
    "mrr":         {"mean": 1.0, "n_evaluated": 1, "n_skipped": 0},
    "map":         {"mean": 0.8, "n_evaluated": 1, "n_skipped": 0}
  },
  "per_category": [],
  "per_query": [],
  "latency": {
    "n_samples": 1, "n_warmup": 0, "n_outliers": 0,
    "mean_ms": 0.5, "p50_ms": 0.5, "p95_ms": 0.5, "p99_ms": 0.5
  },
  "environment": {
    "git_sha": null,
    "git_dirty": false,
    "rustc": "rustc 1.75.0",
    "target": "aarch64-apple-darwin",
    "timestamp_utc": "2026-04-25T00:00:00Z"
  }
}"#
}
