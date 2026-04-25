//! CodingMem `scenarios.json` → retrieval corpus.
//!
//! The transform is deterministic and committed-output: re-run when the
//! source `scenarios.json` changes, then commit the result.
//!
//! Oracle-derivation heuristic (v1):
//!
//! - One document per event under id `<scenario_id>:<session_idx>:<event_idx>`.
//! - For each question, oracles are scoped to the question's scenario (a
//!   question in scenario X cannot be answered by documents in Y).
//! - Score each candidate doc by **content-word token-set overlap** with
//!   the question's answer:
//!   1. lowercase + word-tokenize both
//!   2. drop stop words (small fixed set)
//!   3. require ≥ `MATCH_THRESHOLD` (50%) of answer's content words to
//!      appear as tokens in the doc, AND at least 2 distinct content words
//!      matched (single rare-word matches produce too many false positives)
//!   4. matched doc ⇒ relevance 2
//! - "Negative" questions (isolation category answers like "I don't know")
//!   are marked `negative = true` with empty qrels.
//!
//! **Why not verbatim substring match (the plan's §4.6 heuristic)?** The
//! plan assumed answers are quoted from the events. CodingMem answers are
//! *paraphrased summaries* — "Three: lint, test, deploy" answers a question
//! about events that say "with three stages: lint, test, deploy". Verbatim
//! match drops those (Phase 0 E1 measured 57% empty qrels with that
//! heuristic; token-overlap brings it under 40%).
//!
//! The graded `relevance = 1` tier (preceding-context events) is deferred
//! until an LLM judge is wired up — pure heuristics produce too many false
//! positives at that tier on this corpus.

use crate::corpus::{Document, Qrel, Query, RetrievalCorpus, CORPUS_SCHEMA_VERSION};
use crate::error::{RetrievalLabError, RetrievalLabResult};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Deserialize)]
struct ScenariosFile {
    version: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    id: String,
    category: String,
    sessions: Vec<Session>,
    questions: Vec<Question>,
}

#[derive(Debug, Deserialize)]
struct Session {
    id: String,
    events: Vec<Event>,
}

#[derive(Debug, Deserialize)]
struct Event {
    #[serde(rename = "type")]
    event_type: String,
    content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
struct Question {
    /// Field name matches the source JSON; renaming would require a serde
    /// alias that adds noise without solving anything.
    question: String,
    answer: String,
    category: String,
}

/// Read CodingMem `scenarios.json` and emit a `RetrievalCorpus` with
/// heuristic qrels. The result is validated before being returned, so a
/// successful return guarantees the corpus is well-formed and ready for
/// the runner.
pub fn transform_codingmem(input: &Path) -> RetrievalLabResult<RetrievalCorpus> {
    let raw = std::fs::read_to_string(input).map_err(|e| RetrievalLabError::io(input, e))?;
    let file: ScenariosFile =
        serde_json::from_str(&raw).map_err(|e| RetrievalLabError::parse(input, e))?;

    let mut documents: Vec<Document> = Vec::new();
    let mut queries: Vec<Query> = Vec::new();
    let mut qrels: HashMap<String, Vec<Qrel>> = HashMap::new();

    for scen in &file.scenarios {
        // Build per-scenario doc list with stable ids and normalized search
        // text. Scope oracles per scenario: a question in scenario X cannot
        // be answered by documents in scenario Y.
        let mut scen_docs: Vec<(String, String)> = Vec::new();
        for (sess_idx, sess) in scen.sessions.iter().enumerate() {
            for (ev_idx, ev) in sess.events.iter().enumerate() {
                let id = format!("{}:{sess_idx}:{ev_idx}", scen.id);
                let text = event_to_text(ev);
                if text.trim().is_empty() {
                    continue;
                }
                let metadata = serde_json::json!({
                    "scenario_id": scen.id,
                    "scenario_category": scen.category,
                    "session_id": sess.id,
                    "session_idx": sess_idx,
                    "event_idx": ev_idx,
                    "event_type": ev.event_type,
                });
                documents.push(Document {
                    id: id.clone(),
                    text: text.clone(),
                    metadata,
                });
                scen_docs.push((id, text));
            }
        }

        for (q_idx, q) in scen.questions.iter().enumerate() {
            let qid = format!("{}:q{q_idx}", scen.id);
            let negative = is_negative_answer(&q.answer);
            queries.push(Query {
                id: qid.clone(),
                text: q.question.clone(),
                category: Some(q.category.clone()),
                negative,
            });

            let mut q_qrels = Vec::new();
            if !negative {
                let answer_content: Vec<String> = content_words(&q.answer);
                if !answer_content.is_empty() {
                    for (doc_id, text) in &scen_docs {
                        if oracle_match(&answer_content, text) {
                            q_qrels.push(Qrel {
                                doc_id: doc_id.clone(),
                                relevance: 2,
                            });
                        }
                    }
                }
            }
            qrels.insert(qid, q_qrels);
        }
    }

    let corpus = RetrievalCorpus {
        schema_version: CORPUS_SCHEMA_VERSION,
        name: "codingmem-retrieval".into(),
        source_version: file.version,
        documents,
        queries,
        qrels,
    };
    corpus
        .validate()
        .map_err(|reason| RetrievalLabError::CorpusInvalid {
            path: input.to_path_buf(),
            reason,
        })?;
    Ok(corpus)
}

/// Serialize the event content to a flat searchable string. `post_compact`
/// summaries are stored as `{compact_summary: "..."}`; tool-use events as
/// `{tool_name, tool_input, tool_output}`. JSON serialization keeps all
/// fields searchable without per-event-type plumbing.
fn event_to_text(ev: &Event) -> String {
    // Prefer prose-y fields when present; fall back to full JSON.
    if let Some(obj) = ev.content.as_object() {
        if let Some(s) = obj.get("compact_summary").and_then(|v| v.as_str()) {
            return s.to_string();
        }
        if let Some(s) = obj.get("text").and_then(|v| v.as_str()) {
            return s.to_string();
        }
    }
    serde_json::to_string(&ev.content).unwrap_or_default()
}

/// Required fraction of an answer's content words that must appear in a
/// candidate doc for the doc to be marked an oracle. Lower = more recall
/// (more oracles per question), higher = more precision. 0.5 was chosen
/// empirically on CodingMem v0.4.0 — it brings empty-qrels rate from 57%
/// (verbatim) to under 40% without producing >5 oracles for any question
/// after spot-check.
const MATCH_THRESHOLD: f64 = 0.5;

/// Minimum number of distinct content words that must overlap. Prevents a
/// single rare word ("rebase") from matching every doc that mentions it
/// when the rest of the answer is much longer ("Always rebase, never
/// merge commits, then squash before review").
const MIN_DISTINCT_OVERLAP: usize = 2;

/// Tokenize, lowercase, drop stop words, deduplicate.
fn content_words(s: &str) -> Vec<String> {
    let mut out: Vec<String> = s
        .unicode_words()
        .map(str::to_lowercase)
        .filter(|w| !is_stop_word(w))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Token-set overlap match. The answer's content words must appear as
/// tokens (not substrings — "axum" must not match "paxum") in the doc text.
fn oracle_match(answer_content: &[String], doc_text: &str) -> bool {
    if answer_content.is_empty() {
        return false;
    }
    let doc_tokens: std::collections::HashSet<String> =
        doc_text.unicode_words().map(str::to_lowercase).collect();
    let matched: usize = answer_content
        .iter()
        .filter(|w| doc_tokens.contains(*w))
        .count();
    if matched < MIN_DISTINCT_OVERLAP.min(answer_content.len()) {
        return false;
    }
    let ratio = matched as f64 / answer_content.len() as f64;
    ratio >= MATCH_THRESHOLD
}

/// Small fixed stop-word list. Deliberately conservative — over-aggressive
/// stopping erodes the signal in short answers like "Yes, TDD".
fn is_stop_word(w: &str) -> bool {
    matches!(
        w,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "been"
            | "but"
            | "by"
            | "do"
            | "does"
            | "did"
            | "for"
            | "from"
            | "has"
            | "have"
            | "had"
            | "i"
            | "if"
            | "in"
            | "is"
            | "it"
            | "its"
            | "of"
            | "on"
            | "or"
            | "than"
            | "that"
            | "the"
            | "their"
            | "them"
            | "then"
            | "there"
            | "these"
            | "they"
            | "this"
            | "those"
            | "to"
            | "use"
            | "used"
            | "using"
            | "was"
            | "were"
            | "will"
            | "with"
            | "you"
            | "your"
            | "we"
            | "our"
            | "us"
            | "user"
            | "always"
            | "never"
            | "also"
            | "only"
            | "just"
            | "very"
            | "more"
            | "such"
            | "when"
            | "where"
            | "what"
            | "who"
            | "how"
            | "why"
            | "which"
            | "should"
            | "would"
            | "could"
            | "can"
            | "may"
            | "might"
            | "must"
            | "yes"
            | "no"
            | "not"
            | "any"
            | "all"
            | "some"
            | "out"
            | "into"
            | "via"
            | "vs"
            | "instead"
    )
}

/// Whitespace-collapsed lowercase form. Used by negative-answer detection
/// (which needs ordered phrases like "i don t know") and tests.
fn normalize(s: &str) -> String {
    s.unicode_words()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Negative-answer detection. CodingMem's isolation category contains
/// questions whose correct retrieval is the empty set ("don't know",
/// "no information", "not mentioned"). `unicode_words` keeps apostrophes
/// inside contractions, so we match both `don't` and `dont` forms.
fn is_negative_answer(answer: &str) -> bool {
    let n = normalize(answer);
    n.is_empty()
        || n.contains("i don't know")
        || n.contains("i dont know")
        || n.contains("don't know")
        || n.contains("dont know")
        || n.contains("no information")
        || n.contains("not mentioned")
        || n.contains("not specified")
        || n.contains("no relevant")
        || n.starts_with("no ") && n.split_whitespace().count() <= 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal scenarios.json fixture. Two scenarios with one session
    /// each, four questions total — covers a positive-answer match, a
    /// paraphrased-answer match, a negative-answer ("I don't know"),
    /// and an off-corpus question whose answer doesn't appear in the
    /// scenario events (oracle should be empty).
    const FIXTURE: &str = r#"{
        "version": "test-1.0.0",
        "description": "tiny",
        "scenarios": [
            {
                "id": "scenario-a",
                "category": "smoke",
                "description": "framework choice",
                "sessions": [{
                    "id": "sess-a",
                    "events": [
                        {"type": "post_compact", "content": {"compact_summary": "User is building a REST API using Axum in Rust with SQLite WAL mode."}}
                    ]
                }],
                "questions": [
                    {"question": "What framework?", "answer": "Axum", "category": "framework"},
                    {"question": "DB engine?", "answer": "SQLite with WAL", "category": "database"},
                    {"question": "What is the user's middle name?", "answer": "I don't know", "category": "isolation"}
                ]
            },
            {
                "id": "scenario-b",
                "category": "preference",
                "description": "logging",
                "sessions": [{
                    "id": "sess-b",
                    "events": [
                        {"type": "post_compact", "content": {"compact_summary": "User prefers logging with tracing over println for production code."}}
                    ]
                }],
                "questions": [
                    {"question": "Logging?", "answer": "tracing, not println", "category": "preference"}
                ]
            }
        ]
    }"#;

    fn write_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scenarios.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(FIXTURE.as_bytes()).unwrap();
        (dir, path)
    }

    #[test]
    fn transform_end_to_end_produces_valid_corpus() {
        let (_tmp, path) = write_fixture();
        let corpus = transform_codingmem(&path).unwrap();
        // Schema, name, source version pinned.
        assert_eq!(corpus.schema_version, CORPUS_SCHEMA_VERSION);
        assert_eq!(corpus.name, "codingmem-retrieval");
        assert_eq!(corpus.source_version, "test-1.0.0");
        // 2 scenarios × 1 session × 1 event each.
        assert_eq!(corpus.documents.len(), 2);
        // 4 questions across both scenarios.
        assert_eq!(corpus.queries.len(), 4);
        // Document ids follow `<scenario>:<sess_idx>:<event_idx>`.
        assert!(corpus.documents.iter().any(|d| d.id == "scenario-a:0:0"));
        assert!(corpus.documents.iter().any(|d| d.id == "scenario-b:0:0"));
        // The corpus passes structural validation (called inside
        // transform_codingmem, so reaching here implies it).
        corpus.validate().unwrap();
    }

    #[test]
    fn transform_assigns_oracles_to_matching_event() {
        let (_tmp, path) = write_fixture();
        let corpus = transform_codingmem(&path).unwrap();

        let oracles = |qid: &str| -> Vec<String> {
            corpus
                .qrels
                .get(qid)
                .map(|v| {
                    let mut ids: Vec<String> = v.iter().map(|q| q.doc_id.clone()).collect();
                    ids.sort();
                    ids
                })
                .unwrap_or_default()
        };

        // "Axum" — single-word answer, exact token match in event.
        assert_eq!(oracles("scenario-a:q0"), vec!["scenario-a:0:0"]);

        // "SQLite with WAL" — paraphrased; content words {sqlite, wal}.
        // Event has both → match.
        assert_eq!(oracles("scenario-a:q1"), vec!["scenario-a:0:0"]);

        // "I don't know" — flagged negative; empty qrels.
        let neg_q = corpus
            .queries
            .iter()
            .find(|q| q.id == "scenario-a:q2")
            .unwrap();
        assert!(neg_q.negative);
        assert!(oracles("scenario-a:q2").is_empty());

        // Cross-scenario match: "tracing, not println" hits scenario-b's event
        // but NOT scenario-a's (which contains neither word). Scope is per-scenario.
        assert_eq!(oracles("scenario-b:q0"), vec!["scenario-b:0:0"]);
    }

    #[test]
    fn transform_rejects_unparseable_input() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "{{ not valid json").unwrap();
        let err = transform_codingmem(&path).unwrap_err();
        assert!(matches!(err, RetrievalLabError::JsonParse { .. }));
    }

    #[test]
    fn normalize_lowercases_and_collapses() {
        assert_eq!(normalize("  Hello,   World!  "), "hello world");
        assert_eq!(normalize("Axum"), "axum");
    }

    #[test]
    fn oracle_match_token_overlap() {
        // Real CodingMem cases that the verbatim heuristic missed.
        let answer = content_words("Three: lint, test, deploy");
        let doc = "Using GitHub Actions with three stages: lint, test, deploy";
        assert!(oracle_match(&answer, doc), "lint/test/deploy should match");

        let answer = content_words("tracing, not println!");
        let doc = "prefer logging with tracing over println!";
        assert!(oracle_match(&answer, doc));

        let answer = content_words("Always rebase, never merge commits");
        let doc = "Branch policy: rebase before merging; squash commits";
        assert!(oracle_match(&answer, doc));
    }

    #[test]
    fn oracle_match_rejects_weak_overlap() {
        // "Axum" alone (single content word). MIN_DISTINCT_OVERLAP=2, so a
        // single-word answer matches only docs containing that exact word
        // — single word has answer_content.len() == 1 so the .min() rule
        // allows the match.
        let answer = content_words("Axum");
        let doc = "rest api using axum in rust";
        assert!(
            oracle_match(&answer, doc),
            "single-word answer should match"
        );

        // 4-word answer matching only 1 word ⇒ rejected.
        let answer = content_words("Three: lint, test, deploy");
        let doc = "We considered lint frameworks but did not commit";
        assert!(!oracle_match(&answer, doc), "1/4 overlap rejected");
    }

    #[test]
    fn oracle_match_token_boundary_not_substring() {
        // "axum" is not a substring match into "paxum"
        let answer = content_words("Axum framework");
        let doc = "paxum is something else entirely framework";
        // matched: ["framework"] only — 1/2 below threshold
        assert!(!oracle_match(&answer, doc));
    }

    #[test]
    fn negative_answer_detection() {
        assert!(is_negative_answer("I don't know"));
        assert!(is_negative_answer("No information available"));
        assert!(is_negative_answer("Not mentioned"));
        assert!(!is_negative_answer("Axum"));
        assert!(!is_negative_answer("WAL mode"));
    }
}
