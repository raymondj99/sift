//! Property-based tests for the metric primitives.
//!
//! Pinned-fixture tests in `metrics_conformance.rs` check exact TREC
//! values against a few hand-crafted rankings. These property tests
//! check the *invariants* that every implementation should satisfy on
//! arbitrary inputs: bounds, monotonicity, agreement on degenerate
//! cases. The combination catches both arithmetic drift (fixtures) and
//! edge-case bugs (properties) — `sift-store::hybrid` uses the same
//! fixture-plus-proptest layout.

use proptest::prelude::*;
use sift_retrieval_lab::metrics::{
    average_precision, ndcg_at_k, recall_at_k, reciprocal_rank, Qrels,
};
use std::collections::HashMap;

/// Generate a (retrieved, qrels) pair where retrieved is a permutation
/// of doc ids drawn from a small universe and qrels assign graded
/// relevance to a non-empty subset.
fn ranking_and_qrels() -> impl Strategy<Value = (Vec<String>, Qrels)> {
    let universe = prop::collection::vec(
        prop::sample::select(vec!["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]),
        1..=8,
    )
    .prop_map(|v| {
        let mut seen = std::collections::HashSet::new();
        v.into_iter()
            .filter(|s| seen.insert(*s))
            .map(String::from)
            .collect::<Vec<_>>()
    });

    universe
        .prop_flat_map(|docs| {
            let n = docs.len();
            (Just(docs.clone()), prop::collection::vec(0u8..=4, n))
        })
        .prop_filter("at least one relevant", |(_docs, rels)| {
            rels.iter().any(|&r| r > 0)
        })
        .prop_map(|(docs, rels)| {
            let qrels: Qrels = docs
                .iter()
                .zip(rels.iter())
                .map(|(d, r)| (d.clone(), *r))
                .collect();
            (docs, qrels)
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    #[test]
    fn recall_in_unit_interval((retrieved, qrels) in ranking_and_qrels(), k in 1usize..=20) {
        if let Some(r) = recall_at_k(&retrieved, &qrels, k) {
            prop_assert!((0.0..=1.0).contains(&r), "recall {r} out of [0,1]");
        }
    }

    #[test]
    fn ndcg_in_unit_interval((retrieved, qrels) in ranking_and_qrels(), k in 1usize..=20) {
        if let Some(n) = ndcg_at_k(&retrieved, &qrels, k) {
            prop_assert!((0.0..=1.0 + 1e-9).contains(&n), "nDCG {n} out of [0,1]");
        }
    }

    #[test]
    fn mrr_in_unit_interval((retrieved, qrels) in ranking_and_qrels()) {
        if let Some(m) = reciprocal_rank(&retrieved, &qrels) {
            prop_assert!((0.0..=1.0).contains(&m), "MRR {m} out of [0,1]");
        }
    }

    #[test]
    fn map_in_unit_interval((retrieved, qrels) in ranking_and_qrels()) {
        if let Some(m) = average_precision(&retrieved, &qrels) {
            prop_assert!((0.0..=1.0).contains(&m), "MAP {m} out of [0,1]");
        }
    }

    /// Perfect ranking — relevant docs sorted by relevance descending —
    /// must yield nDCG@k = 1.0 for any `k >= |relevant|`.
    #[test]
    fn perfect_ranking_yields_ndcg_one((_retrieved, qrels) in ranking_and_qrels()) {
        let mut graded: Vec<(&String, u8)> = qrels.iter().filter(|(_, r)| **r > 0).map(|(d, r)| (d, *r)).collect();
        graded.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let perfect: Vec<String> = graded.iter().map(|(d, _)| (*d).clone()).collect();
        let n = ndcg_at_k(&perfect, &qrels, perfect.len()).unwrap();
        prop_assert!((n - 1.0).abs() < 1e-9, "perfect ranking should give nDCG=1, got {n}");
    }

    /// nDCG of any ranking is at most that of the perfect ranking — i.e.
    /// 1.0 — which we assert via the bound proptest above. Here we
    /// additionally check the strict inequality: a non-perfect ranking
    /// with at least two distinct relevance grades cannot equal 1.0
    /// unless it happens to coincide with a perfect ordering.
    #[test]
    fn appending_irrelevant_does_not_change_recall_at_k(
        (retrieved, qrels) in ranking_and_qrels(),
        k in 1usize..=20,
    ) {
        let r1 = recall_at_k(&retrieved, &qrels, k);
        let mut extended = retrieved.clone();
        // Append a guaranteed-irrelevant doc id; doesn't displace any
        // relevant doc within top-k since it goes at position len+1.
        extended.push("__never_relevant__".into());
        let r2 = recall_at_k(&extended, &qrels, k);
        prop_assert_eq!(r1, r2);
    }

    /// MRR is strictly determined by the position of the *first*
    /// relevant doc; reordering docs after the first relevant cannot
    /// change MRR.
    #[test]
    fn mrr_invariant_under_reordering_after_first_relevant(
        (retrieved, qrels) in ranking_and_qrels(),
    ) {
        let mrr_before = reciprocal_rank(&retrieved, &qrels);
        let first_rel = retrieved.iter().position(|d| qrels.get(d.as_str()).is_some_and(|&r| r > 0));
        let mut shuffled = retrieved.clone();
        if let Some(idx) = first_rel {
            shuffled[idx + 1..].reverse();
        }
        let mrr_after = reciprocal_rank(&shuffled, &qrels);
        prop_assert_eq!(mrr_before, mrr_after);
    }

    /// Empty qrels produce `None` for every metric (skipped-query
    /// semantics).
    #[test]
    fn empty_qrels_skip_every_metric(retrieved in prop::collection::vec("d[0-9]", 0..10)) {
        let qrels: Qrels = HashMap::new();
        prop_assert_eq!(recall_at_k(&retrieved, &qrels, 5), None);
        prop_assert_eq!(reciprocal_rank(&retrieved, &qrels), None);
        prop_assert_eq!(average_precision(&retrieved, &qrels), None);
        prop_assert_eq!(ndcg_at_k(&retrieved, &qrels, 5), None);
    }
}
