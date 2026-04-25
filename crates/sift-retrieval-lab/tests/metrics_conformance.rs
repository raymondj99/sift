//! Pinned-fixture conformance tests for the metric implementations.
//!
//! Numbers were computed by hand against TREC convention (`pytrec_eval`-
//! compatible). If these tests fail, the metric math has drifted and every
//! committed baseline is invalidated.

use sift_retrieval_lab::metrics::{
    aggregate, average_precision, ndcg_at_k, recall_at_k, reciprocal_rank, Qrels,
};

fn qrels(pairs: &[(&str, u8)]) -> Qrels {
    pairs.iter().map(|(d, r)| ((*d).to_string(), *r)).collect()
}

fn ranked(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn fixture_a_perfect_ranking() {
    let q = qrels(&[("d1", 1), ("d2", 1), ("d3", 1)]);
    let r = ranked(&["d1", "d2", "d3", "d4", "d5"]);
    assert_eq!(recall_at_k(&r, &q, 5), Some(1.0));
    assert_eq!(reciprocal_rank(&r, &q), Some(1.0));
    assert_eq!(average_precision(&r, &q), Some(1.0));
    assert_eq!(ndcg_at_k(&r, &q, 5).map(round6), Some(1.0));
}

#[test]
fn fixture_b_one_relevant_at_position_4() {
    let q = qrels(&[("d4", 1)]);
    let r = ranked(&["d1", "d2", "d3", "d4", "d5"]);
    assert_eq!(recall_at_k(&r, &q, 5), Some(1.0));
    assert!((reciprocal_rank(&r, &q).unwrap() - 0.25).abs() < 1e-9);
    assert!((average_precision(&r, &q).unwrap() - 0.25).abs() < 1e-9);
    // nDCG@5: DCG = 1 / log2(5) = 1/2.32193 = 0.43067; IDCG = 1/log2(2) = 1.0
    let n = ndcg_at_k(&r, &q, 5).unwrap();
    assert!((n - 0.430676558).abs() < 1e-6, "got {n}");
}

#[test]
fn fixture_c_graded_relevance_partial() {
    // Qrels: d1=3, d2=1, d3=2. Retrieved: [d3, d1, d2, d4, d5]
    // DCG = (2^2-1)/log2(2) + (2^3-1)/log2(3) + (2^1-1)/log2(4)
    //     = 3/1 + 7/1.58496 + 1/2 = 3 + 4.41618 + 0.5 = 7.91618
    // IDCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    //      = 7 + 1.89279 + 0.5 = 9.39279
    // DCG = 3.0 + 7/log2(3) + 0.5 = 3.0 + 4.4165083 + 0.5 = 7.9165083
    // IDCG = 7.0 + 3/log2(3) + 0.5 = 7.0 + 1.8927893 + 0.5 = 9.3927893
    // nDCG = 7.9165083 / 9.3927893 = 0.8428283
    let q = qrels(&[("d1", 3), ("d2", 1), ("d3", 2)]);
    let r = ranked(&["d3", "d1", "d2", "d4", "d5"]);
    let n = ndcg_at_k(&r, &q, 5).unwrap();
    assert!((n - 0.8428283).abs() < 1e-6, "got {n}");
}

#[test]
fn fixture_d_no_relevant_skips_query() {
    let q = qrels(&[]);
    let r = ranked(&["d1", "d2"]);
    assert_eq!(recall_at_k(&r, &q, 5), None);
    assert_eq!(reciprocal_rank(&r, &q), None);
    assert_eq!(average_precision(&r, &q), None);
    assert_eq!(ndcg_at_k(&r, &q, 5), None);
}

#[test]
fn fixture_e_recall_caps_at_k_over_relevant() {
    // 5 relevant docs but k=2; retrieved 2 of them ⇒ recall = 2/5
    let q = qrels(&[("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1)]);
    let r = ranked(&["a", "b", "z", "y"]);
    assert!((recall_at_k(&r, &q, 2).unwrap() - 0.4).abs() < 1e-9);
}

#[test]
fn aggregate_skips_none_values() {
    let queries: Vec<Option<f64>> = vec![Some(1.0), Some(0.5), None, Some(0.0)];
    let agg = aggregate(queries);
    assert_eq!(agg.n_evaluated, 3);
    assert_eq!(agg.n_skipped, 1);
    assert!((agg.mean - 0.5).abs() < 1e-9);
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}
