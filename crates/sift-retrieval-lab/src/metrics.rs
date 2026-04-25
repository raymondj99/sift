//! TREC-standard IR metrics.
//!
//! Skipped-query semantics: queries with empty relevance sets return `None`
//! from per-query functions. The aggregator excludes them from the mean and
//! reports `n_skipped` separately. This matches `pytrec_eval` and the IR
//! literature; treating skipped queries as zero silently shrinks all means.
//!
//! Numerator convention for nDCG is the TREC `2^rel - 1` gain. Discount is
//! `1 / log2(rank + 1)` with rank starting at 1 (i.e. the 0-indexed
//! position `i` contributes `1 / log2(i + 2)`).
//!
//! Tie-break is the caller's responsibility — `retrieved` must already be
//! ordered. The runner enforces lexicographic `doc_id` tie-break after the
//! engine returns, so identical scores produce identical rankings across
//! runs.

use std::collections::HashMap;

/// Map of doc_id -> graded relevance for one query.
pub type Qrels = HashMap<String, u8>;

/// Recall@k. Binary judgment: relevance > 0 ⇒ relevant.
///
/// When `|relevant| > k`, recall caps at `k / |relevant| < 1.0`. This matches
/// `pytrec_eval`; do not renormalize to 1.0.
///
/// Returns `None` when `|relevant| == 0` so the aggregator can skip the
/// query rather than counting it as zero.
#[must_use]
pub fn recall_at_k(retrieved: &[String], qrels: &Qrels, k: usize) -> Option<f64> {
    let relevant_count = qrels.values().filter(|&&r| r > 0).count();
    if relevant_count == 0 {
        return None;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|d| qrels.get(d.as_str()).is_some_and(|&r| r > 0))
        .count();
    Some(hits as f64 / relevant_count as f64)
}

/// Reciprocal rank of the first relevant doc. `0.0` if none in `retrieved`.
/// `None` when `|relevant| == 0`.
#[must_use]
pub fn reciprocal_rank(retrieved: &[String], qrels: &Qrels) -> Option<f64> {
    if !qrels.values().any(|&r| r > 0) {
        return None;
    }
    for (i, d) in retrieved.iter().enumerate() {
        if qrels.get(d.as_str()).is_some_and(|&r| r > 0) {
            return Some(1.0 / (i + 1) as f64);
        }
    }
    Some(0.0)
}

/// Average precision. Denominator is total relevant in the corpus, not
/// retrieved-relevant — wrong denominator improves trivially with k.
#[must_use]
pub fn average_precision(retrieved: &[String], qrels: &Qrels) -> Option<f64> {
    let relevant_count = qrels.values().filter(|&&r| r > 0).count();
    if relevant_count == 0 {
        return None;
    }
    let mut hits = 0usize;
    let mut sum = 0.0;
    for (i, d) in retrieved.iter().enumerate() {
        if qrels.get(d.as_str()).is_some_and(|&r| r > 0) {
            hits += 1;
            sum += hits as f64 / (i + 1) as f64;
        }
    }
    Some(sum / relevant_count as f64)
}

/// nDCG@k with TREC `(2^rel - 1)` gain and `1 / log2(rank+1)` discount.
///
/// IDCG is computed on the actual graded qrels truncated to k, not
/// `[4, 4, ...]`. Skip queries with empty graded sets.
#[must_use]
pub fn ndcg_at_k(retrieved: &[String], qrels: &Qrels, k: usize) -> Option<f64> {
    if !qrels.values().any(|&r| r > 0) {
        return None;
    }
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| {
            let rel = f64::from(qrels.get(d.as_str()).copied().unwrap_or(0));
            (2f64.powf(rel) - 1.0) / ((i + 2) as f64).log2()
        })
        .sum();
    let mut graded: Vec<u8> = qrels.values().copied().filter(|&r| r > 0).collect();
    graded.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = graded
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| (2f64.powf(f64::from(rel)) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 {
        return None;
    }
    Some(dcg / idcg)
}

/// Aggregated metric across queries. Skipped queries (those that
/// returned `None`) are excluded from the mean and counted in
/// `n_skipped`.
///
/// **`mean` is `0.0` when `n_evaluated == 0`.** Consumers must check
/// `n_evaluated > 0` before treating `mean` as a metric — a sentinel
/// `NaN` was rejected because it complicates JSON encoding and many
/// downstream tools (CSV, plotters) treat NaN inconsistently.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Aggregated {
    pub mean: f64,
    pub n_evaluated: usize,
    pub n_skipped: usize,
}

#[must_use]
pub fn aggregate(values: impl IntoIterator<Item = Option<f64>>) -> Aggregated {
    let mut sum = 0.0;
    let mut n = 0usize;
    let mut skipped = 0usize;
    for v in values {
        match v {
            Some(x) => {
                sum += x;
                n += 1;
            }
            None => skipped += 1,
        }
    }
    Aggregated {
        mean: if n > 0 { sum / n as f64 } else { 0.0 },
        n_evaluated: n,
        n_skipped: skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qrels(pairs: &[(&str, u8)]) -> Qrels {
        pairs.iter().map(|(d, r)| ((*d).to_string(), *r)).collect()
    }

    fn retrieved(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn recall_full_match_at_k() {
        let r = retrieved(&["a", "b", "c"]);
        let q = qrels(&[("a", 1), ("b", 1), ("c", 1)]);
        assert_eq!(recall_at_k(&r, &q, 3), Some(1.0));
        assert_eq!(recall_at_k(&r, &q, 5), Some(1.0));
    }

    #[test]
    fn recall_caps_at_k_over_relevant() {
        // |relevant|=4, k=2, retrieved=2 of them ⇒ 2/4 = 0.5
        let r = retrieved(&["a", "b", "x", "y"]);
        let q = qrels(&[("a", 1), ("b", 1), ("c", 1), ("d", 1)]);
        assert_eq!(recall_at_k(&r, &q, 2), Some(0.5));
    }

    #[test]
    fn recall_skipped_when_no_relevant() {
        let r = retrieved(&["a", "b"]);
        let q = qrels(&[]);
        assert_eq!(recall_at_k(&r, &q, 5), None);
    }

    #[test]
    fn mrr_first_relevant_at_position_3() {
        let r = retrieved(&["x", "y", "a"]);
        let q = qrels(&[("a", 2), ("b", 1)]);
        assert!((reciprocal_rank(&r, &q).unwrap() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_zero_when_no_relevant_retrieved() {
        let r = retrieved(&["x", "y", "z"]);
        let q = qrels(&[("a", 1)]);
        assert_eq!(reciprocal_rank(&r, &q), Some(0.0));
    }

    #[test]
    fn map_uses_total_relevant_denominator() {
        // Retrieved a (rel) at 1, c (rel) at 3. Relevant total = 3 (a,b,c).
        // AP = (1/1 + 2/3) / 3 = (1.0 + 0.6667) / 3 = 0.5556
        let r = retrieved(&["a", "x", "c", "y"]);
        let q = qrels(&[("a", 1), ("b", 1), ("c", 1)]);
        let ap = average_precision(&r, &q).unwrap();
        assert!((ap - (1.0 + 2.0 / 3.0) / 3.0).abs() < 1e-9);
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // Graded qrels {a:3, b:2, c:1}. Perfect ranking puts them in that order.
        let r = retrieved(&["a", "b", "c"]);
        let q = qrels(&[("a", 3), ("b", 2), ("c", 1)]);
        let n = ndcg_at_k(&r, &q, 3).unwrap();
        assert!((n - 1.0).abs() < 1e-9, "expected 1.0, got {n}");
    }

    #[test]
    fn ndcg_reverse_ranking_is_lower() {
        let perfect = retrieved(&["a", "b", "c"]);
        let reversed = retrieved(&["c", "b", "a"]);
        let q = qrels(&[("a", 3), ("b", 2), ("c", 1)]);
        let np = ndcg_at_k(&perfect, &q, 3).unwrap();
        let nr = ndcg_at_k(&reversed, &q, 3).unwrap();
        assert!(np > nr, "perfect ({np}) should beat reversed ({nr})");
        assert!(nr < 1.0);
    }

    #[test]
    fn ndcg_skipped_when_no_relevant() {
        let r = retrieved(&["a", "b"]);
        let q = qrels(&[("a", 0), ("b", 0)]);
        assert_eq!(ndcg_at_k(&r, &q, 5), None);
    }

    #[test]
    fn aggregate_counts_skipped_separately() {
        let values = vec![Some(1.0), None, Some(0.5), None, Some(0.0)];
        let agg = aggregate(values);
        assert_eq!(agg.n_evaluated, 3);
        assert_eq!(agg.n_skipped, 2);
        assert!((agg.mean - 0.5).abs() < 1e-9);
    }

    #[test]
    fn aggregate_zero_evaluated_returns_zero_mean() {
        let agg = aggregate([None, None]);
        assert_eq!(agg.n_evaluated, 0);
        assert_eq!(agg.n_skipped, 2);
        assert!(agg.mean.abs() < f64::EPSILON);
    }
}
