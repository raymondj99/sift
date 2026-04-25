//! Latency collector with linear-interpolated percentiles.
//!
//! Replaces the buggy `LatencyCollector::percentile` at
//! `crates/sift-cli/src/commands/bench.rs:276` which computes
//! `idx = (pct * len / 100).min(len - 1)` — for `len = 100, pct = 99` that
//! returns `sorted[99]` (the maximum). The fix is the standard
//! linear-interpolation formula: `rank = pct/100 * (n - 1)`.
//!
//! The cli-side bug-fix back-port is out of scope for this crate; once the
//! lab proves the implementation in production, the cli module can adopt
//! the same code.

use std::time::Duration;

#[derive(Debug, Default)]
pub struct LatencyCollector {
    samples: Vec<Duration>,
    sorted: Option<Vec<Duration>>,
}

impl LatencyCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, d: Duration) {
        self.samples.push(d);
        self.sorted = None;
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Linear-interpolated percentile, `pct` in `[0, 100]`.
    ///
    /// Returns `None` when there are no samples. Sorts on first call after
    /// each `record`; subsequent percentile queries reuse the sorted cache.
    pub fn percentile(&mut self, pct: f64) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let sorted = self.sorted.get_or_insert_with(|| {
            let mut v = self.samples.clone();
            v.sort();
            v
        });
        let n = sorted.len();
        if n == 1 {
            return Some(sorted[0]);
        }
        let rank = (pct.clamp(0.0, 100.0) / 100.0) * (n - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        if lo == hi {
            return Some(sorted[lo]);
        }
        let frac = rank - lo as f64;
        let lo_ns = sorted[lo].as_nanos() as f64;
        let hi_ns = sorted[hi].as_nanos() as f64;
        let interp = lo_ns + frac * (hi_ns - lo_ns);
        Some(Duration::from_nanos(interp as u64))
    }

    pub fn p50(&mut self) -> Option<Duration> {
        self.percentile(50.0)
    }
    pub fn p95(&mut self) -> Option<Duration> {
        self.percentile(95.0)
    }
    pub fn p99(&mut self) -> Option<Duration> {
        self.percentile(99.0)
    }

    pub fn mean(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let total_ns: u128 = self.samples.iter().map(|d| d.as_nanos()).sum();
        let mean_ns = total_ns / self.samples.len() as u128;
        // `Duration::from_nanos` takes `u64`. The mean of any realistic
        // latency distribution fits comfortably (`u64::MAX` ns is ~584
        // years), but make the truncation explicit rather than silent so
        // a pathological input fails loudly via the saturating bound.
        let clamped = u64::try_from(mean_ns).unwrap_or(u64::MAX);
        Some(Duration::from_nanos(clamped))
    }

    /// Tukey-fence outlier count (samples > Q3 + 1.5·IQR). Reported, not
    /// dropped — the whole point of p99 is to catch tail latency.
    pub fn outlier_count(&mut self) -> usize {
        if self.samples.len() < 4 {
            return 0;
        }
        let q1 = self.percentile(25.0).map_or(0.0, |d| d.as_nanos() as f64);
        let q3 = self.percentile(75.0).map_or(0.0, |d| d.as_nanos() as f64);
        let iqr = q3 - q1;
        let fence = q3 + 1.5 * iqr;
        self.samples
            .iter()
            .filter(|d| d.as_nanos() as f64 > fence)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector_from_micros(micros: &[u64]) -> LatencyCollector {
        let mut c = LatencyCollector::new();
        for &m in micros {
            c.record(Duration::from_micros(m));
        }
        c
    }

    #[test]
    fn p99_of_100_samples_is_not_max() {
        // The cli-side bug returns sorted[99] (the max). Linear
        // interpolation on 100 samples for p99 yields rank = 0.99 * 99 = 98.01,
        // which interpolates between sorted[98] and sorted[99]:
        // sorted[98] = 99us, sorted[99] = 100us, frac = 0.01 ⇒ 99.01us.
        // Use as_nanos() to preserve sub-microsecond precision (as_micros
        // truncates 99_010 ns to 99us).
        let samples: Vec<u64> = (1..=100).collect();
        let mut c = collector_from_micros(&samples);
        let p99_us = c.p99().unwrap().as_nanos() as f64 / 1000.0;
        assert!(
            (p99_us - 99.01).abs() < 0.01,
            "expected 99.01us, got {p99_us}us"
        );
    }

    #[test]
    fn p50_of_odd_samples_is_middle() {
        let mut c = collector_from_micros(&[10, 20, 30, 40, 50]);
        // rank = 0.5 * 4 = 2.0 ⇒ sorted[2] = 30
        assert_eq!(c.p50().unwrap(), Duration::from_micros(30));
    }

    #[test]
    fn empty_collector_returns_none() {
        let mut c = LatencyCollector::new();
        assert_eq!(c.p99(), None);
        assert_eq!(c.mean(), None);
    }

    #[test]
    fn cached_sort_invalidated_on_record() {
        let mut c = collector_from_micros(&[100, 200, 300]);
        let _ = c.p99();
        c.record(Duration::from_micros(1));
        // After recording, sorted = [1, 100, 200, 300]. p50 with linear
        // interpolation: rank = 0.5 * 3 = 1.5 ⇒ between sorted[1]=100 and
        // sorted[2]=200 ⇒ 150us. (Pre-cached state would have used the old
        // 3-element sort; this test asserts the cache was invalidated.)
        assert_eq!(c.p50().unwrap(), Duration::from_micros(150));
    }
}
