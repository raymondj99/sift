//! `retrieval-lab` — single-cell runner, Phase 0 hypothesis suite, and
//! research sweeps.
//!
//! Subcommand summary:
//!
//! - `validate <corpus>` — load + structural-validate a corpus JSON.
//!   Cheap; no model needed. Useful as a CI smoke test.
//! - `run` — execute a single configuration and emit a result JSON.
//! - `report <run.json>` — render a result JSON as a human-readable
//!   Markdown summary (aggregate + per-category + latency).
//! - `phase0` — run the seven hypothesis-validation experiments. Stops
//!   at the first failure with a categorized message.
//! - `research` — finer-grained sweeps that complement Phase 0:
//!   alpha-curve at 0.0..=1.0 step 0.1, search-mode comparison
//!   (hybrid / vector-only / keyword-only), top-k saturation curve.

use clap::{Parser, Subcommand, ValueEnum};
use sift_core::SearchMode;
use sift_embed::models::NOMIC_EMBED_TEXT_V1_5;
use sift_retrieval_lab::corpus::RetrievalCorpus;
use sift_retrieval_lab::runner::{
    run_cell, CellConfig, CellResult, DEFAULT_FETCH_K_MULTIPLIER, DEFAULT_WARMUP_QUERIES,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "retrieval-lab",
    about = "Reproducible IR-metric harness for sift retrieval.",
    version
)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Load + structurally validate a retrieval corpus JSON. Exits 0 on
    /// success, non-zero with a categorized error on any defect (schema
    /// mismatch, duplicate ids, dangling qrels, etc.).
    Validate {
        #[arg(long, default_value = "evals/codingmem/codingmem-retrieval.json")]
        corpus: PathBuf,
    },
    /// Run a single configuration and emit a result JSON.
    Run {
        #[arg(long, default_value = "evals/codingmem/codingmem-retrieval.json")]
        corpus: PathBuf,
        #[arg(long, default_value_t = 0.7)]
        alpha: f32,
        #[arg(long, default_value_t = 768)]
        dim: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, value_enum, default_value_t = ModeArg::Hybrid)]
        mode: ModeArg,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Render a result JSON as a Markdown summary.
    Report {
        run: PathBuf,
        /// Optional output file; defaults to stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the seven Phase 0 hypothesis checks. Stops at the first
    /// failure with a categorized message.
    Phase0 {
        #[arg(long, default_value = "evals/codingmem/codingmem-retrieval.json")]
        corpus: PathBuf,
        #[arg(long, default_value = "target/retrieval-lab/phase0")]
        out_dir: PathBuf,
    },
    /// Finer-grained sweeps for research / characterization. Writes
    /// per-cell JSONs and a summary Markdown into `out_dir`.
    Research {
        #[arg(long, default_value = "evals/codingmem/codingmem-retrieval.json")]
        corpus: PathBuf,
        #[arg(long, default_value = "target/retrieval-lab/research")]
        out_dir: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModeArg {
    Hybrid,
    Vector,
    Keyword,
}

impl From<ModeArg> for SearchMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Hybrid => SearchMode::Hybrid,
            ModeArg::Vector => SearchMode::VectorOnly,
            ModeArg::Keyword => SearchMode::KeywordOnly,
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let args = Args::parse();
    match args.cmd {
        Cmd::Validate { corpus } => cmd_validate(&corpus),
        Cmd::Run {
            corpus,
            alpha,
            dim,
            top_k,
            mode,
            out,
        } => cmd_run(&corpus, alpha, dim, top_k, mode.into(), out.as_deref()),
        Cmd::Report { run, out } => cmd_report(&run, out.as_deref()),
        Cmd::Phase0 { corpus, out_dir } => cmd_phase0(&corpus, &out_dir),
        Cmd::Research { corpus, out_dir } => cmd_research(&corpus, &out_dir),
    }
}

// ---------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------

fn cmd_validate(path: &Path) -> anyhow::Result<()> {
    let corpus = RetrievalCorpus::load(path)?;
    println!("ok                   : {}", path.display());
    println!("schema_version       : {}", corpus.schema_version);
    println!("name                 : {}", corpus.name);
    println!("source_version       : {}", corpus.source_version);
    println!("documents            : {}", corpus.documents.len());
    println!("queries              : {}", corpus.queries.len());
    println!(
        "negative queries     : {}",
        corpus.queries.iter().filter(|q| q.negative).count()
    );
    println!("categories           : {}", corpus.categories().join(", "));
    println!("blake3               : {}", corpus.content_hash());
    Ok(())
}

// ---------------------------------------------------------------------
// run
// ---------------------------------------------------------------------

fn cmd_run(
    corpus_path: &Path,
    alpha: f32,
    dim: usize,
    top_k: usize,
    mode: SearchMode,
    out: Option<&Path>,
) -> anyhow::Result<()> {
    let corpus = RetrievalCorpus::load(corpus_path)?;
    let cfg = CellConfig {
        model: &NOMIC_EMBED_TEXT_V1_5,
        dim,
        alpha,
        top_k,
        fetch_k_multiplier: DEFAULT_FETCH_K_MULTIPLIER,
        warmup_queries: DEFAULT_WARMUP_QUERIES,
        mode,
    };
    let started = Instant::now();
    let result = run_cell(&corpus, &cfg)?;
    let wall = started.elapsed();
    print_summary(&result, wall);
    if let Some(path) = out {
        write_json(&result, path)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------
// report
// ---------------------------------------------------------------------

fn cmd_report(run_path: &Path, out: Option<&Path>) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(run_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", run_path.display()))?;
    let result: CellResult = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", run_path.display()))?;
    if result.schema_version != sift_retrieval_lab::runner::CELL_RESULT_SCHEMA_VERSION {
        anyhow::bail!(
            "result schema version mismatch at {}: file is v{}, lab expects v{}. \
             Re-run the cell with the current binary.",
            run_path.display(),
            result.schema_version,
            sift_retrieval_lab::runner::CELL_RESULT_SCHEMA_VERSION
        );
    }
    let md = render_markdown(&result);
    match out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, &md)?;
            println!("wrote {}", p.display());
        }
        None => print!("{md}"),
    }
    Ok(())
}

fn render_markdown(r: &CellResult) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "# retrieval-lab cell result");
    let _ = writeln!(s);
    let _ = writeln!(s, "## Configuration");
    let _ = writeln!(s);
    let _ = writeln!(s, "| Field | Value |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| model | `{}` |", r.config.model_name);
    let _ = writeln!(
        s,
        "| dim | {} of {} |",
        r.config.dim_used, r.config.model_dimensions
    );
    let _ = writeln!(s, "| alpha | {:.3} |", r.config.alpha);
    let _ = writeln!(s, "| mode | `{}` |", r.config.mode);
    let _ = writeln!(s, "| top_k | {} |", r.config.top_k);
    let _ = writeln!(s, "| fetch_k | {} |", r.config.fetch_k);
    let _ = writeln!(s, "| warmup_queries | {} |", r.config.warmup_queries);
    let _ = writeln!(s);

    let _ = writeln!(s, "## Corpus");
    let _ = writeln!(s);
    let _ = writeln!(s, "| Field | Value |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| name | {} |", r.corpus.name);
    let _ = writeln!(s, "| source version | `{}` |", r.corpus.source_version);
    let _ = writeln!(s, "| documents | {} |", r.corpus.n_documents);
    let _ = writeln!(s, "| queries | {} |", r.corpus.n_queries);
    let _ = writeln!(s, "| negative queries | {} |", r.corpus.n_negative_queries);
    let _ = writeln!(s, "| blake3 | `{}` |", &r.corpus.corpus_blake3[..16]);
    let _ = writeln!(s);

    let _ = writeln!(s, "## Aggregate metrics");
    let _ = writeln!(s);
    let _ = writeln!(s, "| Metric | Mean | n_evaluated | n_skipped |");
    let _ = writeln!(s, "|---|---|---|---|");
    let row = |label: &str, m: &sift_retrieval_lab::metrics::Aggregated| {
        format!(
            "| {label} | {:.4} | {} | {} |\n",
            m.mean, m.n_evaluated, m.n_skipped
        )
    };
    s.push_str(&row(
        &format!("recall@{}", r.config.top_k),
        &r.aggregate.recall_at_k,
    ));
    s.push_str(&row(
        &format!("nDCG@{}", r.config.top_k),
        &r.aggregate.ndcg_at_k,
    ));
    s.push_str(&row("MRR", &r.aggregate.mrr));
    s.push_str(&row("MAP", &r.aggregate.map));
    let _ = writeln!(s);

    if !r.per_category.is_empty() {
        let _ = writeln!(s, "## Per-category metrics");
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| Category | n | recall@{k} | nDCG@{k} | MRR | MAP |",
            k = r.config.top_k
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|");
        for c in &r.per_category {
            let _ = writeln!(
                s,
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |",
                c.category,
                c.n_queries,
                c.recall_at_k.mean,
                c.ndcg_at_k.mean,
                c.mrr.mean,
                c.map.mean
            );
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "## Latency");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "p50 = {:.3}ms · p95 = {:.3}ms · p99 = {:.3}ms · mean = {:.3}ms · outliers {}/{} (warmup {})",
        r.latency.p50_ms,
        r.latency.p95_ms,
        r.latency.p99_ms,
        r.latency.mean_ms,
        r.latency.n_outliers,
        r.latency.n_samples,
        r.latency.n_warmup
    );
    let _ = writeln!(s);

    let _ = writeln!(s, "## Environment");
    let _ = writeln!(s);
    let _ = writeln!(s, "- target: `{}`", r.environment.target);
    let _ = writeln!(s, "- rustc: `{}`", r.environment.rustc);
    let _ = writeln!(
        s,
        "- git: `{}{}`",
        r.environment.git_sha.as_deref().unwrap_or("unknown"),
        if r.environment.git_dirty {
            " (dirty)"
        } else {
            ""
        }
    );
    let _ = writeln!(s, "- timestamp: `{}`", r.environment.timestamp_utc);
    s
}

// ---------------------------------------------------------------------
// phase0
// ---------------------------------------------------------------------

fn cmd_phase0(corpus_path: &Path, out_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let corpus = RetrievalCorpus::load(corpus_path)?;

    println!("================================================================");
    println!("Phase 0: hypothesis validation");
    println!("corpus  : {} (v{})", corpus.name, corpus.source_version);
    println!("docs    : {}", corpus.documents.len());
    println!("queries : {}", corpus.queries.len());
    println!("blake3  : {}", &corpus.content_hash()[..16]);
    println!("================================================================");

    // E1: qrels signal
    let hist = corpus.n_relevant_histogram();
    let n_zero = hist.iter().filter(|(_, n)| *n == 0).count();
    let n_total = hist.len();
    let zero_frac = n_zero as f64 / n_total.max(1) as f64;
    let median = {
        let mut counts: Vec<usize> = hist.iter().map(|(_, n)| *n).collect();
        counts.sort_unstable();
        counts.get(counts.len() / 2).copied().unwrap_or(0)
    };
    println!(
        "[E1] qrels signal: {n_zero}/{n_total} queries have 0 relevant ({:.1}%); median |relevant| = {median}",
        zero_frac * 100.0
    );
    if zero_frac > 0.40 {
        anyhow::bail!(
            "[E1 KILL] >40% of queries have empty qrels; oracle heuristic too strict. \
             Improve the transform (e.g. LLM-judge graded relevance) or expand the source corpus."
        );
    }
    if median >= 10 {
        anyhow::bail!(
            "[E1 KILL] median |relevant| >= 10; oracle heuristic too lenient. recall@10 will saturate trivially."
        );
    }

    // E2: current-state baseline
    let cfg_base = base_cell_config(0.7, 768, 10, SearchMode::Hybrid);
    let started = Instant::now();
    let baseline = run_cell(&corpus, &cfg_base)?;
    let wall_base = started.elapsed();
    write_json(&baseline, &out_dir.join("e2-baseline.json"))?;
    println!(
        "[E2] baseline (alpha=0.7, dim=768): recall@10={:.4} ndcg@10={:.4} mrr={:.4} map={:.4}  ({:.1}s)",
        baseline.aggregate.recall_at_k.mean,
        baseline.aggregate.ndcg_at_k.mean,
        baseline.aggregate.mrr.mean,
        baseline.aggregate.map.mean,
        wall_base.as_secs_f64()
    );
    if baseline.aggregate.recall_at_k.mean >= 0.95 {
        anyhow::bail!("[E2 KILL] recall@10 >= 0.95 — corpus has no headroom.");
    }
    if baseline.aggregate.recall_at_k.mean <= 0.30 {
        anyhow::bail!(
            "[E2 KILL] recall@10 <= 0.30 — foundational chunking/embedding/qrels problem."
        );
    }

    // E3: alpha sweep
    let alphas = [0.0_f32, 0.3, 0.7, 1.0];
    let mut alpha_results: Vec<(f32, CellResult)> = Vec::new();
    for &a in &alphas {
        if (a - 0.7).abs() < f32::EPSILON {
            alpha_results.push((a, baseline.clone()));
            continue;
        }
        let cfg = CellConfig {
            alpha: a,
            ..cfg_base.clone()
        };
        let r = run_cell(&corpus, &cfg)?;
        write_json(&r, &out_dir.join(format!("e3-alpha-{a:.1}.json")))?;
        println!(
            "[E3] alpha={a:.1}: recall@10={:.4} ndcg@10={:.4}",
            r.aggregate.recall_at_k.mean, r.aggregate.ndcg_at_k.mean
        );
        alpha_results.push((a, r));
    }
    let alpha_delta = max_minus_min(
        alpha_results
            .iter()
            .map(|(_, r)| r.aggregate.recall_at_k.mean),
    );
    println!("[E3] Δrecall@10 across alpha = {alpha_delta:.4}");
    if alpha_delta < 0.02 {
        anyhow::bail!("[E3 KILL] Δrecall@10 across alpha < 0.02 — fusion-weight axis is dead.");
    }

    // E4: matryoshka dim sweep (informational only)
    let dims = [128_usize, 256, 768];
    let mut dim_results: Vec<(usize, CellResult)> = Vec::new();
    for &d in &dims {
        if d == 768 {
            dim_results.push((d, baseline.clone()));
            continue;
        }
        let cfg = CellConfig {
            dim: d,
            ..cfg_base.clone()
        };
        let r = run_cell(&corpus, &cfg)?;
        write_json(&r, &out_dir.join(format!("e4-dim-{d}.json")))?;
        println!(
            "[E4] dim={d}: recall@10={:.4} ndcg@10={:.4}",
            r.aggregate.recall_at_k.mean, r.aggregate.ndcg_at_k.mean
        );
        dim_results.push((d, r));
    }
    let dim_delta = max_minus_min(
        dim_results
            .iter()
            .map(|(_, r)| r.aggregate.recall_at_k.mean),
    );
    println!("[E4] Δrecall@10 across dim = {dim_delta:.4}");
    if dim_delta < 0.005 {
        eprintln!(
            "[E4 NOTE] Δrecall@10 across dim < 0.005; matryoshka axis carries little signal."
        );
    }

    // E5: statistical power
    let per_query: Vec<f64> = baseline
        .per_query
        .iter()
        .filter_map(|p| p.ndcg_at_k)
        .collect();
    let n = per_query.len();
    if n < 5 {
        anyhow::bail!("[E5 KILL] only {n} evaluable nDCG samples.");
    }
    let half_width = bootstrap_ci_half_width(&per_query, 1000, 0.95);
    println!("[E5] paired bootstrap 95% CI half-width on nDCG@10 (n={n}): ±{half_width:.4}");
    if half_width > 0.10 {
        eprintln!(
            "[E5 NOTE] CI half-width > 0.10 — only large effects can be claimed significant."
        );
    }

    // E6: aggregate-metric stability across runs
    println!("[E6] running twice and asserting aggregate-metric stability...");
    let r1 = run_cell(&corpus, &cfg_base)?;
    let r2 = run_cell(&corpus, &cfg_base)?;
    let differing = r1
        .per_query
        .iter()
        .zip(r2.per_query.iter())
        .filter(|(a, b)| {
            let aids: Vec<&str> = a.hits.iter().map(|h| h.doc_id.as_str()).collect();
            let bids: Vec<&str> = b.hits.iter().map(|h| h.doc_id.as_str()).collect();
            aids != bids
        })
        .count();
    let drift_recall = (r1.aggregate.recall_at_k.mean - r2.aggregate.recall_at_k.mean).abs();
    let drift_ndcg = (r1.aggregate.ndcg_at_k.mean - r2.aggregate.ndcg_at_k.mean).abs();
    let drift_mrr = (r1.aggregate.mrr.mean - r2.aggregate.mrr.mean).abs();
    let drift_map = (r1.aggregate.map.mean - r2.aggregate.map.mean).abs();
    let max_drift = drift_recall.max(drift_ndcg).max(drift_mrr).max(drift_map);
    println!(
        "[E6] differing per-query rankings: {differing}/{} (expected with multi-threaded ONNX)",
        r1.per_query.len()
    );
    println!(
        "[E6] aggregate drift: recall={drift_recall:.5} ndcg={drift_ndcg:.5} mrr={drift_mrr:.5} map={drift_map:.5}"
    );
    const STABILITY_THRESHOLD: f64 = 0.0005;
    if max_drift > STABILITY_THRESHOLD {
        anyhow::bail!(
            "[E6 KILL] aggregate metric drift {max_drift:.5} > {STABILITY_THRESHOLD:.5} — runs not metric-stable."
        );
    }

    // E7: latency-boundary sanity (informational)
    let p50 = baseline.latency.p50_ms;
    let p99 = baseline.latency.p99_ms;
    let ratio = if p50 > 0.0 { p99 / p50 } else { f64::INFINITY };
    println!(
        "[E7] latency: p50={p50:.2}ms p99={p99:.2}ms (ratio={ratio:.2}) outliers={}/{}",
        baseline.latency.n_outliers, baseline.latency.n_samples
    );
    if ratio > 10.0 {
        eprintln!(
            "[E7 NOTE] p99/p50 > 10 — latency tail noisy on this small corpus; quality metrics fine."
        );
    }

    println!("================================================================");
    println!("Phase 0 complete: every kill-criterion passed.");
    println!("Result JSONs in {}", out_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------
// research
// ---------------------------------------------------------------------

/// Fine-grained sweeps that complement Phase 0.
///
/// 1. **Alpha curve**: 11 cells at α ∈ {0.0, 0.1, .., 1.0}. Characterizes
///    the recall-vs-nDCG tradeoff that Phase 0 E3 only sketched.
/// 2. **Search-mode comparison**: hybrid vs vector-only vs keyword-only
///    at α = 0.7. Quantifies how much the hybrid blend buys vs the pure
///    backends.
/// 3. **Top-k saturation**: k ∈ {1, 3, 5, 10, 20} at α = 0.7. Shows
///    where recall plateaus on this corpus.
fn cmd_research(corpus_path: &Path, out_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let corpus = RetrievalCorpus::load(corpus_path)?;

    println!("================================================================");
    println!("Research sweeps");
    println!("corpus  : {} (v{})", corpus.name, corpus.source_version);
    println!("blake3  : {}", &corpus.content_hash()[..16]);
    println!("================================================================");

    // 1. Alpha curve.
    println!("\n## Alpha curve (dim=768, top_k=10, mode=hybrid)\n");
    println!("| alpha | recall@10 | nDCG@10 | MRR    | MAP    |");
    println!("|-------|-----------|---------|--------|--------|");
    let alphas: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
    let mut alpha_curve: Vec<(f32, CellResult)> = Vec::new();
    for a in &alphas {
        let cfg = base_cell_config(*a, 768, 10, SearchMode::Hybrid);
        let r = run_cell(&corpus, &cfg)?;
        write_json(&r, &out_dir.join(format!("alpha-{a:.1}.json")))?;
        println!(
            "| {a:.1}   | {:.4}    | {:.4}  | {:.4} | {:.4} |",
            r.aggregate.recall_at_k.mean,
            r.aggregate.ndcg_at_k.mean,
            r.aggregate.mrr.mean,
            r.aggregate.map.mean
        );
        alpha_curve.push((*a, r));
    }
    let best_recall = alpha_curve
        .iter()
        .max_by(|a, b| {
            a.1.aggregate
                .recall_at_k
                .mean
                .partial_cmp(&b.1.aggregate.recall_at_k.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let best_ndcg = alpha_curve
        .iter()
        .max_by(|a, b| {
            a.1.aggregate
                .ndcg_at_k
                .mean
                .partial_cmp(&b.1.aggregate.ndcg_at_k.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    println!(
        "\nbest recall@10 at α={:.1} ({:.4}); best nDCG@10 at α={:.1} ({:.4})",
        best_recall.0,
        best_recall.1.aggregate.recall_at_k.mean,
        best_ndcg.0,
        best_ndcg.1.aggregate.ndcg_at_k.mean
    );

    // 2. Search-mode comparison.
    println!("\n## Search-mode comparison (alpha=0.7, dim=768, top_k=10)\n");
    println!("| mode    | recall@10 | nDCG@10 | MRR    | MAP    |");
    println!("|---------|-----------|---------|--------|--------|");
    for mode in [
        SearchMode::Hybrid,
        SearchMode::VectorOnly,
        SearchMode::KeywordOnly,
    ] {
        let cfg = base_cell_config(0.7, 768, 10, mode);
        let r = run_cell(&corpus, &cfg)?;
        write_json(&r, &out_dir.join(format!("mode-{}.json", mode.as_str())))?;
        println!(
            "| {:7} | {:.4}    | {:.4}  | {:.4} | {:.4} |",
            mode.as_str(),
            r.aggregate.recall_at_k.mean,
            r.aggregate.ndcg_at_k.mean,
            r.aggregate.mrr.mean,
            r.aggregate.map.mean
        );
    }

    // 3. Top-k saturation.
    println!("\n## Top-k saturation (alpha=0.7, dim=768, mode=hybrid)\n");
    println!("| top_k | recall   | nDCG     | MRR    | MAP    |");
    println!("|-------|----------|----------|--------|--------|");
    for &k in &[1_usize, 3, 5, 10, 20] {
        let cfg = base_cell_config(0.7, 768, k, SearchMode::Hybrid);
        let r = run_cell(&corpus, &cfg)?;
        write_json(&r, &out_dir.join(format!("topk-{k}.json")))?;
        println!(
            "| {k:5} | {:.4}   | {:.4}   | {:.4} | {:.4} |",
            r.aggregate.recall_at_k.mean,
            r.aggregate.ndcg_at_k.mean,
            r.aggregate.mrr.mean,
            r.aggregate.map.mean
        );
    }

    println!("\nResearch sweeps written to {}", out_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn base_cell_config(alpha: f32, dim: usize, top_k: usize, mode: SearchMode) -> CellConfig {
    CellConfig {
        model: &NOMIC_EMBED_TEXT_V1_5,
        dim,
        alpha,
        top_k,
        fetch_k_multiplier: DEFAULT_FETCH_K_MULTIPLIER,
        warmup_queries: DEFAULT_WARMUP_QUERIES,
        mode,
    }
}

fn max_minus_min<I: IntoIterator<Item = f64>>(iter: I) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in iter {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if max.is_finite() && min.is_finite() {
        max - min
    } else {
        0.0
    }
}

/// Bootstrap 95% CI half-width for the mean of a sample. Deterministic
/// 64-bit LCG so Phase 0 is reproducible without an extra crate
/// dependency.
///
/// LCG constants are Knuth's MMIX (multiplier `6364136223846793005`,
/// increment `1442695040888963407`). Seed is the SplitMix64 golden-
/// gamma `0x9E37_79B9_7F4A_7C15`. The high bits of an LCG output are
/// far better than the low bits, hence `state >> 33` for index
/// extraction. Good enough for a bootstrap; not cryptographic.
fn bootstrap_ci_half_width(samples: &[f64], n_resamples: usize, ci: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = samples.len();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut means: Vec<f64> = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = ((state >> 33) as usize) % n;
            sum += samples[idx];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = 1.0 - ci;
    let lo_idx = ((alpha / 2.0) * n_resamples as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64) as usize;
    let lo = means[lo_idx.min(n_resamples - 1)];
    let hi = means[hi_idx.min(n_resamples - 1)];
    (hi - lo) / 2.0
}

fn print_summary(r: &CellResult, wall: std::time::Duration) {
    println!(
        "model={} dim={} alpha={:.2} mode={} corpus={} (n_docs={}, n_queries={})",
        r.config.model_name,
        r.config.dim_used,
        r.config.alpha,
        r.config.mode,
        r.corpus.name,
        r.corpus.n_documents,
        r.corpus.n_queries
    );
    println!(
        "recall@{} = {:.4}  (n_eval={}, n_skip={})",
        r.config.top_k,
        r.aggregate.recall_at_k.mean,
        r.aggregate.recall_at_k.n_evaluated,
        r.aggregate.recall_at_k.n_skipped
    );
    println!(
        "ndcg@{}   = {:.4}  (n_eval={}, n_skip={})",
        r.config.top_k,
        r.aggregate.ndcg_at_k.mean,
        r.aggregate.ndcg_at_k.n_evaluated,
        r.aggregate.ndcg_at_k.n_skipped
    );
    println!("mrr      = {:.4}", r.aggregate.mrr.mean);
    println!("map      = {:.4}", r.aggregate.map.mean);
    println!(
        "latency  : p50={:.2}ms p95={:.2}ms p99={:.2}ms  outliers={}/{}",
        r.latency.p50_ms,
        r.latency.p95_ms,
        r.latency.p99_ms,
        r.latency.n_outliers,
        r.latency.n_samples
    );
    println!("wall     : {:.2}s", wall.as_secs_f64());
}

fn write_json(r: &CellResult, path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(r)?;
    std::fs::write(path, json)?;
    Ok(())
}
