//! `codingmem-transform`: derive a retrieval corpus + qrels from
//! `evals/codingmem/scenarios.json`.
//!
//! Output is committed to disk so the lab and downstream consumers share
//! ground truth. Re-run when `scenarios.json` changes.
//!
//! The transform is heuristic — verbatim word-boundary substring match of
//! the question's `answer` against event text. The maintainer is expected
//! to spot-check the output before commit; the binary prints summary
//! statistics for that purpose.

use clap::Parser;
use sift_retrieval_lab::transform::transform_codingmem;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "codingmem-transform",
    about = "Transform CodingMem scenarios.json into a retrieval corpus + qrels.",
    version
)]
struct Args {
    /// Path to the source CodingMem scenarios JSON.
    #[arg(long, default_value = "evals/codingmem/scenarios.json")]
    input: PathBuf,

    /// Path to write the resulting retrieval corpus JSON.
    #[arg(long, default_value = "evals/codingmem/codingmem-retrieval.json")]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let corpus = transform_codingmem(&args.input)?;

    println!("source version       : {}", corpus.source_version);
    println!("documents            : {}", corpus.documents.len());
    println!("queries              : {}", corpus.queries.len());

    let n_negative = corpus.queries.iter().filter(|q| q.negative).count();
    println!("negative queries     : {n_negative}");

    let hist = corpus.n_relevant_histogram();
    let n_zero = hist.iter().filter(|(_, n)| *n == 0).count();
    let n_one = hist.iter().filter(|(_, n)| *n == 1).count();
    let n_multi = hist.iter().filter(|(_, n)| *n > 1).count();
    println!("queries with 0 relev : {n_zero} (incl. {n_negative} negatives)");
    println!("queries with 1 relev : {n_one}");
    println!("queries with 2+ relev: {n_multi}");

    let total_relevant: usize = hist.iter().map(|(_, n)| n).sum();
    let mean = if !hist.is_empty() {
        total_relevant as f64 / hist.len() as f64
    } else {
        0.0
    };
    println!("mean |relevant|      : {mean:.2}");

    corpus.save(&args.output)?;
    println!("wrote                : {}", args.output.display());
    Ok(())
}
