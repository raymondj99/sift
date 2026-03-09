#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
use crate::{pipeline, OutputFormat};
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::{Config, ScanOptions, SiftResult};

pub fn run(
    config: &Config,
    options: &ScanOptions,
    model: Option<&str>,
    format: &OutputFormat,
    quiet: bool,
) -> SiftResult<()> {
    if options.dry_run && !quiet && *format == OutputFormat::Human {
        println!("{}", "Dry run — no files will be indexed".yellow());
    }

    let (engine, metadata) = pipeline::open_engine(config)?;

    #[cfg(feature = "embeddings")]
    let embedder = pipeline::load_embedder(model);
    #[cfg(feature = "embeddings")]
    let embedder_ref = embedder.as_ref().map(|e| e as &dyn sift_core::Embedder);
    #[cfg(not(feature = "embeddings"))]
    let embedder_ref: Option<&dyn sift_core::Embedder> = {
        let _ = model;
        None
    };

    #[cfg(feature = "vision")]
    let vision_embedder = pipeline::load_vision_embedder();

    let stats = pipeline::run_scan_pipeline(
        config,
        options,
        &engine,
        &metadata,
        embedder_ref,
        #[cfg(feature = "vision")]
        vision_embedder.as_ref(),
        quiet,
    )?;

    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::json!({
                    "discovered": stats.discovered,
                    "indexed": stats.indexed,
                    "skipped": stats.skipped,
                    "chunks": stats.chunks,
                    "errors": stats.errors,
                    "cache_hits": stats.cache_hits,
                    "file_types": stats.file_types,
                })
            );
        }
        OutputFormat::Csv => {
            println!("discovered,indexed,skipped,chunks,errors,cache_hits");
            println!(
                "{},{},{},{},{},{}",
                stats.discovered,
                stats.indexed,
                stats.skipped,
                stats.chunks,
                stats.errors,
                stats.cache_hits,
            );
        }
        OutputFormat::Human => {
            if !quiet {
                if options.dry_run {
                    println!(
                        "Would index {} files ({} already up to date)",
                        stats.discovered.to_string().green(),
                        stats.skipped.to_string().dimmed(),
                    );
                } else {
                    println!(
                        "Indexed {} files ({} chunks) — {} skipped, {} errors",
                        stats.indexed.to_string().green(),
                        stats.chunks.to_string().green(),
                        stats.skipped.to_string().dimmed(),
                        if stats.errors > 0 {
                            stats.errors.to_string().red().to_string()
                        } else {
                            stats.errors.to_string().dimmed().to_string()
                        },
                    );

                    if stats.cache_hits > 0 {
                        println!(
                            "  {} embedding cache hits",
                            stats.cache_hits.to_string().dimmed()
                        );
                    }

                    if !stats.file_types.is_empty() {
                        let mut types: Vec<_> = stats.file_types.iter().collect();
                        types.sort_by(|a, b| b.1.cmp(a.1));
                        let type_summary: Vec<String> =
                            types.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
                        println!("  {}", type_summary.join("  ").dimmed());
                    }
                }
            }
        }
    }

    Ok(())
}
