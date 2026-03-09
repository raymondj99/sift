use crate::{output, pipeline, OutputFormat};
use tracing::info;
#[cfg(feature = "embeddings")]
use sift_core::Embedder;
use sift_core::{Config, SearchMode, SearchOptions, SiftResult};

pub fn run(
    config: &Config,
    options: &SearchOptions,
    format: &OutputFormat,
    open: bool,
) -> SiftResult<()> {
    let (engine, metadata) = pipeline::open_engine(config)?;

    #[cfg(feature = "embeddings")]
    let embedder = pipeline::load_embedder(None);

    // Embed query for vector search, or fall back to keyword-only
    #[cfg(feature = "embeddings")]
    let (query_vector, effective_mode) = match (&embedder, options.mode) {
        (Some(emb), mode) => {
            let vec = emb.embed(&format!("search_query: {}", &options.query))?;
            (vec, mode)
        }
        (None, SearchMode::VectorOnly) => {
            info!("No embedding model available. Falling back to keyword search.");
            (vec![0.0f32; 768], SearchMode::KeywordOnly)
        }
        (None, SearchMode::Hybrid) => {
            info!("No embedding model available. Using keyword-only search.");
            (vec![0.0f32; 768], SearchMode::KeywordOnly)
        }
        (None, mode) => (vec![0.0f32; 768], mode),
    };

    #[cfg(not(feature = "embeddings"))]
    let (query_vector, effective_mode) = {
        if options.mode == SearchMode::VectorOnly {
            info!("Embeddings feature not enabled. Falling back to keyword search.");
        }
        (vec![0.0f32; 768], SearchMode::KeywordOnly)
    };

    let mut results = engine.search(
        &query_vector,
        &options.query,
        options.max_results,
        effective_mode,
    )?;

    // Apply threshold filter
    results.retain(|r| r.score >= options.threshold);

    // Apply file type filter
    if let Some(ref ft) = options.file_type {
        results.retain(|r| r.file_type == *ft);
    }

    // Apply path glob filter
    if let Some(ref glob) = options.path_glob {
        results.retain(|r| r.uri.contains(glob));
    }

    // Apply --after date filter
    if let Some(after_ts) = options.after {
        let allowed = metadata.uris_modified_after(after_ts)?;
        results.retain(|r| allowed.contains(&r.uri));
    }

    output::print_search_results(&results, format, options.context);

    // Open top result in default application
    if open {
        if let Some(top) = results.first() {
            if let Some(path) = top.uri.strip_prefix("file://") {
                let cmd = if cfg!(target_os = "macos") {
                    "open"
                } else if cfg!(target_os = "windows") {
                    "start"
                } else {
                    "xdg-open"
                };
                let _ = std::process::Command::new(cmd).arg(path).spawn();
            }
        }
    }

    Ok(())
}
