use crate::{output, pipeline, OutputFormat};
#[cfg(feature = "embeddings")]
use sift_core::Embedder;
use sift_core::{Config, SearchMode, SearchOptions, SiftResult};
use tracing::info;

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

    // Over-fetch when filters are active so post-filtering doesn't
    // reduce the result count below max_results.
    let has_filters = options.file_type.is_some()
        || options.path_glob.is_some()
        || options.after.is_some()
        || options.threshold > 0.0;
    let fetch_k = if has_filters {
        options.max_results * 5
    } else {
        options.max_results
    };

    let mut results = engine.search(&query_vector, &options.query, fetch_k, effective_mode)?;

    // Apply threshold filter
    results.retain(|r| r.score >= options.threshold);

    // Apply file type filter
    if let Some(ref ft) = options.file_type {
        results.retain(|r| r.file_type == *ft);
    }

    // Apply path filter (substring match, or glob if pattern contains wildcards)
    if let Some(ref pattern) = options.path_glob {
        let has_glob_chars =
            pattern.contains('*') || pattern.contains('?') || pattern.contains('[');
        if has_glob_chars {
            if let Ok(glob) = globset::GlobBuilder::new(pattern)
                .literal_separator(false)
                .build()
            {
                let matcher = glob.compile_matcher();
                results.retain(|r| {
                    let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                    matcher.is_match(path)
                });
            } else {
                results.retain(|r| {
                    let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                    path.contains(pattern.as_str())
                });
            }
        } else {
            // Plain string: use substring match on the file path
            results.retain(|r| {
                let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                path.contains(pattern.as_str())
            });
        }
    }

    // Apply --after date filter
    if let Some(after_ts) = options.after {
        let allowed = metadata.uris_modified_after(after_ts)?;
        results.retain(|r| allowed.contains(&r.uri));
    }

    // Truncate to requested max after all filters have been applied.
    results.truncate(options.max_results);

    output::format_search_results(&results, format, options.context);

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
