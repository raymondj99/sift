use crate::{output, pipeline, OutputFormat, StartupProfiler};
#[cfg(feature = "embeddings")]
use sift_core::Embedder;
use sift_core::{Config, SearchMode, SearchOptions, SearchResult, SiftResult};
use tracing::info;

pub fn run(
    config: &Config,
    options: &SearchOptions,
    format: &OutputFormat,
    open: bool,
    profiler: &mut StartupProfiler,
) -> SiftResult<()> {
    // Try daemon first for faster response (avoids model loading)
    #[cfg(unix)]
    if let Some(results) = try_daemon_search(options) {
        profiler.checkpoint("daemon_search");
        output::format_search_results(&results, format, options.context);
        if open {
            open_top_result(&results);
        }
        return Ok(());
    }

    // Load engine and embedder concurrently — they are independent.
    #[cfg(feature = "embeddings")]
    let (engine_result, embedder) = std::thread::scope(|s| {
        let engine_handle = s.spawn(|| pipeline::open_engine(config));
        let embedder = pipeline::load_embedder(config, None);
        (
            engine_handle.join().expect("engine thread panicked"),
            embedder,
        )
    });
    #[cfg(not(feature = "embeddings"))]
    let engine_result = pipeline::open_engine(config);

    let (engine, metadata) = engine_result?;
    profiler.checkpoint("store_open");

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
    profiler.checkpoint("model_load");

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
    profiler.checkpoint("search_execute");

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
        open_top_result(&results);
    }

    Ok(())
}

fn open_top_result(results: &[SearchResult]) {
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

/// Try to run search via the daemon's HTTP API.
#[cfg(unix)]
fn try_daemon_search(options: &SearchOptions) -> Option<Vec<SearchResult>> {
    if !crate::daemon_client::is_available() {
        return None;
    }

    let mode = match options.mode {
        SearchMode::Hybrid => "hybrid",
        SearchMode::VectorOnly => "vector",
        SearchMode::KeywordOnly => "keyword",
    };

    // Over-fetch when filters are active
    let has_filters = options.file_type.is_some()
        || options.path_glob.is_some()
        || options.after.is_some()
        || options.threshold > 0.0;
    let limit = if has_filters {
        options.max_results * 5
    } else {
        options.max_results
    };

    // Build query URL — file_type filter is handled server-side
    let mut url = format!(
        "/api/search?q={}&limit={}&mode={}",
        urlenc(&options.query),
        limit,
        mode,
    );
    if let Some(ref ft) = options.file_type {
        use std::fmt::Write;
        let _ = write!(url, "&type={}", urlenc(ft));
    }

    let body = crate::daemon_client::get(&url)?;
    let resp: serde_json::Value = serde_json::from_str(&body).ok()?;
    let items = resp.get("results")?.as_array()?;

    let mut results: Vec<SearchResult> = items
        .iter()
        .filter_map(|item| {
            Some(SearchResult {
                uri: item.get("uri")?.as_str()?.to_string(),
                text: item.get("text")?.as_str()?.to_string(),
                score: item.get("score")?.as_f64()? as f32,
                chunk_index: item.get("chunk_index")?.as_u64()? as u32,
                content_type: serde_json::from_value(item.get("content_type")?.clone())
                    .unwrap_or(sift_core::ContentType::Text),
                file_type: item.get("file_type")?.as_str()?.to_string(),
                title: item.get("title").and_then(|v| v.as_str()).map(String::from),
                byte_range: item.get("byte_range").and_then(|v| {
                    let arr = v.as_array()?;
                    Some((arr.first()?.as_u64()?, arr.get(1)?.as_u64()?))
                }),
            })
        })
        .collect();

    // Apply client-side filters that the API doesn't handle
    results.retain(|r| r.score >= options.threshold);

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
            }
        } else {
            results.retain(|r| {
                let path = r.uri.strip_prefix("file://").unwrap_or(&r.uri);
                path.contains(pattern.as_str())
            });
        }
    }

    results.truncate(options.max_results);
    Some(results)
}

/// Minimal percent-encoding for query parameters.
fn urlenc(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "+")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('#', "%23")
}
