#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use std::path::PathBuf;
use tracing::{error, info};
use sift_core::{Config, ScanOptions, SiftResult};
use sift_server::WatchDaemon;

pub fn run(config: &Config, path: Option<PathBuf>, debounce: u64) -> SiftResult<()> {
    let watch_path = path.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    println!(
        "Watching {} for changes...",
        watch_path.display().to_string().cyan()
    );

    let config = config.clone();
    let daemon = WatchDaemon::new(vec![watch_path], debounce);

    daemon
        .run(move |changed_paths| {
            let paths: Vec<PathBuf> = changed_paths;
            info!(count = paths.len(), "Files changed");

            // Re-index changed files
            let options = ScanOptions {
                paths: paths
                    .iter()
                    .filter_map(|p| p.parent().map(|pp| pp.to_path_buf()))
                    .collect(),
                ..Default::default()
            };

            match crate::pipeline::open_engine(&config) {
                Ok((engine, metadata)) => {
                    #[cfg(feature = "embeddings")]
                    let embedder = crate::pipeline::load_embedder(None);
                    #[cfg(feature = "embeddings")]
                    let embedder_ref = embedder.as_ref().map(|e| e as &dyn sift_core::Embedder);
                    #[cfg(not(feature = "embeddings"))]
                    let embedder_ref: Option<&dyn sift_core::Embedder> = None;
                    #[cfg(feature = "vision")]
                    let vision_embedder = crate::pipeline::load_vision_embedder();
                    match crate::pipeline::run_scan_pipeline(
                        &config,
                        &options,
                        &engine,
                        &metadata,
                        embedder_ref,
                        #[cfg(feature = "vision")]
                        vision_embedder.as_ref(),
                        false,
                    ) {
                        Ok(stats) => {
                            if stats.indexed > 0 {
                                info!(
                                    indexed = stats.indexed,
                                    chunks = stats.chunks,
                                    "Re-indexed files"
                                );
                            }
                        }
                        Err(e) => {
                            error!("Re-index error: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Engine error: {}", e);
                }
            }
        })
        .map_err(sift_core::SiftError::Other)?;

    Ok(())
}
