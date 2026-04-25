#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use sift_core::{Config, SiftResult};
use sift_server::routes::{create_router, AppState};
use std::sync::Arc;

pub async fn run(config: &Config, host: &str, port: u16) -> SiftResult<()> {
    let (engine, metadata) = crate::pipeline::open_engine(config)?;

    #[cfg(feature = "embeddings")]
    let embedder: Option<Box<dyn sift_core::Embedder>> =
        crate::pipeline::load_embedder(config, None)
            .map(|e| Box::new(e) as Box<dyn sift_core::Embedder>);
    #[cfg(not(feature = "embeddings"))]
    let embedder: Option<Box<dyn sift_core::Embedder>> = None;

    let has_embedder = embedder.is_some();
    let memory_dir = config.index_dir().ok().map(|d| d.join("memory"));
    let memory = memory_dir.as_ref().and_then(|dir| {
        sift_memory::MemoryStore::open(dir)
            .map(|s| Arc::new(s.with_decay_rate(config.memory.decay_rate)))
            .ok()
    });
    let state = Arc::new(AppState {
        engine: std::sync::RwLock::new(Arc::new(engine)),
        metadata: std::sync::RwLock::new(Arc::new(metadata)),
        embedder,
        memory_dir,
        memory,
        consolidation_config: Some(sift_memory::ConsolidationConfig::from(&config.memory)),
    });
    let app = create_router(state);

    let addr = format!("{}:{}", host, port);
    println!(
        "Starting sift server on {}",
        format!("http://{}", addr).cyan()
    );
    println!("  {} GET /health", "".green());
    println!(
        "  {} GET /api/search?q=<query>{}",
        "".green(),
        if has_embedder {
            " (hybrid search)"
        } else {
            " (keyword-only, no embedding model)"
        }
    );
    println!("  {} GET /api/status", "".green());

    sift_server::serve(&addr, app)
        .await
        .map_err(|e| sift_core::SiftError::Runtime(e.to_string()))?;

    Ok(())
}
