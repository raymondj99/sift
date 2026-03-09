#[cfg(not(feature = "fancy"))]
use crate::color_stub::*;
#[cfg(feature = "fancy")]
use colored::*;
use std::sync::Arc;
use sift_core::{Config, SiftResult};
use sift_server::routes::{create_router, AppState};

pub async fn run(config: &Config, host: &str, port: u16) -> SiftResult<()> {
    let (engine, metadata) = crate::pipeline::open_engine(config)?;

    #[cfg(feature = "embeddings")]
    let embedder: Option<Box<dyn sift_core::Embedder>> =
        crate::pipeline::load_embedder(None).map(|e| Box::new(e) as Box<dyn sift_core::Embedder>);
    #[cfg(not(feature = "embeddings"))]
    let embedder: Option<Box<dyn sift_core::Embedder>> = None;

    let has_embedder = embedder.is_some();
    let state = Arc::new(AppState {
        engine,
        metadata,
        embedder,
    });
    let app = create_router(state);

    let addr = format!("{}:{}", host, port);
    println!(
        "Starting vx server on {}",
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
        .map_err(sift_core::SiftError::Other)?;

    Ok(())
}
