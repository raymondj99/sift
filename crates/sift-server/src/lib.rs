//! HTTP API server and filesystem watcher.
//!
//! Provides an Axum-based REST API for searching the index (`/api/search`)
//! and checking status (`/api/status`), plus a [`WatchDaemon`] that monitors
//! the filesystem for changes and triggers re-indexing.

pub mod routes;
pub mod watch;

pub use routes::create_router;
pub use watch::WatchDaemon;

/// Bind to `addr` and serve the given router.
pub async fn serve(addr: &str, app: axum::Router) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
