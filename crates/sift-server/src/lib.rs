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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_re_exports() {
        // Verify that the public re-exports are accessible
        let _: fn(Vec<std::path::PathBuf>, u64) -> WatchDaemon = WatchDaemon::new;
        // create_router is re-exported
        let _: fn(std::sync::Arc<routes::AppState>) -> axum::Router = create_router;
    }

    #[tokio::test]
    async fn test_serve_invalid_address() {
        let app = axum::Router::new();
        // Binding to an invalid address should fail
        let result = serve("invalid-not-an-address:99999", app).await;
        assert!(result.is_err());
    }
}
