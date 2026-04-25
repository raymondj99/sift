use sift_core::{Config, SiftResult};

/// Run the MCP server on stdio.
///
/// This starts the sift MCP server which communicates via JSON-RPC 2.0
/// over stdin/stdout. All logging goes to stderr.
pub fn run(config: &Config) -> SiftResult<()> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        sift_core::SiftError::Runtime(format!("Failed to start async runtime: {e}"))
    })?;

    rt.block_on(async {
        sift_mcp::run_stdio_server(config.clone())
            .await
            .map_err(|e| sift_core::SiftError::Runtime(e.to_string()))
    })
}

/// Run the MCP server over Streamable HTTP.
///
/// Starts an axum server at `http://<host>:<port>/mcp`. All stdio-accessible
/// tools are exposed. When `bearer_token` is set, every request must carry
/// `Authorization: Bearer <token>`.
#[cfg(feature = "mcp-http")]
pub fn run_http(
    config: &Config,
    host: &str,
    port: u16,
    bearer_token: Option<String>,
    stateful: bool,
) -> SiftResult<()> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        sift_core::SiftError::Runtime(format!("Failed to start async runtime: {e}"))
    })?;

    rt.block_on(async {
        let opts = sift_mcp::http::HttpServerOptions {
            host: host.to_string(),
            port,
            bearer_token,
            stateful,
            ..Default::default()
        };

        // Wire Ctrl-C into the cancellation token so the server shuts down
        // gracefully on SIGINT.
        let ct = opts.cancellation_token.clone();
        let server = tokio::spawn(sift_mcp::http::run_http_server(config.clone(), opts));

        tokio::select! {
            r = server => r
                .map_err(|e| sift_core::SiftError::Runtime(format!("http server task: {e}")))?
                .map_err(|e| sift_core::SiftError::Runtime(e.to_string())),
            _ = tokio::signal::ctrl_c() => {
                ct.cancel();
                Ok(())
            }
        }
    })
}
