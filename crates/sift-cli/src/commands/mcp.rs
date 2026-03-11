use sift_core::{Config, SiftResult};

/// Run the MCP server on stdio.
///
/// This starts the sift MCP server which communicates via JSON-RPC 2.0
/// over stdin/stdout. All logging goes to stderr.
pub fn run(config: &Config) -> SiftResult<()> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        sift_core::SiftError::Other(anyhow::anyhow!("Failed to start async runtime: {}", e))
    })?;

    rt.block_on(async {
        sift_mcp::run_stdio_server(config.clone())
            .await
            .map_err(sift_core::SiftError::Other)
    })
}
