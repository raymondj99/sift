//! Streamable HTTP transport for the sift MCP server.
//!
//! Spawns an axum server exposing the MCP endpoint at `POST/GET/DELETE /mcp`
//! per the MCP 2025-11-25 Streamable HTTP spec. A fresh `SiftMcpServer` is
//! built per session by the `service_factory` closure, so multiple clients
//! can share the same HTTP endpoint while each getting their own cache.
//!
//! Bearer token authentication is optional but recommended whenever binding
//! to any address other than loopback.

use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
    Router,
};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use sift_core::Config;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::SiftMcpServer;

/// Options for the HTTP MCP transport.
#[derive(Debug, Clone)]
pub struct HttpServerOptions {
    /// Host to bind to (default `127.0.0.1`).
    pub host: String,
    /// Port to bind to (default `7821`).
    pub port: u16,
    /// Optional bearer token. When set, clients must send
    /// `Authorization: Bearer <token>` on every request.
    pub bearer_token: Option<String>,
    /// Stateful session mode (default `true`). Required for multi-turn
    /// conversations that rely on SSE reconnect; disable only for simple
    /// stateless request/response deployments.
    pub stateful: bool,
    /// Cancellation token for graceful shutdown. A child token is wired into
    /// the HTTP service so terminating it closes all in-flight sessions.
    pub cancellation_token: CancellationToken,
}

impl Default for HttpServerOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 7821,
            bearer_token: None,
            stateful: true,
            cancellation_token: CancellationToken::new(),
        }
    }
}

/// Start the HTTP MCP server. Blocks until the listener is cancelled.
pub async fn run_http_server(config: Config, opts: HttpServerOptions) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{}:{}", opts.host, opts.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid host/port '{}:{}': {e}", opts.host, opts.port))?;

    if !addr.ip().is_loopback() && opts.bearer_token.is_none() {
        warn!(
            %addr,
            "binding MCP HTTP server to a non-loopback address without a bearer token \
             — anyone who can reach this port can call every sift tool"
        );
    }

    let ct = opts.cancellation_token.child_token();
    let service_config = StreamableHttpServerConfig {
        stateful_mode: opts.stateful,
        cancellation_token: ct.clone(),
        ..Default::default()
    };

    let service: StreamableHttpService<SiftMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            {
                let config = config.clone();
                move || {
                    SiftMcpServer::new(config.clone())
                        .map_err(|e| std::io::Error::other(format!("{e}")))
                }
            },
            Arc::new(LocalSessionManager::default()),
            service_config,
        );

    let mut router = Router::new()
        .nest_service("/mcp", service)
        .route(
            "/health",
            axum::routing::get(|| async { (StatusCode::OK, "ok") }),
        )
        .layer(TraceLayer::new_for_http());

    if let Some(token) = opts.bearer_token.clone() {
        let token = Arc::new(token);
        router = router.layer(axum::middleware::from_fn(
            move |req: Request, next: Next| {
                let token = Arc::clone(&token);
                async move { bearer_auth(token.as_str(), req, next).await }
            },
        ));
    }

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    info!("sift MCP HTTP server listening on http://{addr}/mcp");

    let shutdown = {
        let ct = opts.cancellation_token.clone();
        async move { ct.cancelled_owned().await }
    };
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| anyhow::anyhow!("HTTP server error: {e}"))?;
    Ok(())
}

/// Require an `Authorization: Bearer <token>` header matching `expected`.
///
/// Applied only when a bearer token is configured; `/health` is also gated
/// because it sits on the same router (fine — consumers can probe with the
/// same token).
async fn bearer_auth(expected: &str, req: Request, next: Next) -> Result<Response, StatusCode> {
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim);

    match presented {
        Some(tok) if constant_time_eq(tok.as_bytes(), expected.as_bytes()) => {
            Ok(next.run(req).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Constant-time byte comparison. Length-independent so attackers can't
/// short-circuit by brute-forcing the length first.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // Still walk the shorter slice so the timing difference is bounded by
        // the rejected slice length, not the secret length.
        let n = a.len().min(b.len());
        let mut diff: u8 = 1;
        for i in 0..n {
            diff |= a[i] ^ b[i];
        }
        return diff == 0 && a.len() == b.len();
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn constant_time_eq_matches() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"abcd", b"abc"));
        assert!(constant_time_eq(b"", b""));
    }

    /// Build a minimal router with the bearer-auth layer for testing. We
    /// skip `SiftMcpServer` here because constructing it requires a real
    /// index directory; auth is what we actually want to exercise.
    fn auth_router(token: &str) -> Router {
        let tok = Arc::new(token.to_string());
        Router::new()
            .route(
                "/health",
                axum::routing::get(|| async { (StatusCode::OK, "ok") }),
            )
            .layer(axum::middleware::from_fn(
                move |req: Request, next: Next| {
                    let tok = Arc::clone(&tok);
                    async move { bearer_auth(tok.as_str(), req, next).await }
                },
            ))
    }

    async fn send(router: Router, req: axum::http::Request<Body>) -> axum::http::Response<Body> {
        router.oneshot(req).await.unwrap()
    }

    #[tokio::test]
    async fn bearer_auth_rejects_missing_header() {
        let router = auth_router("s3cret");
        let req = axum::http::Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = send(router, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    async fn bearer_auth_rejects_wrong_token() {
        let router = auth_router("s3cret");
        let req = axum::http::Request::builder()
            .uri("/health")
            .header("Authorization", "Bearer wrong")
            .body(Body::empty())
            .unwrap();
        let resp = send(router, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    async fn bearer_auth_rejects_malformed_header() {
        let router = auth_router("s3cret");
        let req = axum::http::Request::builder()
            .uri("/health")
            .header("Authorization", "s3cret") // missing "Bearer " prefix
            .body(Body::empty())
            .unwrap();
        let resp = send(router, req).await;
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    async fn bearer_auth_accepts_valid_token() {
        let router = auth_router("s3cret");
        let req = axum::http::Request::builder()
            .uri("/health")
            .header("Authorization", "Bearer s3cret")
            .body(Body::empty())
            .unwrap();
        let resp = send(router, req).await;
        assert_eq!(resp.status(), 200);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"ok");
    }
}
