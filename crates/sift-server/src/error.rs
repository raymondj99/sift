//! HTTP error type for route handlers.
//!
//! Replaces the per-handler `(StatusCode, String)` tuples with a typed
//! [`AppError`] that implements [`IntoResponse`]. Handlers return
//! `Result<T, AppError>` and use `?` with `.map_err(AppError::internal)` (or
//! whichever constructor matches the failure mode) instead of writing the
//! status code at every call site.

use axum::{http::StatusCode, response::IntoResponse, response::Response};

/// Typed handler error, mapped to an HTTP status by [`IntoResponse`].
#[derive(Debug)]
pub struct AppError {
    pub status: StatusCode,
    pub message: String,
}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
        }
    }

    pub fn internal<E: std::fmt::Display>(err: E) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, self.message).into_response()
    }
}
