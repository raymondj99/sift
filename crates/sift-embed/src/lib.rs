//! Embedding engine backed by ONNX Runtime.
//!
//! Manages model downloads, ONNX inference sessions, and an embedding cache.
//! Supports text embeddings via [`OnnxEmbedder`] and optionally vision embeddings
//! via [`VisionEmbedder`] (behind the `vision` feature flag).
//!
//! Requires the ONNX Runtime shared library at runtime (`ORT_DYLIB_PATH`).

pub mod error;

#[cfg(feature = "sqlite")]
pub mod cache;
#[cfg(not(feature = "sqlite"))]
pub mod json_cache;
pub mod models;
pub mod onnx;
pub mod traits;
#[cfg(feature = "vision")]
pub mod vision;

#[cfg(feature = "sqlite")]
pub use cache::EmbeddingCache;
#[cfg(not(feature = "sqlite"))]
pub use json_cache::EmbeddingCache;
pub use models::{optimal_batch_size, ModelManager, QuantizationType};
pub use onnx::OnnxEmbedder;
pub use error::EmbeddingError;
pub use traits::Embedder;
#[cfg(feature = "vision")]
pub use vision::VisionEmbedder;
