//! Embedder bootstrap shared by `sift-cli` and `sift-mcp`.
//!
//! Resolves the configured ONNX text-embedding model, downloads it on demand
//! (when `model_override` is unset), and returns an [`OnnxEmbedder`] ready
//! for use. Returns `None` (with `info!`-level logging) on any "soft" miss
//! — model not downloaded, runtime not supported on this platform, or
//! ONNX session load failure — so callers fall back to keyword-only search
//! at the boundary instead of inside the embedder.

use crate::{
    models::{get_model, NOMIC_EMBED_TEXT_V1_5},
    ModelManager, OnnxEmbedder,
};
use sift_core::Config;
use tracing::{info, warn};

/// Load the configured text embedder, or `None` if unavailable.
///
/// `model_override`:
/// - `None` → use `config.default.model`
/// - `Some(name)` → look up by name in the registry; if `name` is an
///   absolute directory path, treat it as a literal model directory
///   (using the default model spec).
pub fn load_embedder(config: &Config, model_override: Option<&str>) -> Option<OnnxEmbedder> {
    let manager = ModelManager::new().ok()?;
    manager.init_ort_env_with_override(config.default.ort_dylib_path.as_deref());

    let (model_dir, model_def) = if let Some(name) = model_override {
        let path = std::path::Path::new(name);
        if path.is_absolute() && path.is_dir() {
            (path.to_path_buf(), &NOMIC_EMBED_TEXT_V1_5)
        } else {
            let Some(model_def) = get_model(name) else {
                info!(
                    "Unknown embedding model '{}' — using keyword-only mode. \
                     Run `sift models list` to see supported models.",
                    name
                );
                return None;
            };
            if !model_def.is_download_supported() {
                info!(
                    "Embedding model '{}' requires runtime '{}' — using keyword-only mode. {}",
                    model_def.name,
                    model_def.runtime.as_str(),
                    model_def.notes
                );
                return None;
            }
            let Some(model_dir) = manager.downloaded_model_dir(model_def) else {
                info!(
                    "Model '{}' not downloaded — using keyword-only mode. \
                     Run `sift models download {}` first.",
                    model_def.name, model_def.name
                );
                return None;
            };
            (model_dir, model_def)
        }
    } else {
        let model_def = get_model(&config.default.model).unwrap_or(&NOMIC_EMBED_TEXT_V1_5);
        if !model_def.is_download_supported() {
            info!(
                "Default embedding model '{}' requires runtime '{}' — using keyword-only mode. {}",
                model_def.name,
                model_def.runtime.as_str(),
                model_def.notes
            );
            return None;
        }
        let Some(model_dir) = manager.downloaded_model_dir(model_def) else {
            info!(
                "Embedding model not downloaded — using keyword-only mode. \
                 Run `sift models download {}` for semantic search.",
                model_def.name
            );
            return None;
        };
        (model_dir, model_def)
    };

    match OnnxEmbedder::load_model(&model_dir, model_def) {
        Ok(embedder) => {
            info!("Loaded embedding model: {}", model_def.name);
            Some(embedder)
        }
        Err(e) => {
            warn!(
                "Failed to load embedding model: {}. Falling back to keyword-only.",
                e
            );
            None
        }
    }
}
