use sift_core::{Config, SiftResult};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// ONNX Runtime version compatible with ort 2.0.0-rc.9.
const ORT_VERSION: &str = "1.20.1";

/// Library filename for the current platform.
#[cfg(target_os = "macos")]
const ORT_LIB_FILENAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "linux")]
const ORT_LIB_FILENAME: &str = "libonnxruntime.so";

#[cfg(target_os = "windows")]
const ORT_LIB_FILENAME: &str = "onnxruntime.dll";

/// Model quantization level, used to select the appropriate ONNX file variant.
///
/// Many model hubs publish FP16 and INT8 quantized variants alongside the
/// default FP32 model. Quantized models trade a small amount of accuracy for
/// significantly lower memory usage and faster inference -- especially useful
/// on edge devices and GPUs with limited VRAM.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QuantizationType {
    /// Full-precision 32-bit floating point (the default).
    #[default]
    FP32,
    /// Half-precision 16-bit floating point.
    FP16,
    /// 8-bit integer quantization.
    INT8,
}

/// Pooling strategy used to reduce per-token hidden states into a single
/// sentence-level embedding vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average all token embeddings weighted by the attention mask.
    MeanPooling,
    /// Use the hidden state of the `[CLS]` token (index 0).
    ClsToken,
}

/// Extended model specification with prefix, pooling, and Matryoshka metadata.
pub struct ModelSpec {
    pub name: &'static str,
    pub repo_id: &'static str,
    pub dimensions: usize,
    pub max_tokens: usize,
    /// Prefix prepended to search/query texts before embedding.
    pub search_prefix: &'static str,
    /// Prefix prepended to document/passage texts before embedding.
    pub document_prefix: &'static str,
    /// How the model pools token-level representations.
    pub pooling: PoolingStrategy,
    /// Matryoshka-supported truncated dimensions (largest first).
    /// Empty slice means the model does not support Matryoshka truncation.
    pub matryoshka_dims: &'static [usize],
}

/// Backwards-compatible alias so that existing code referencing `ModelDef`
/// continues to compile without changes.
pub type ModelDef = ModelSpec;

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

pub const NOMIC_EMBED_TEXT_V1_5: ModelSpec = ModelSpec {
    name: "nomic-embed-text-v1.5",
    repo_id: "nomic-ai/nomic-embed-text-v1.5",
    dimensions: 768,
    max_tokens: 8192,
    search_prefix: "search_query: ",
    document_prefix: "search_document: ",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[768, 512, 256, 128, 64],
};

pub const NOMIC_EMBED_TEXT_V2: ModelSpec = ModelSpec {
    name: "nomic-embed-text-v2",
    repo_id: "nomic-ai/nomic-embed-text-v1.5",
    dimensions: 768,
    max_tokens: 8192,
    search_prefix: "search_query: ",
    document_prefix: "search_document: ",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[768, 512, 256, 128],
};

pub const BGE_M3: ModelSpec = ModelSpec {
    name: "bge-m3",
    repo_id: "BAAI/bge-m3",
    dimensions: 1024,
    max_tokens: 8192,
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::ClsToken,
    matryoshka_dims: &[],
};

/// Nomic Embed Vision v1.5 — image embedding model.
/// Produces 768-dim vectors in the same embedding space as the text model,
/// enabling cross-modal search (search images with text queries).
pub const NOMIC_EMBED_VISION_V1_5: ModelSpec = ModelSpec {
    name: "nomic-embed-vision-v1.5",
    repo_id: "nomic-ai/nomic-embed-vision-v1.5",
    dimensions: 768,
    max_tokens: 0, // not applicable for vision models
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[],
};

/// All known model specs, for programmatic iteration and lookup.
const ALL_MODELS: &[&ModelSpec] = &[
    &NOMIC_EMBED_TEXT_V1_5,
    &NOMIC_EMBED_TEXT_V2,
    &BGE_M3,
    &NOMIC_EMBED_VISION_V1_5,
];

/// Look up a model specification by its short name (e.g. `"bge-m3"`).
pub fn get_model(name: &str) -> Option<&'static ModelSpec> {
    ALL_MODELS.iter().find(|m| m.name == name).copied()
}

/// Return a slice of all registered model specs.
pub fn all_models() -> &'static [&'static ModelSpec] {
    ALL_MODELS
}

impl ModelSpec {
    /// Return the ONNX filename for a given quantization level.
    ///
    /// By convention, quantized variants live alongside the full-precision
    /// model with a `_fp16` or `_int8` suffix before the `.onnx` extension.
    pub fn onnx_file_for_quant(&self, quant: QuantizationType) -> String {
        const BASE: &str = "model.onnx";
        match quant {
            QuantizationType::FP32 => BASE.to_string(),
            QuantizationType::FP16 => BASE.replace(".onnx", "_fp16.onnx"),
            QuantizationType::INT8 => BASE.replace(".onnx", "_int8.onnx"),
        }
    }
}

/// Estimate the optimal batch size for embedding given the model dimension and
/// available memory.
///
/// The heuristic accounts for both the output embedding vectors (dimension x 4
/// bytes per f32) and an approximation of the intermediate transformer
/// activations. The result is clamped to the range `[8, 256]`.
pub fn optimal_batch_size(dimension: usize, available_memory_mb: usize) -> usize {
    // Per-item cost: output vector + rough estimate of transient activations
    let per_item_bytes = (dimension * 4) + (512 * 8 * 2);
    // Use ~25% of available memory for the embedding batch
    let budget = available_memory_mb * 1024 * 1024 / 4;
    let batch = budget / per_item_bytes;
    batch.clamp(8, 256)
}

pub struct ModelManager {
    models_dir: PathBuf,
}

impl ModelManager {
    pub fn new() -> SiftResult<Self> {
        Ok(Self {
            models_dir: Config::models_dir()?,
        })
    }

    pub fn with_dir(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
    }

    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.model_dir(model_name).join("model.onnx")
    }

    pub fn tokenizer_path(&self, model_name: &str) -> PathBuf {
        self.model_dir(model_name).join("tokenizer.json")
    }

    pub fn is_downloaded(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists() && self.tokenizer_path(model_name).exists()
    }

    /// Check if a model's ONNX file exists (without requiring a tokenizer).
    /// Used for vision models that take pixel data rather than text.
    pub fn is_model_file_downloaded(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists()
    }

    /// Download model files from `HuggingFace`.
    pub fn download(&self, model_def: &ModelDef) -> SiftResult<()> {
        let dir = self.model_dir(model_def.name);
        std::fs::create_dir_all(&dir)?;

        let base_url = format!("https://huggingface.co/{}/resolve/main", model_def.repo_id);

        let files = ["model.onnx", "tokenizer.json"];

        for file in &files {
            let dest = dir.join(file);
            if dest.exists() {
                info!("{} already exists, skipping", file);
                continue;
            }

            let url = if *file == "model.onnx" {
                format!("{base_url}/onnx/model.onnx")
            } else {
                format!("{base_url}/{file}")
            };

            info!("Downloading {} from {}", file, url);

            let response = ureq::get(&url)
                .call()
                .map_err(|e| sift_core::SiftError::Model(format!("Download failed: {e}")))?;

            let mut bytes = Vec::new();
            response
                .into_reader()
                .read_to_end(&mut bytes)
                .map_err(|e| sift_core::SiftError::Model(format!("Download read failed: {e}")))?;

            std::fs::write(&dest, &bytes)?;
            info!("Saved {} ({} bytes)", file, bytes.len());
        }

        Ok(())
    }

    pub fn list_downloaded(&self) -> Vec<String> {
        let mut models = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        if self.is_downloaded(name) {
                            models.push(name.to_string());
                        }
                    }
                }
            }
        }
        models
    }

    pub fn ensure_model(&self, model_name: &str) -> SiftResult<PathBuf> {
        if !self.is_downloaded(model_name) {
            return Err(sift_core::SiftError::Model(format!(
                "Model '{model_name}' not found. Run `sift models download {model_name}` first."
            )));
        }
        Ok(self.model_dir(model_name))
    }

    /// Path where the ONNX Runtime shared library is stored.
    pub fn ort_lib_path(&self) -> PathBuf {
        self.models_dir.join("ort").join(ORT_LIB_FILENAME)
    }

    /// Check if the ONNX Runtime library has been downloaded.
    pub fn is_ort_downloaded(&self) -> bool {
        self.ort_lib_path().exists()
    }

    /// Download the ONNX Runtime shared library for the current platform.
    pub fn download_ort(&self) -> SiftResult<()> {
        let lib_path = self.ort_lib_path();
        if lib_path.exists() {
            info!("ONNX Runtime already downloaded");
            return Ok(());
        }

        let ort_dir = self.models_dir.join("ort");
        std::fs::create_dir_all(&ort_dir)?;

        let url = ort_download_url();
        info!("Downloading ONNX Runtime {} from {}", ORT_VERSION, url);

        let response = ureq::get(&url).call().map_err(|e| {
            sift_core::SiftError::Model(format!("ONNX Runtime download failed: {e}"))
        })?;

        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| sift_core::SiftError::Model(format!("Download read failed: {e}")))?;

        // The download is a .tgz archive — extract the library file
        extract_ort_lib(&bytes, &lib_path)?;

        info!(
            "ONNX Runtime {} saved to {}",
            ORT_VERSION,
            lib_path.display()
        );
        Ok(())
    }

    /// Set `ORT_DYLIB_PATH` if the runtime library exists in our managed directory
    /// and the env var is not already set.
    pub fn init_ort_env(&self) {
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            debug!("ORT_DYLIB_PATH already set, skipping auto-config");
            return;
        }
        let lib_path = self.ort_lib_path();
        if lib_path.exists() {
            debug!("Setting ORT_DYLIB_PATH to {}", lib_path.display());
            std::env::set_var("ORT_DYLIB_PATH", &lib_path);
        } else {
            warn!(
                "ONNX Runtime not found at {}. Run `sift models download` or set ORT_DYLIB_PATH.",
                lib_path.display()
            );
        }
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().expect("failed to resolve models directory")
    }
}

/// Build the ONNX Runtime download URL for the current platform.
fn ort_download_url() -> String {
    let (os, arch, ext) = if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        ("osx", "arm64", "tgz")
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        ("osx", "x86_64", "tgz")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        ("linux", "x64", "tgz")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        ("linux", "aarch64", "tgz")
    } else {
        // Fallback — will likely fail at download time with a clear HTTP error
        ("linux", "x64", "tgz")
    };

    format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-{os}-{arch}-{ORT_VERSION}.{ext}",
    )
}

/// Extract the ONNX Runtime shared library from a .tgz archive.
fn extract_ort_lib(archive_bytes: &[u8], dest: &Path) -> SiftResult<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let gz = GzDecoder::new(archive_bytes);
    let mut archive = Archive::new(gz);

    for entry in archive
        .entries()
        .map_err(|e| sift_core::SiftError::Model(format!("Failed to read archive: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| sift_core::SiftError::Model(format!("Archive entry error: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| sift_core::SiftError::Model(format!("Archive path error: {e}")))?;

        let path_str = path.to_string_lossy();
        // The library is at lib/libonnxruntime.so (or .dylib) inside the archive
        if path_str.contains(ORT_LIB_FILENAME) && !path_str.ends_with('/') {
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf)
                .map_err(|e| sift_core::SiftError::Model(format!("Extract read error: {e}")))?;
            std::fs::write(dest, &buf)?;
            return Ok(());
        }
    }

    Err(sift_core::SiftError::Model(format!(
        "{ORT_LIB_FILENAME} not found in ONNX Runtime archive"
    )))
}

/// Check if a path looks like a valid ONNX model directory.
pub fn validate_model_dir(dir: &Path) -> SiftResult<()> {
    if !dir.join("model.onnx").exists() {
        return Err(sift_core::SiftError::Model(format!(
            "model.onnx not found in {}",
            dir.display()
        )));
    }
    if !dir.join("tokenizer.json").exists() {
        return Err(sift_core::SiftError::Model(format!(
            "tokenizer.json not found in {}",
            dir.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // -----------------------------------------------------------------------
    // ModelSpec::onnx_file_for_quant
    // -----------------------------------------------------------------------

    #[test]
    fn onnx_file_for_quant_fp32() {
        assert_eq!(
            NOMIC_EMBED_TEXT_V1_5.onnx_file_for_quant(QuantizationType::FP32),
            "model.onnx"
        );
    }

    #[test]
    fn onnx_file_for_quant_fp16() {
        assert_eq!(
            BGE_M3.onnx_file_for_quant(QuantizationType::FP16),
            "model_fp16.onnx"
        );
    }

    #[test]
    fn onnx_file_for_quant_int8() {
        assert_eq!(
            NOMIC_EMBED_TEXT_V2.onnx_file_for_quant(QuantizationType::INT8),
            "model_int8.onnx"
        );
    }

    // -----------------------------------------------------------------------
    // get_model
    // -----------------------------------------------------------------------

    #[test]
    fn get_model_existing() {
        let m = get_model("bge-m3").expect("bge-m3 should exist");
        assert_eq!(m.name, "bge-m3");
        assert_eq!(m.dimensions, 1024);
    }

    #[test]
    fn get_model_nomic_v1_5() {
        let m = get_model("nomic-embed-text-v1.5").expect("should exist");
        assert_eq!(m.repo_id, "nomic-ai/nomic-embed-text-v1.5");
    }

    #[test]
    fn get_model_vision() {
        let m = get_model("nomic-embed-vision-v1.5").expect("should exist");
        assert_eq!(m.max_tokens, 0);
        assert_eq!(m.pooling, PoolingStrategy::MeanPooling);
    }

    #[test]
    fn get_model_unknown_returns_none() {
        assert!(get_model("does-not-exist").is_none());
    }

    // -----------------------------------------------------------------------
    // all_models
    // -----------------------------------------------------------------------

    #[test]
    fn all_models_returns_four() {
        let models = all_models();
        assert_eq!(models.len(), 4);
    }

    #[test]
    fn all_models_contains_expected_names() {
        let names: Vec<&str> = all_models().iter().map(|m| m.name).collect();
        assert!(names.contains(&"nomic-embed-text-v1.5"));
        assert!(names.contains(&"nomic-embed-text-v2"));
        assert!(names.contains(&"bge-m3"));
        assert!(names.contains(&"nomic-embed-vision-v1.5"));
    }

    // -----------------------------------------------------------------------
    // optimal_batch_size
    // -----------------------------------------------------------------------

    #[test]
    fn optimal_batch_size_clamps_to_min() {
        // With extremely large dimension and tiny memory, the calculated batch
        // should fall below 8 and get clamped up to the minimum.
        let bs = optimal_batch_size(1_000_000, 1);
        assert_eq!(bs, 8);
    }

    #[test]
    fn optimal_batch_size_clamps_to_max() {
        // Very high memory should clamp to 256
        let bs = optimal_batch_size(64, 100_000);
        assert_eq!(bs, 256);
    }

    #[test]
    fn optimal_batch_size_mid_range() {
        // With a moderate amount of memory and typical dimension, should be between 8 and 256
        let bs = optimal_batch_size(768, 256);
        assert!((8..=256).contains(&bs), "batch size {bs} out of range");
    }

    #[test]
    fn optimal_batch_size_large_dimension() {
        // Large dimension means more per-item cost
        let bs = optimal_batch_size(4096, 512);
        assert!((8..=256).contains(&bs), "batch size {bs} out of range");
    }

    // -----------------------------------------------------------------------
    // QuantizationType::default
    // -----------------------------------------------------------------------

    #[test]
    fn quantization_default_is_fp32() {
        assert_eq!(QuantizationType::default(), QuantizationType::FP32);
    }

    // -----------------------------------------------------------------------
    // ModelManager::with_dir and path methods
    // -----------------------------------------------------------------------

    #[test]
    fn model_manager_with_dir_paths() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());

        let model_dir = mgr.model_dir("test-model");
        assert_eq!(model_dir, tmp.path().join("test-model"));

        let model_path = mgr.model_path("test-model");
        assert_eq!(model_path, tmp.path().join("test-model").join("model.onnx"));

        let tokenizer_path = mgr.tokenizer_path("test-model");
        assert_eq!(
            tokenizer_path,
            tmp.path().join("test-model").join("tokenizer.json")
        );
    }

    // -----------------------------------------------------------------------
    // ModelManager::is_downloaded
    // -----------------------------------------------------------------------

    #[test]
    fn is_downloaded_false_when_empty() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(!mgr.is_downloaded("some-model"));
    }

    #[test]
    fn is_downloaded_false_when_only_model_exists() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("my-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(!mgr.is_downloaded("my-model"));
    }

    #[test]
    fn is_downloaded_false_when_only_tokenizer_exists() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("my-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(!mgr.is_downloaded("my-model"));
    }

    #[test]
    fn is_downloaded_true_when_both_exist() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("my-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(mgr.is_downloaded("my-model"));
    }

    // -----------------------------------------------------------------------
    // ModelManager::is_model_file_downloaded
    // -----------------------------------------------------------------------

    #[test]
    fn is_model_file_downloaded_false_when_missing() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(!mgr.is_model_file_downloaded("vision-model"));
    }

    #[test]
    fn is_model_file_downloaded_true_when_onnx_exists() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("vision-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();
        // No tokenizer needed for vision models
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(mgr.is_model_file_downloaded("vision-model"));
    }

    // -----------------------------------------------------------------------
    // ModelManager::list_downloaded
    // -----------------------------------------------------------------------

    #[test]
    fn list_downloaded_empty_dir() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(mgr.list_downloaded().is_empty());
    }

    #[test]
    fn list_downloaded_with_valid_model() {
        let tmp = tempdir().unwrap();

        // Create a fully downloaded model
        let model_dir = tmp.path().join("model-a");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"fake").unwrap();

        // Create a partially downloaded model (should not appear)
        let partial_dir = tmp.path().join("model-b");
        std::fs::create_dir_all(&partial_dir).unwrap();
        std::fs::write(partial_dir.join("model.onnx"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        let downloaded = mgr.list_downloaded();
        assert_eq!(downloaded.len(), 1);
        assert!(downloaded.contains(&"model-a".to_string()));
    }

    #[test]
    fn list_downloaded_ignores_files() {
        let tmp = tempdir().unwrap();
        // A file (not a directory) should be ignored
        std::fs::write(tmp.path().join("stray-file.txt"), b"data").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(mgr.list_downloaded().is_empty());
    }

    // -----------------------------------------------------------------------
    // ModelManager::ensure_model
    // -----------------------------------------------------------------------

    #[test]
    fn ensure_model_error_when_not_downloaded() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        let result = mgr.ensure_model("missing-model");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("missing-model"));
    }

    #[test]
    fn ensure_model_ok_when_downloaded() {
        let tmp = tempdir().unwrap();
        let model_dir = tmp.path().join("good-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        let result = mgr.ensure_model("good-model");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_dir);
    }

    // -----------------------------------------------------------------------
    // ModelManager::ort_lib_path / is_ort_downloaded
    // -----------------------------------------------------------------------

    #[test]
    fn ort_lib_path_construction() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        let lib_path = mgr.ort_lib_path();
        assert_eq!(lib_path, tmp.path().join("ort").join(ORT_LIB_FILENAME));
    }

    #[test]
    fn is_ort_downloaded_false_when_missing() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(!mgr.is_ort_downloaded());
    }

    #[test]
    fn is_ort_downloaded_true_when_exists() {
        let tmp = tempdir().unwrap();
        let ort_dir = tmp.path().join("ort");
        std::fs::create_dir_all(&ort_dir).unwrap();
        std::fs::write(ort_dir.join(ORT_LIB_FILENAME), b"fake-lib").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert!(mgr.is_ort_downloaded());
    }

    // -----------------------------------------------------------------------
    // validate_model_dir
    // -----------------------------------------------------------------------

    #[test]
    fn validate_model_dir_missing_model_onnx() {
        let tmp = tempdir().unwrap();
        let result = validate_model_dir(tmp.path());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("model.onnx"));
    }

    #[test]
    fn validate_model_dir_missing_tokenizer() {
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("model.onnx"), b"fake").unwrap();
        let result = validate_model_dir(tmp.path());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("tokenizer.json"));
    }

    #[test]
    fn validate_model_dir_ok_when_both_present() {
        let tmp = tempdir().unwrap();
        std::fs::write(tmp.path().join("model.onnx"), b"fake").unwrap();
        std::fs::write(tmp.path().join("tokenizer.json"), b"fake").unwrap();
        assert!(validate_model_dir(tmp.path()).is_ok());
    }

    // -----------------------------------------------------------------------
    // ModelManager::init_ort_env
    // -----------------------------------------------------------------------

    #[test]
    fn init_ort_env_does_not_panic() {
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        // Should not panic even if the ORT lib does not exist
        mgr.init_ort_env();
    }

    // -----------------------------------------------------------------------
    // ModelSpec field coverage
    // -----------------------------------------------------------------------

    #[test]
    fn model_spec_prefixes() {
        assert_eq!(NOMIC_EMBED_TEXT_V1_5.search_prefix, "search_query: ");
        assert_eq!(NOMIC_EMBED_TEXT_V1_5.document_prefix, "search_document: ");
        assert_eq!(BGE_M3.search_prefix, "");
        assert_eq!(BGE_M3.document_prefix, "");
    }

    #[test]
    fn model_spec_pooling_strategies() {
        assert_eq!(NOMIC_EMBED_TEXT_V1_5.pooling, PoolingStrategy::MeanPooling);
        assert_eq!(BGE_M3.pooling, PoolingStrategy::ClsToken);
    }

    #[test]
    fn model_spec_matryoshka_dims() {
        assert_eq!(
            NOMIC_EMBED_TEXT_V1_5.matryoshka_dims,
            &[768, 512, 256, 128, 64]
        );
        assert_eq!(NOMIC_EMBED_TEXT_V2.matryoshka_dims, &[768, 512, 256, 128]);
        assert!(BGE_M3.matryoshka_dims.is_empty());
        assert!(NOMIC_EMBED_VISION_V1_5.matryoshka_dims.is_empty());
    }

    // -----------------------------------------------------------------------
    // ort_download_url (internal helper, exercised for coverage)
    // -----------------------------------------------------------------------

    #[test]
    fn ort_download_url_contains_version() {
        let url = ort_download_url();
        assert!(url.contains(ORT_VERSION));
        assert!(url.starts_with("https://github.com/microsoft/onnxruntime/"));
    }

    #[test]
    fn ort_download_url_contains_platform() {
        let url = ort_download_url();
        // On macOS arm64, should contain "osx" and "arm64"
        // On other platforms, should contain relevant strings
        assert!(url.contains(".tgz"), "URL should end with .tgz");
        // Just verify it contains the onnxruntime prefix
        assert!(url.contains("onnxruntime-"));
    }

    // -----------------------------------------------------------------------
    // extract_ort_lib — success path
    // -----------------------------------------------------------------------

    fn make_ort_tgz() -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        let data = b"fake ort library";
        header.set_size(data.len() as u64);
        header
            .set_path(format!(
                "onnxruntime-osx-arm64-{}/lib/{}",
                ORT_VERSION, ORT_LIB_FILENAME
            ))
            .unwrap();
        header.set_cksum();
        builder.append(&header, &data[..]).unwrap();
        let tar_bytes = builder.into_inner().unwrap();

        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut gz, &tar_bytes).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn extract_ort_lib_success() {
        let tgz = make_ort_tgz();
        let tmp = tempdir().unwrap();
        let dest = tmp.path().join(ORT_LIB_FILENAME);

        extract_ort_lib(&tgz, &dest).unwrap();
        assert!(dest.exists());

        let content = std::fs::read(&dest).unwrap();
        assert_eq!(content, b"fake ort library");
    }

    // -----------------------------------------------------------------------
    // extract_ort_lib — library not found in archive
    // -----------------------------------------------------------------------

    fn make_ort_tgz_missing_lib() -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        let data = b"some other file";
        header.set_size(data.len() as u64);
        header
            .set_path("onnxruntime-osx-arm64-1.20.1/lib/some_other_file.txt")
            .unwrap();
        header.set_cksum();
        builder.append(&header, &data[..]).unwrap();
        let tar_bytes = builder.into_inner().unwrap();

        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut gz, &tar_bytes).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn extract_ort_lib_missing_library_returns_error() {
        let tgz = make_ort_tgz_missing_lib();
        let tmp = tempdir().unwrap();
        let dest = tmp.path().join(ORT_LIB_FILENAME);

        let result = extract_ort_lib(&tgz, &dest);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found in ONNX Runtime archive"));
    }

    // -----------------------------------------------------------------------
    // extract_ort_lib — invalid archive bytes
    // -----------------------------------------------------------------------

    #[test]
    fn extract_ort_lib_invalid_archive() {
        let tmp = tempdir().unwrap();
        let dest = tmp.path().join(ORT_LIB_FILENAME);

        let result = extract_ort_lib(b"not a gzip archive", &dest);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ModelManager::init_ort_env — with library present
    // -----------------------------------------------------------------------

    #[test]
    fn init_ort_env_with_lib_present() {
        let tmp = tempdir().unwrap();
        let ort_dir = tmp.path().join("ort");
        std::fs::create_dir_all(&ort_dir).unwrap();
        std::fs::write(ort_dir.join(ORT_LIB_FILENAME), b"fake-lib").unwrap();

        // Clear ORT_DYLIB_PATH so the function can set it
        let prev = std::env::var("ORT_DYLIB_PATH").ok();
        std::env::remove_var("ORT_DYLIB_PATH");

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        mgr.init_ort_env();

        // Restore previous value
        if let Some(val) = prev {
            std::env::set_var("ORT_DYLIB_PATH", val);
        } else {
            std::env::remove_var("ORT_DYLIB_PATH");
        }
    }

    // -----------------------------------------------------------------------
    // ModelManager::download_ort — already downloaded branch
    // -----------------------------------------------------------------------

    #[test]
    fn download_ort_already_exists() {
        let tmp = tempdir().unwrap();
        let ort_dir = tmp.path().join("ort");
        std::fs::create_dir_all(&ort_dir).unwrap();
        std::fs::write(ort_dir.join(ORT_LIB_FILENAME), b"fake-lib").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        // Should return Ok immediately since lib already exists
        let result = mgr.download_ort();
        assert!(result.is_ok());
    }
}
