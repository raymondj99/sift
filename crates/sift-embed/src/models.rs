use std::io::Read as _;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use sift_core::{Config, SiftResult};

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
    pub fn new() -> Self {
        Self {
            models_dir: Config::models_dir(),
        }
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

    /// Download model files from HuggingFace.
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
                format!("{}/onnx/model.onnx", base_url)
            } else {
                format!("{}/{}", base_url, file)
            };

            info!("Downloading {} from {}", file, url);

            let response = ureq::get(&url)
                .call()
                .map_err(|e| sift_core::SiftError::Model(format!("Download failed: {}", e)))?;

            let mut bytes = Vec::new();
            response
                .into_reader()
                .read_to_end(&mut bytes)
                .map_err(|e| sift_core::SiftError::Model(format!("Download read failed: {}", e)))?;

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
                "Model '{}' not found. Run `vx models download {}` first.",
                model_name, model_name
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

        let response = ureq::get(&url)
            .call()
            .map_err(|e| sift_core::SiftError::Model(format!("ONNX Runtime download failed: {}", e)))?;

        let mut bytes = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| sift_core::SiftError::Model(format!("Download read failed: {}", e)))?;

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
                "ONNX Runtime not found at {}. Run `vx models download` or set ORT_DYLIB_PATH.",
                lib_path.display()
            );
        }
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
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
        "https://github.com/microsoft/onnxruntime/releases/download/v{ver}/onnxruntime-{os}-{arch}-{ver}.{ext}",
        ver = ORT_VERSION,
        os = os,
        arch = arch,
        ext = ext,
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
        .map_err(|e| sift_core::SiftError::Model(format!("Failed to read archive: {}", e)))?
    {
        let mut entry =
            entry.map_err(|e| sift_core::SiftError::Model(format!("Archive entry error: {}", e)))?;
        let path = entry
            .path()
            .map_err(|e| sift_core::SiftError::Model(format!("Archive path error: {}", e)))?;

        let path_str = path.to_string_lossy();
        // The library is at lib/libonnxruntime.so (or .dylib) inside the archive
        if path_str.contains(ORT_LIB_FILENAME) && !path_str.ends_with('/') {
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut entry, &mut buf)
                .map_err(|e| sift_core::SiftError::Model(format!("Extract read error: {}", e)))?;
            std::fs::write(dest, &buf)?;
            return Ok(());
        }
    }

    Err(sift_core::SiftError::Model(format!(
        "{} not found in ONNX Runtime archive",
        ORT_LIB_FILENAME
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
