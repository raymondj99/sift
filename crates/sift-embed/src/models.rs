use sift_core::{Config, SiftResult};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Once;
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

/// Broad model category. Text models need a tokenizer; vision models consume
/// image tensors directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Text,
    Vision,
}

/// Runtime required to execute a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRuntime {
    /// Supported by sift's native ONNX Runtime embedder.
    Onnx,
    /// Requires a Transformers/SentenceTransformers-style runtime.
    Transformers,
    /// Requires a dedicated server or integration outside sift's native path.
    External,
}

impl ModelRuntime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Transformers => "transformers",
            Self::External => "external",
        }
    }
}

/// License family. Kept deliberately coarse for CLI display and policy checks;
/// model cards remain the source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLicense {
    Apache2,
    Mit,
    Gemma,
    CcByNc4,
    NvidiaOpenModelNonCommercial,
    Unknown,
}

impl ModelLicense {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Apache2 => "Apache-2.0",
            Self::Mit => "MIT",
            Self::Gemma => "Gemma terms",
            Self::CcByNc4 => "CC-BY-NC-4.0",
            Self::NvidiaOpenModelNonCommercial => "NVIDIA non-commercial",
            Self::Unknown => "unknown",
        }
    }
}

/// File copied from a Hugging Face repo into sift's canonical local model dir.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    /// Path in the Hugging Face repo.
    pub source: &'static str,
    /// Path relative to the local model directory.
    pub dest: &'static str,
}

impl ModelFile {
    pub const fn new(source: &'static str, dest: &'static str) -> Self {
        Self { source, dest }
    }
}

const BASIC_ONNX_FILES: &[ModelFile] = &[
    ModelFile::new("onnx/model.onnx", "model.onnx"),
    ModelFile::new("tokenizer.json", "tokenizer.json"),
];

const ONNX_WITH_EXTERNAL_DATA_FILES: &[ModelFile] = &[
    ModelFile::new("onnx/model.onnx", "model.onnx"),
    ModelFile::new("onnx/model.onnx_data", "model.onnx_data"),
    ModelFile::new("tokenizer.json", "tokenizer.json"),
];

const BGE_M3_ONNX_FILES: &[ModelFile] = &[
    ModelFile::new("onnx/model.onnx", "model.onnx"),
    ModelFile::new("onnx/model.onnx_data", "model.onnx_data"),
    ModelFile::new("onnx/Constant_7_attr__value", "Constant_7_attr__value"),
    ModelFile::new("onnx/tokenizer.json", "tokenizer.json"),
];

/// Extended model specification with runtime, license, pooling, file-layout,
/// and Matryoshka metadata.
pub struct ModelSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub repo_id: &'static str,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub kind: ModelKind,
    pub runtime: ModelRuntime,
    pub license: ModelLicense,
    /// Prefix prepended to search/query texts before embedding.
    pub search_prefix: &'static str,
    /// Prefix prepended to document/passage texts before embedding.
    pub document_prefix: &'static str,
    /// How the model pools token-level representations.
    pub pooling: PoolingStrategy,
    /// Matryoshka-supported truncated dimensions (largest first).
    /// Empty slice means the model does not support Matryoshka truncation.
    pub matryoshka_dims: &'static [usize],
    /// Files that sift's native downloader knows how to materialize locally.
    /// Empty means the model is tracked for evaluation but not natively
    /// downloadable by this binary yet.
    pub download_files: &'static [ModelFile],
    /// Short operational note for CLI display and future policy decisions.
    pub notes: &'static str,
}

/// Backwards-compatible alias so that existing code referencing `ModelDef`
/// continues to compile without changes.
pub type ModelDef = ModelSpec;

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

pub const NOMIC_EMBED_TEXT_V1_5: ModelSpec = ModelSpec {
    name: "nomic-embed-text-v1.5",
    // Backwards compatibility: earlier sift releases exposed the v1.5 ONNX
    // repo under `nomic-embed-text-v2`. Keep resolving that name to the model
    // users actually downloaded, while tracking the true v2 MoE model below.
    aliases: &["nomic-embed-text-v2"],
    repo_id: "nomic-ai/nomic-embed-text-v1.5",
    dimensions: 768,
    max_tokens: 8192,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Onnx,
    license: ModelLicense::Apache2,
    search_prefix: "search_query: ",
    document_prefix: "search_document: ",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[768, 512, 256, 128, 64],
    download_files: BASIC_ONNX_FILES,
    notes: "Current native local default; legacy alias: nomic-embed-text-v2.",
};

pub const NOMIC_EMBED_TEXT_V2_MOE: ModelSpec = ModelSpec {
    name: "nomic-embed-text-v2-moe",
    aliases: &[],
    repo_id: "nomic-ai/nomic-embed-text-v2-moe",
    dimensions: 768,
    max_tokens: 2048,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Transformers,
    license: ModelLicense::Apache2,
    search_prefix: "search_query: ",
    document_prefix: "search_document: ",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[768, 512, 256, 128],
    download_files: &[],
    notes: "True Nomic v2 MoE model; tracked for evaluation, native ONNX path not available in the upstream repo.",
};

pub const BGE_M3: ModelSpec = ModelSpec {
    name: "bge-m3",
    aliases: &[],
    repo_id: "BAAI/bge-m3",
    dimensions: 1024,
    max_tokens: 8192,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Onnx,
    license: ModelLicense::Mit,
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::ClsToken,
    matryoshka_dims: &[],
    download_files: BGE_M3_ONNX_FILES,
    notes: "Dense path is native; sparse and multi-vector outputs need future runtime support.",
};

pub const EMBEDDING_GEMMA_300M: ModelSpec = ModelSpec {
    name: "embeddinggemma-300m",
    aliases: &["google-embeddinggemma-300m"],
    repo_id: "onnx-community/embeddinggemma-300m-ONNX",
    dimensions: 768,
    max_tokens: 2048,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Onnx,
    license: ModelLicense::Gemma,
    search_prefix: "task: search result | query: ",
    document_prefix: "title: none | text: ",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[],
    download_files: ONNX_WITH_EXTERNAL_DATA_FILES,
    notes: "Small multilingual on-device model via ONNX community export.",
};

pub const SNOWFLAKE_ARCTIC_EMBED_L_V2: ModelSpec = ModelSpec {
    name: "snowflake-arctic-embed-l-v2.0",
    aliases: &["arctic-embed-l-v2.0"],
    repo_id: "Snowflake/snowflake-arctic-embed-l-v2.0",
    dimensions: 1024,
    max_tokens: 8192,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Onnx,
    license: ModelLicense::Apache2,
    search_prefix: "query: ",
    document_prefix: "",
    pooling: PoolingStrategy::ClsToken,
    matryoshka_dims: &[1024, 256],
    download_files: ONNX_WITH_EXTERNAL_DATA_FILES,
    notes: "Apache-2.0 multilingual retrieval model with Matryoshka-friendly storage tradeoffs.",
};

pub const QWEN3_EMBEDDING_0_6B: ModelSpec = ModelSpec {
    name: "qwen3-embedding-0.6b",
    aliases: &["Qwen/Qwen3-Embedding-0.6B", "qwen3-embedding-0.6B"],
    repo_id: "Qwen/Qwen3-Embedding-0.6B",
    dimensions: 1024,
    max_tokens: 32768,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Transformers,
    license: ModelLicense::Apache2,
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[1024, 768, 512, 256, 128, 64, 32],
    download_files: &[],
    notes: "Instruction-aware SOTA-class model; needs Transformers/runtime support before native download.",
};

pub const JINA_EMBEDDINGS_V5_TEXT_SMALL: ModelSpec = ModelSpec {
    name: "jina-embeddings-v5-text-small-retrieval",
    aliases: &["jina-embeddings-v5-text-small"],
    repo_id: "jinaai/jina-embeddings-v5-text-small-retrieval",
    dimensions: 1024,
    max_tokens: 32768,
    kind: ModelKind::Text,
    runtime: ModelRuntime::Onnx,
    license: ModelLicense::CcByNc4,
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[1024, 768, 512, 256, 128, 64, 32],
    download_files: ONNX_WITH_EXTERNAL_DATA_FILES,
    notes: "High-quality multilingual ONNX retrieval adapter; non-commercial license.",
};

pub const NV_EMBED_V2: ModelSpec = ModelSpec {
    name: "nv-embed-v2",
    aliases: &["nvidia/NV-Embed-v2"],
    repo_id: "nvidia/NV-Embed-v2",
    dimensions: 4096,
    max_tokens: 32768,
    kind: ModelKind::Text,
    runtime: ModelRuntime::External,
    license: ModelLicense::NvidiaOpenModelNonCommercial,
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[],
    download_files: &[],
    notes: "Benchmark/reference model only; gated and not native-downloadable.",
};

/// Nomic Embed Vision v1.5 — image embedding model.
/// Produces 768-dim vectors in the same embedding space as the text model,
/// enabling cross-modal search (search images with text queries).
pub const NOMIC_EMBED_VISION_V1_5: ModelSpec = ModelSpec {
    name: "nomic-embed-vision-v1.5",
    aliases: &[],
    repo_id: "nomic-ai/nomic-embed-vision-v1.5",
    dimensions: 768,
    max_tokens: 0, // not applicable for vision models
    kind: ModelKind::Vision,
    runtime: ModelRuntime::Onnx,
    license: ModelLicense::Apache2,
    search_prefix: "",
    document_prefix: "",
    pooling: PoolingStrategy::MeanPooling,
    matryoshka_dims: &[],
    download_files: &[ModelFile::new("onnx/model.onnx", "model.onnx")],
    notes: "Vision-side model sharing Nomic text v1.5 latent space.",
};

/// All known model specs, for programmatic iteration and lookup.
const ALL_MODELS: &[&ModelSpec] = &[
    &NOMIC_EMBED_TEXT_V1_5,
    &NOMIC_EMBED_TEXT_V2_MOE,
    &BGE_M3,
    &EMBEDDING_GEMMA_300M,
    &SNOWFLAKE_ARCTIC_EMBED_L_V2,
    &QWEN3_EMBEDDING_0_6B,
    &JINA_EMBEDDINGS_V5_TEXT_SMALL,
    &NV_EMBED_V2,
    &NOMIC_EMBED_VISION_V1_5,
];

/// Look up a model specification by its short name (e.g. `"bge-m3"`).
pub fn get_model(name: &str) -> Option<&'static ModelSpec> {
    ALL_MODELS
        .iter()
        .find(|m| m.name == name || m.aliases.contains(&name))
        .copied()
}

/// Return a slice of all registered model specs.
pub fn all_models() -> &'static [&'static ModelSpec] {
    ALL_MODELS
}

/// Return all text embedding models that sift can natively download and run.
pub fn downloadable_text_models() -> impl Iterator<Item = &'static ModelSpec> {
    ALL_MODELS
        .iter()
        .copied()
        .filter(|m| m.kind == ModelKind::Text && m.is_download_supported())
}

impl ModelSpec {
    pub fn is_download_supported(&self) -> bool {
        self.runtime == ModelRuntime::Onnx && !self.download_files.is_empty()
    }

    pub fn is_text_embedding(&self) -> bool {
        self.kind == ModelKind::Text
    }

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
        if let Some(model) = get_model(model_name) {
            return self.is_downloaded_model(model);
        }
        self.model_path(model_name).exists() && self.tokenizer_path(model_name).exists()
    }

    pub fn is_downloaded_model(&self, model: &ModelSpec) -> bool {
        self.downloaded_model_dir(model).is_some()
    }

    pub fn downloaded_model_dir(&self, model: &ModelSpec) -> Option<PathBuf> {
        if !model.is_download_supported() {
            return None;
        }

        let mut candidates = Vec::with_capacity(1 + model.aliases.len());
        candidates.push(model.name);
        candidates.extend_from_slice(model.aliases);

        candidates.into_iter().find_map(|name| {
            let dir = self.model_dir(name);
            let complete = model
                .download_files
                .iter()
                .all(|file| dir.join(file.dest).exists());
            complete.then_some(dir)
        })
    }

    fn has_legacy_basic_download(&self, model_name: &str) -> bool {
        let dir = self.model_dir(model_name);
        dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists()
    }

    pub fn is_downloaded_legacy_dir(&self, model_name: &str) -> bool {
        if let Some(model) = get_model(model_name) {
            return self.is_downloaded_model(model);
        }
        self.has_legacy_basic_download(model_name)
    }

    /// Check if a model's ONNX file exists (without requiring a tokenizer).
    /// Used for vision models that take pixel data rather than text.
    pub fn is_model_file_downloaded(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists()
    }

    /// Download model files from `HuggingFace`.
    /// Retries transient network failures with exponential backoff.
    pub fn download(&self, model_def: &ModelDef) -> SiftResult<()> {
        if !model_def.is_download_supported() {
            return Err(sift_core::SiftError::Model(format!(
                "Model '{}' requires runtime '{}' and cannot be downloaded by sift's native ONNX downloader yet. {}",
                model_def.name,
                model_def.runtime.as_str(),
                model_def.notes
            )));
        }

        let dir = self.model_dir(model_def.name);
        std::fs::create_dir_all(&dir)?;

        let base_url = format!("https://huggingface.co/{}/resolve/main", model_def.repo_id);

        let retry_config = sift_core::RetryConfig::network();
        let is_retryable = |e: &sift_core::SiftError| {
            matches!(
                e,
                sift_core::SiftError::Model(_) | sift_core::SiftError::Io(_)
            )
        };

        for file in model_def.download_files {
            let dest = dir.join(file.dest);
            if dest.exists() {
                info!("{} already exists, skipping", file.dest);
                continue;
            }
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let url = format!("{base_url}/{}", file.source);

            info!("Downloading {} from {}", file.source, url);

            let bytes = sift_core::with_retry(&retry_config, is_retryable, || {
                let response = ureq::get(&url)
                    .call()
                    .map_err(|e| sift_core::SiftError::Model(format!("Download failed: {e}")))?;

                let mut buf = Vec::new();
                response.into_reader().read_to_end(&mut buf).map_err(|e| {
                    sift_core::SiftError::Model(format!("Download read failed: {e}"))
                })?;
                Ok::<_, sift_core::SiftError>(buf)
            })?;

            std::fs::write(&dest, &bytes)?;
            info!("Saved {} ({} bytes)", file.dest, bytes.len());
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
        let Some(model) = get_model(model_name) else {
            return Err(sift_core::SiftError::Model(format!(
                "Unknown model '{model_name}'. Run `sift models list` to see supported models."
            )));
        };

        let Some(model_dir) = self.downloaded_model_dir(model) else {
            return Err(sift_core::SiftError::Model(format!(
                "Model '{model_name}' not found. Run `sift models download {}` first.",
                model.name
            )));
        };
        Ok(model_dir)
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
    /// Retries transient network failures with exponential backoff.
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

        let retry_config = sift_core::RetryConfig::network();
        let is_retryable = |e: &sift_core::SiftError| {
            matches!(
                e,
                sift_core::SiftError::Model(_) | sift_core::SiftError::Io(_)
            )
        };

        let bytes = sift_core::with_retry(&retry_config, is_retryable, || {
            let response = ureq::get(&url).call().map_err(|e| {
                sift_core::SiftError::Model(format!("ONNX Runtime download failed: {e}"))
            })?;

            let mut buf = Vec::new();
            response
                .into_reader()
                .read_to_end(&mut buf)
                .map_err(|e| sift_core::SiftError::Model(format!("Download read failed: {e}")))?;
            Ok::<_, sift_core::SiftError>(buf)
        })?;

        // The download is a .tgz archive — extract the library file
        extract_ort_lib(&bytes, &lib_path)?;

        info!(
            "ONNX Runtime {} saved to {}",
            ORT_VERSION,
            lib_path.display()
        );
        Ok(())
    }

    /// Set `ORT_DYLIB_PATH` if the runtime library exists in our managed
    /// directory and the env var is not already set.
    ///
    /// Equivalent to [`Self::init_ort_env_with_override`] with no override path.
    pub fn init_ort_env(&self) {
        self.init_ort_env_with_override(None);
    }

    /// Same as [`Self::init_ort_env`] but with a user-provided override path
    /// taking precedence over the default `~/.sift/models/ort/` location.
    ///
    /// Precedence (highest first):
    ///   1. `ORT_DYLIB_PATH` environment variable (respected if already set).
    ///   2. `override_path` (typically `config.default.ort_dylib_path`).
    ///   3. Auto-detected `~/.sift/models/ort/<platform-lib>` if it exists.
    ///
    /// If none of the above resolve to an existing file, a single warning is
    /// emitted (guarded by `Once` for the process lifetime) pointing the user
    /// at `sift models download` or the config override.
    pub fn init_ort_env_with_override(&self, override_path: Option<&Path>) {
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            debug!("ORT_DYLIB_PATH already set, skipping auto-config");
            return;
        }

        if let Some(custom) = override_path {
            if custom.exists() {
                debug!(
                    "Setting ORT_DYLIB_PATH to {} (from config)",
                    custom.display()
                );
                std::env::set_var("ORT_DYLIB_PATH", custom);
                return;
            }
            // Fall through to auto-detect, but log why the override was skipped.
            debug!(
                "Configured ort_dylib_path does not exist: {}",
                custom.display()
            );
        }

        let lib_path = self.ort_lib_path();
        if lib_path.exists() {
            debug!("Setting ORT_DYLIB_PATH to {}", lib_path.display());
            std::env::set_var("ORT_DYLIB_PATH", &lib_path);
        } else {
            warn_ort_missing_once(&lib_path);
        }
    }
}

/// Emit the "ORT not found" warning at most once per process.
fn warn_ort_missing_once(lib_path: &Path) {
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        warn!(
            "ONNX Runtime not found at {}. Run `sift models download` to install it, \
             or set `default.ort_dylib_path` in ~/.sift/config.toml (or the ORT_DYLIB_PATH env var).",
            lib_path.display()
        );
    });
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
        // Match the real dylib, not the unversioned symlink that shares the same
        // base name. On macOS the archive contains both:
        //   lib/libonnxruntime.1.20.1.dylib  ← regular file (what we want)
        //   lib/libonnxruntime.dylib          ← symlink, 0 bytes (skip)
        // Checking entry_type().is_file() skips symlinks on all platforms.
        let is_regular_file = entry.header().entry_type().is_file();
        let name_matches = path_str.contains("libonnxruntime")
            && path_str.ends_with(if cfg!(target_os = "windows") {
                ".dll"
            } else if cfg!(target_os = "macos") {
                ".dylib"
            } else {
                ".so"
            });
        if is_regular_file && name_matches {
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
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Serialises every test that mutates the process-global `ORT_DYLIB_PATH`.
    /// Cargo runs tests in parallel by default; without this lock, reads and
    /// writes across tests interleave and break hard assertions.
    static ORT_ENV_MUTEX: Mutex<()> = Mutex::new(());

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
            NOMIC_EMBED_TEXT_V2_MOE.onnx_file_for_quant(QuantizationType::INT8),
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
    fn get_model_alias_resolves() {
        let m = get_model("nomic-embed-text-v2").expect("alias should resolve");
        assert_eq!(m.name, "nomic-embed-text-v1.5");
        assert_eq!(m.repo_id, "nomic-ai/nomic-embed-text-v1.5");
    }

    #[test]
    fn get_model_unknown_returns_none() {
        assert!(get_model("does-not-exist").is_none());
    }

    // -----------------------------------------------------------------------
    // all_models
    // -----------------------------------------------------------------------

    #[test]
    fn all_models_returns_nine() {
        let models = all_models();
        assert_eq!(models.len(), 9);
    }

    #[test]
    fn all_models_contains_expected_names() {
        let names: Vec<&str> = all_models().iter().map(|m| m.name).collect();
        assert!(names.contains(&"nomic-embed-text-v1.5"));
        assert!(names.contains(&"nomic-embed-text-v2-moe"));
        assert!(names.contains(&"bge-m3"));
        assert!(names.contains(&"embeddinggemma-300m"));
        assert!(names.contains(&"snowflake-arctic-embed-l-v2.0"));
        assert!(names.contains(&"qwen3-embedding-0.6b"));
        assert!(names.contains(&"jina-embeddings-v5-text-small-retrieval"));
        assert!(names.contains(&"nv-embed-v2"));
        assert!(names.contains(&"nomic-embed-vision-v1.5"));
    }

    #[test]
    fn downloadable_text_models_excludes_external_runtime_models() {
        let names: Vec<&str> = downloadable_text_models().map(|m| m.name).collect();
        assert!(names.contains(&"nomic-embed-text-v1.5"));
        assert!(names.contains(&"bge-m3"));
        assert!(names.contains(&"embeddinggemma-300m"));
        assert!(names.contains(&"snowflake-arctic-embed-l-v2.0"));
        assert!(names.contains(&"jina-embeddings-v5-text-small-retrieval"));
        assert!(!names.contains(&"nomic-embed-text-v2-moe"));
        assert!(!names.contains(&"qwen3-embedding-0.6b"));
        assert!(!names.contains(&"nv-embed-v2"));
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

    #[test]
    fn downloaded_model_dir_accepts_legacy_alias_directory() {
        let tmp = tempdir().unwrap();
        let legacy_dir = tmp.path().join("nomic-embed-text-v2");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        std::fs::write(legacy_dir.join("model.onnx"), b"fake").unwrap();
        std::fs::write(legacy_dir.join("tokenizer.json"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        assert_eq!(
            mgr.downloaded_model_dir(&NOMIC_EMBED_TEXT_V1_5),
            Some(legacy_dir)
        );
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
        let model_dir = tmp.path().join(NOMIC_EMBED_TEXT_V1_5.name);
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
        assert!(downloaded.contains(&NOMIC_EMBED_TEXT_V1_5.name.to_string()));
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
        let model_dir = tmp.path().join(NOMIC_EMBED_TEXT_V1_5.name);
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"fake").unwrap();

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        let result = mgr.ensure_model(NOMIC_EMBED_TEXT_V1_5.name);
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
        let _guard = ORT_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempdir().unwrap();
        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        // Should not panic even if the ORT lib does not exist
        mgr.init_ort_env();
    }

    #[test]
    fn init_ort_env_override_honored_when_exists() {
        // Precedence check: when the override path exists, it must win over
        // the managed directory, regardless of what's in there.
        let _guard = ORT_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempdir().unwrap();
        let override_dir = tempdir().unwrap();
        let override_path = override_dir.path().join("my-custom-ort.dylib");
        std::fs::write(&override_path, b"fake").unwrap();

        // ORT_DYLIB_PATH is process-global; start from a known-unset state.
        let prev = std::env::var("ORT_DYLIB_PATH").ok();
        std::env::remove_var("ORT_DYLIB_PATH");

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        mgr.init_ort_env_with_override(Some(&override_path));

        let got = std::env::var("ORT_DYLIB_PATH").expect("ORT_DYLIB_PATH should be set");
        assert_eq!(got, override_path.display().to_string());

        // Restore prior state so unrelated test runs see the environment as
        // they found it.
        match prev {
            Some(v) => std::env::set_var("ORT_DYLIB_PATH", v),
            None => std::env::remove_var("ORT_DYLIB_PATH"),
        }
    }

    #[test]
    fn init_ort_env_override_missing_falls_through_to_managed() {
        // When the configured override does not exist, the managed directory
        // lib should be used. Proves the fallthrough branch in
        // `init_ort_env_with_override`.
        let _guard = ORT_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempdir().unwrap();
        let ort_dir = tmp.path().join("ort");
        std::fs::create_dir_all(&ort_dir).unwrap();
        let managed_lib = ort_dir.join(ORT_LIB_FILENAME);
        std::fs::write(&managed_lib, b"fake-lib").unwrap();

        let bogus_override = tmp.path().join("does-not-exist.dylib");

        let prev = std::env::var("ORT_DYLIB_PATH").ok();
        std::env::remove_var("ORT_DYLIB_PATH");

        let mgr = ModelManager::with_dir(tmp.path().to_path_buf());
        mgr.init_ort_env_with_override(Some(&bogus_override));

        let got = std::env::var("ORT_DYLIB_PATH").expect("ORT_DYLIB_PATH should be set");
        assert_eq!(got, managed_lib.display().to_string());

        match prev {
            Some(v) => std::env::set_var("ORT_DYLIB_PATH", v),
            None => std::env::remove_var("ORT_DYLIB_PATH"),
        }
    }

    // -----------------------------------------------------------------------
    // ModelSpec field coverage
    // -----------------------------------------------------------------------

    #[test]
    fn model_spec_prefixes() {
        assert_eq!(NOMIC_EMBED_TEXT_V1_5.search_prefix, "search_query: ");
        assert_eq!(NOMIC_EMBED_TEXT_V1_5.document_prefix, "search_document: ");
        assert_eq!(
            EMBEDDING_GEMMA_300M.search_prefix,
            "task: search result | query: "
        );
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
        assert_eq!(
            NOMIC_EMBED_TEXT_V2_MOE.matryoshka_dims,
            &[768, 512, 256, 128]
        );
        assert_eq!(
            QWEN3_EMBEDDING_0_6B.matryoshka_dims,
            &[1024, 768, 512, 256, 128, 64, 32]
        );
        assert!(BGE_M3.matryoshka_dims.is_empty());
        assert!(NOMIC_EMBED_VISION_V1_5.matryoshka_dims.is_empty());
    }

    #[test]
    fn onnx_models_with_external_data_require_sidecar_download() {
        assert!(EMBEDDING_GEMMA_300M
            .download_files
            .iter()
            .any(|f| f.dest == "model.onnx_data"));
        assert!(SNOWFLAKE_ARCTIC_EMBED_L_V2
            .download_files
            .iter()
            .any(|f| f.dest == "model.onnx_data"));
        assert!(BGE_M3
            .download_files
            .iter()
            .any(|f| f.dest == "Constant_7_attr__value"));
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
        let _guard = ORT_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
