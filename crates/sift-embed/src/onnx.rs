use crate::traits::Embedder;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::Session;
use std::path::Path;
use std::sync::Arc;
use tracing::debug;
use sift_core::SiftResult;

/// Select the best available execution providers based on enabled features.
///
/// The order matters: ONNX Runtime will try providers in order and fall back to
/// the next one if registration fails. GPU providers are listed first so that
/// they are preferred when available, with CPU as the final fallback.
#[allow(clippy::vec_init_then_push)]
fn select_execution_providers() -> Vec<ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        providers.push(CUDAExecutionProvider::default().build());
    }

    #[cfg(feature = "coreml")]
    {
        use ort::execution_providers::CoreMLExecutionProvider;
        providers.push(CoreMLExecutionProvider::default().build());
    }

    // CPU is always the fallback
    providers.push(CPUExecutionProvider::default().build());

    providers
}

/// ONNX Runtime-based embedder. Thread-safe — the `ort` `Session` (since rc.6+)
/// is `Send + Sync`, so we wrap it in an `Arc` instead of a `Mutex`.
pub struct OnnxEmbedder {
    session: Arc<Session>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    dimensions: usize,
    model_name: String,
    max_tokens: usize,
}

impl OnnxEmbedder {
    pub fn load(model_dir: &Path, model_name: &str, dimensions: usize) -> SiftResult<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let num_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let session = Session::builder()
            .map_err(|e| {
                sift_core::SiftError::Embedding(format!(
                    "ONNX Runtime not found. Install it and set ORT_DYLIB_PATH, \
                     or run `vx models download` which includes the runtime. \
                     Details: {}",
                    e
                ))
            })?
            .with_intra_threads(num_cores)
            .map_err(|e| sift_core::SiftError::Embedding(format!("ONNX thread config error: {}", e)))?
            .with_inter_threads(2)
            .map_err(|e| sift_core::SiftError::Embedding(format!("ONNX thread config error: {}", e)))?
            .with_execution_providers(select_execution_providers())
            .map_err(|e| {
                sift_core::SiftError::Embedding(format!("Execution provider config error: {}", e))
            })?
            .commit_from_file(&model_path)
            .map_err(|e| {
                sift_core::SiftError::Embedding(format!(
                    "Failed to load ONNX model from {}: {}",
                    model_path.display(),
                    e
                ))
            })?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            sift_core::SiftError::Embedding(format!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            ))
        })?;

        Ok(Self {
            session: Arc::new(session),
            tokenizer: Arc::new(tokenizer),
            dimensions,
            model_name: model_name.to_string(),
            max_tokens: 8192,
        })
    }

    /// Embed texts and then truncate to `target_dim` dimensions (Matryoshka).
    ///
    /// If `target_dim >= self.dimensions`, the full embeddings are returned
    /// unchanged. Otherwise the vectors are truncated and re-normalised to
    /// unit length.
    pub fn embed_with_dim(&self, texts: &[&str], target_dim: usize) -> SiftResult<Vec<Vec<f32>>> {
        let full = self.embed_batch(texts)?;
        if target_dim >= self.dimensions {
            return Ok(full);
        }
        // Truncate and re-normalize
        Ok(full
            .into_iter()
            .map(|mut v| {
                v.truncate(target_dim);
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                v
            })
            .collect())
    }

    #[allow(clippy::type_complexity)]
    fn tokenize_batch(
        &self,
        texts: &[&str],
    ) -> SiftResult<(Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| sift_core::SiftError::Embedding(format!("Tokenization failed: {}", e)))?;

        let mut input_ids_batch = Vec::with_capacity(encodings.len());
        let mut attention_mask_batch = Vec::with_capacity(encodings.len());
        let mut token_type_ids_batch = Vec::with_capacity(encodings.len());

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_tokens))
            .max()
            .unwrap_or(0);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len().min(self.max_tokens);

            let mut padded_ids = vec![0i64; max_len];
            let mut padded_mask = vec![0i64; max_len];
            let token_type_ids = vec![0i64; max_len]; // single-segment, always zero

            for i in 0..len {
                padded_ids[i] = ids[i] as i64;
                padded_mask[i] = mask[i] as i64;
            }

            input_ids_batch.push(padded_ids);
            attention_mask_batch.push(padded_mask);
            token_type_ids_batch.push(token_type_ids);
        }

        Ok((input_ids_batch, attention_mask_batch, token_type_ids_batch))
    }

    fn mean_pooling(
        token_embeddings: &[f32],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Vec<Vec<f32>> {
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_size];
            let mut count = 0.0f32;

            for s in 0..seq_len {
                let mask_val = attention_mask[b * seq_len + s] as f32;
                if mask_val > 0.0 {
                    let offset = (b * seq_len + s) * hidden_size;
                    for d in 0..hidden_size {
                        pooled[d] += token_embeddings[offset + d] * mask_val;
                    }
                    count += mask_val;
                }
            }

            if count > 0.0 {
                for v in pooled.iter_mut().take(hidden_size) {
                    *v /= count;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in pooled.iter_mut().take(hidden_size) {
                    *v /= norm;
                }
            }

            results.push(pooled);
        }

        results
    }
}

impl Embedder for OnnxEmbedder {
    fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(batch_size = texts.len(), "Embedding batch");

        let (input_ids_batch, attention_mask_batch, token_type_ids_batch) =
            self.tokenize_batch(texts)?;
        let batch_size = input_ids_batch.len();
        let seq_len = input_ids_batch[0].len();

        // Flatten for ONNX
        let input_ids_flat: Vec<i64> = input_ids_batch.into_iter().flatten().collect();
        let attention_mask_flat: Vec<i64> = attention_mask_batch.into_iter().flatten().collect();
        let token_type_ids_flat: Vec<i64> = token_type_ids_batch.into_iter().flatten().collect();

        let input_ids_array =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids_flat)
                .map_err(|e| sift_core::SiftError::Embedding(format!("Shape error: {}", e)))?;

        let attention_mask_array =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask_flat.clone())
                .map_err(|e| sift_core::SiftError::Embedding(format!("Shape error: {}", e)))?;

        let token_type_ids_array =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), token_type_ids_flat)
                .map_err(|e| sift_core::SiftError::Embedding(format!("Shape error: {}", e)))?;

        let outputs = self
            .session
            .run(
                ort::inputs! {
                    "input_ids" => input_ids_array,
                    "attention_mask" => attention_mask_array,
                    "token_type_ids" => token_type_ids_array,
                }
                .map_err(|e| sift_core::SiftError::Embedding(format!("Input error: {}", e)))?,
            )
            .map_err(|e| sift_core::SiftError::Embedding(format!("ONNX inference failed: {}", e)))?;

        // Extract the first output tensor (last_hidden_state or token_embeddings)
        // Try named output first, fall back to index-based access
        let output_array = if let Some(v) = outputs.get("last_hidden_state") {
            v.try_extract_tensor::<f32>()
                .map_err(|e| sift_core::SiftError::Embedding(format!("Extract error: {}", e)))?
        } else {
            // Use the first output by index
            let first = &outputs[0];
            first
                .try_extract_tensor::<f32>()
                .map_err(|e| sift_core::SiftError::Embedding(format!("Extract error: {}", e)))?
        };

        let shape = output_array.shape();
        let hidden_size = shape[shape.len() - 1];
        let token_embeddings: Vec<f32> = output_array.iter().copied().collect();

        let results = Self::mean_pooling(
            &token_embeddings,
            &attention_mask_flat,
            batch_size,
            seq_len,
            hidden_size,
        );

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}
