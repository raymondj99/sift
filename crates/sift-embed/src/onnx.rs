use crate::traits::Embedder;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::Session;
use sift_core::SiftResult;
use std::path::Path;
use std::sync::Arc;
use tracing::debug;

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
            .map(std::num::NonZero::get)
            .unwrap_or(4);

        let session = Session::builder()
            .map_err(|e| {
                sift_core::SiftError::Embedding(format!(
                    "ONNX Runtime not found. Install it and set ORT_DYLIB_PATH, \
                     or run `sift models download` which includes the runtime. \
                     Details: {e}"
                ))
            })?
            .with_intra_threads(num_cores)
            .map_err(|e| sift_core::SiftError::Embedding(format!("ONNX thread config error: {e}")))?
            .with_inter_threads(2)
            .map_err(|e| sift_core::SiftError::Embedding(format!("ONNX thread config error: {e}")))?
            .with_execution_providers(select_execution_providers())
            .map_err(|e| {
                sift_core::SiftError::Embedding(format!("Execution provider config error: {e}"))
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
                    for x in &mut v {
                        *x /= norm;
                    }
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
            .map_err(|e| sift_core::SiftError::Embedding(format!("Tokenization failed: {e}")))?;

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
                padded_ids[i] = i64::from(ids[i]);
                padded_mask[i] = i64::from(mask[i]);
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
                .map_err(|e| sift_core::SiftError::Embedding(format!("Shape error: {e}")))?;

        let attention_mask_array =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask_flat.clone())
                .map_err(|e| sift_core::SiftError::Embedding(format!("Shape error: {e}")))?;

        let token_type_ids_array =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), token_type_ids_flat)
                .map_err(|e| sift_core::SiftError::Embedding(format!("Shape error: {e}")))?;

        let outputs = self
            .session
            .run(
                ort::inputs! {
                    "input_ids" => input_ids_array,
                    "attention_mask" => attention_mask_array,
                    "token_type_ids" => token_type_ids_array,
                }
                .map_err(|e| sift_core::SiftError::Embedding(format!("Input error: {e}")))?,
            )
            .map_err(|e| sift_core::SiftError::Embedding(format!("ONNX inference failed: {e}")))?;

        // Extract the first output tensor (last_hidden_state or token_embeddings)
        // Try named output first, fall back to index-based access
        let output_array = if let Some(v) = outputs.get("last_hidden_state") {
            v.try_extract_tensor::<f32>()
                .map_err(|e| sift_core::SiftError::Embedding(format!("Extract error: {e}")))?
        } else {
            // Use the first output by index
            let first = &outputs[0];
            first
                .try_extract_tensor::<f32>()
                .map_err(|e| sift_core::SiftError::Embedding(format!("Extract error: {e}")))?
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // select_execution_providers
    // -----------------------------------------------------------------------

    #[test]
    fn select_execution_providers_contains_cpu() {
        let providers = select_execution_providers();
        // There must be at least one provider (CPU fallback)
        assert!(!providers.is_empty());
    }

    // -----------------------------------------------------------------------
    // OnnxEmbedder::mean_pooling — pure math, no ONNX session needed
    // -----------------------------------------------------------------------

    #[test]
    fn mean_pooling_single_item_uniform_mask() {
        // batch_size=1, seq_len=2, hidden_size=3
        // All tokens unmasked (mask=1)
        let token_embeddings: Vec<f32> = vec![
            1.0, 2.0, 3.0, // token 0
            4.0, 5.0, 6.0, // token 1
        ];
        let attention_mask: Vec<i64> = vec![1, 1];

        let results = OnnxEmbedder::mean_pooling(&token_embeddings, &attention_mask, 1, 2, 3);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 3);

        // Mean of [1,2,3] and [4,5,6] = [2.5, 3.5, 4.5]
        // Then L2 normalized: norm = sqrt(2.5^2 + 3.5^2 + 4.5^2) = sqrt(6.25+12.25+20.25)=sqrt(38.75)
        let raw_mean = [2.5f32, 3.5, 4.5];
        let norm: f32 = raw_mean.iter().map(|x| x * x).sum::<f32>().sqrt();

        for (i, &v) in results[0].iter().enumerate() {
            let expected = raw_mean[i] / norm;
            assert!(
                (v - expected).abs() < 1e-5,
                "dim {i}: got {v}, expected {expected}"
            );
        }

        // Verify unit length
        let result_norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (result_norm - 1.0).abs() < 1e-5,
            "result should be L2-normalized, got norm {result_norm}"
        );
    }

    #[test]
    fn mean_pooling_batch_of_two_different_masks() {
        // batch_size=2, seq_len=3, hidden_size=2
        #[rustfmt::skip]
        let token_embeddings: Vec<f32> = vec![
            // batch 0
            1.0, 0.0,   // token 0
            0.0, 1.0,   // token 1
            9.0, 9.0,   // token 2 (will be masked for batch 0)
            // batch 1
            2.0, 2.0,   // token 0
            4.0, 4.0,   // token 1
            6.0, 6.0,   // token 2
        ];
        let attention_mask: Vec<i64> = vec![
            1, 1, 0, // batch 0: only first 2 tokens
            1, 1, 1, // batch 1: all 3 tokens
        ];

        let results = OnnxEmbedder::mean_pooling(&token_embeddings, &attention_mask, 2, 3, 2);
        assert_eq!(results.len(), 2);

        // Batch 0: mean of [1,0],[0,1] = [0.5, 0.5], norm=sqrt(0.5), normalized=[1/sqrt(2), 1/sqrt(2)]
        let expected_0 = 1.0f32 / 2.0f32.sqrt();
        assert!((results[0][0] - expected_0).abs() < 1e-5);
        assert!((results[0][1] - expected_0).abs() < 1e-5);

        // Batch 1: mean of [2,2],[4,4],[6,6] = [4,4], norm=sqrt(32), normalized=[4/sqrt(32), 4/sqrt(32)]
        let norm_1 = (4.0f32 * 4.0 + 4.0 * 4.0).sqrt();
        let expected_1 = 4.0 / norm_1;
        assert!((results[1][0] - expected_1).abs() < 1e-5);
        assert!((results[1][1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn mean_pooling_zero_mask_produces_zero_vector() {
        // All tokens masked out
        let token_embeddings: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attention_mask: Vec<i64> = vec![0, 0];

        let results = OnnxEmbedder::mean_pooling(&token_embeddings, &attention_mask, 1, 2, 3);
        assert_eq!(results.len(), 1);
        // When count == 0 and norm == 0, the result should be all zeros
        for &v in &results[0] {
            assert!((v - 0.0).abs() < 1e-10, "expected 0.0 but got {v}");
        }
    }

    #[test]
    fn mean_pooling_single_token() {
        // batch_size=1, seq_len=1, hidden_size=4
        let token_embeddings: Vec<f32> = vec![3.0, 0.0, 4.0, 0.0];
        let attention_mask: Vec<i64> = vec![1];

        let results = OnnxEmbedder::mean_pooling(&token_embeddings, &attention_mask, 1, 1, 4);
        assert_eq!(results.len(), 1);

        // Mean is [3,0,4,0], norm=5, normalized=[0.6, 0.0, 0.8, 0.0]
        assert!((results[0][0] - 0.6).abs() < 1e-5);
        assert!((results[0][1] - 0.0).abs() < 1e-5);
        assert!((results[0][2] - 0.8).abs() < 1e-5);
        assert!((results[0][3] - 0.0).abs() < 1e-5);
    }
}
