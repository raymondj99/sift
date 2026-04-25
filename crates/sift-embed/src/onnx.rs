use crate::models::{ModelSpec, PoolingStrategy};
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
    pooling: PoolingStrategy,
    output_tensor: &'static str,
    /// Per-spec query prefix (e.g. Nomic's `"search_query: "`). Empty for
    /// symmetric models. Surfaced via [`sift_core::Embedder::search_prefix`]
    /// so call sites can use [`sift_core::Embedder::embed_query`] without
    /// hardcoding the model's expected string.
    search_prefix: &'static str,
    /// Per-spec document prefix (e.g. Nomic's `"search_document: "`). Empty
    /// for symmetric models.
    document_prefix: &'static str,
    /// Optional Matryoshka truncation. `Some(d)` truncates the pooled vector
    /// to the first `d` dimensions and renormalizes; `None` keeps the
    /// model's native dimensionality.
    truncate_dim: Option<usize>,
}

impl OnnxEmbedder {
    pub fn load(model_dir: &Path, model_name: &str, dimensions: usize) -> SiftResult<Self> {
        Self::load_with_options(
            model_dir,
            model_name,
            dimensions,
            8192,
            PoolingStrategy::MeanPooling,
            "last_hidden_state",
            "",
            "",
        )
    }

    pub fn load_model(model_dir: &Path, model: &ModelSpec) -> SiftResult<Self> {
        Self::load_with_options(
            model_dir,
            model.name,
            model.dimensions,
            model.max_tokens,
            model.pooling,
            model.output_tensor,
            model.search_prefix,
            model.document_prefix,
        )
    }

    /// Same as [`Self::load_model`] but truncates the output to the first
    /// `truncate_dim` dimensions (Matryoshka). Returns an error if the model
    /// does not declare `truncate_dim` in its `matryoshka_dims` list.
    pub fn load_model_with_truncation(
        model_dir: &Path,
        model: &ModelSpec,
        truncate_dim: usize,
    ) -> SiftResult<Self> {
        if !model.matryoshka_dims.contains(&truncate_dim) {
            return Err(sift_core::SiftError::Embedding(format!(
                "Model '{}' does not support Matryoshka dimension {}. Supported: {:?}",
                model.name, truncate_dim, model.matryoshka_dims
            )));
        }
        if truncate_dim > model.dimensions {
            return Err(sift_core::SiftError::Embedding(format!(
                "truncate_dim {} exceeds model native dim {}",
                truncate_dim, model.dimensions
            )));
        }
        let mut embedder = Self::load_model(model_dir, model)?;
        embedder.truncate_dim = Some(truncate_dim);
        embedder.dimensions = truncate_dim;
        Ok(embedder)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_with_options(
        model_dir: &Path,
        model_name: &str,
        dimensions: usize,
        max_tokens: usize,
        pooling: PoolingStrategy,
        output_tensor: &'static str,
        search_prefix: &'static str,
        document_prefix: &'static str,
    ) -> SiftResult<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let num_cores = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);

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
            max_tokens,
            pooling,
            output_tensor,
            search_prefix,
            document_prefix,
            truncate_dim: None,
        })
    }

    /// Tokenize a batch of texts into flat, pre-padded tensors ready for ONNX.
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids, batch_size, seq_len)`
    /// as flat `Vec<i64>` buffers of shape `[batch_size × seq_len]`. This avoids
    /// the intermediate `Vec<Vec<i64>>` allocation per encoding and the subsequent
    /// flatten step — a single contiguous allocation per tensor instead.
    #[allow(clippy::type_complexity)]
    fn tokenize_batch(
        &self,
        texts: &[&str],
    ) -> SiftResult<(Vec<i64>, Vec<i64>, Vec<i64>, usize, usize)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| sift_core::SiftError::Embedding(format!("Tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_tokens))
            .max()
            .unwrap_or(0);

        let total = batch_size * seq_len;
        let mut input_ids = vec![0i64; total];
        let mut attention_mask = vec![0i64; total];
        let token_type_ids = vec![0i64; total]; // single-segment, always zero

        for (b, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len().min(self.max_tokens);
            let offset = b * seq_len;

            for i in 0..len {
                input_ids[offset + i] = i64::from(ids[i]);
                attention_mask[offset + i] = i64::from(mask[i]);
            }
        }

        Ok((
            input_ids,
            attention_mask,
            token_type_ids,
            batch_size,
            seq_len,
        ))
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

    fn cls_pooling(
        token_embeddings: &[f32],
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Vec<Vec<f32>> {
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let offset = b * seq_len * hidden_size;
            let mut pooled = token_embeddings[offset..offset + hidden_size].to_vec();
            l2_normalize_in_place(&mut pooled);
            results.push(pooled);
        }

        results
    }

    fn normalize_2d_embeddings(
        embeddings: &[f32],
        batch_size: usize,
        hidden_size: usize,
    ) -> Vec<Vec<f32>> {
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let offset = b * hidden_size;
            let mut row = embeddings[offset..offset + hidden_size].to_vec();
            l2_normalize_in_place(&mut row);
            results.push(row);
        }
        results
    }
}

fn l2_normalize_in_place(values: &mut [f32]) {
    let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in values {
            *v /= norm;
        }
    }
}

impl Embedder for OnnxEmbedder {
    fn embed_batch(&self, texts: &[&str]) -> SiftResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(batch_size = texts.len(), "Embedding batch");

        let (input_ids_flat, attention_mask_flat, token_type_ids_flat, batch_size, seq_len) =
            self.tokenize_batch(texts)?;

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

        // Extract the named output tensor declared in the model spec.
        // The previous index-based fallback silently produced wrong embeddings
        // for models whose first output isn't the embedding tensor.
        let output_array = outputs
            .get(self.output_tensor)
            .ok_or_else(|| {
                sift_core::SiftError::Embedding(format!(
                    "ONNX output tensor `{}` not found in model `{}`. \
                     Declared outputs: {:?}. Update `ModelSpec::output_tensor` \
                     for this model.",
                    self.output_tensor,
                    self.model_name,
                    outputs.keys().collect::<Vec<_>>(),
                ))
            })?
            .try_extract_tensor::<f32>()
            .map_err(|e| sift_core::SiftError::Embedding(format!("Extract error: {e}")))?;

        let shape = output_array.shape();
        let token_embeddings: Vec<f32> = output_array.iter().copied().collect();

        let results = match shape {
            [out_batch, hidden_size] if *out_batch == batch_size => {
                Self::normalize_2d_embeddings(&token_embeddings, batch_size, *hidden_size)
            }
            [out_batch, out_seq_len, hidden_size] if *out_batch == batch_size => {
                match self.pooling {
                    PoolingStrategy::MeanPooling => Self::mean_pooling(
                        &token_embeddings,
                        &attention_mask_flat,
                        batch_size,
                        *out_seq_len,
                        *hidden_size,
                    ),
                    PoolingStrategy::ClsToken => {
                        Self::cls_pooling(&token_embeddings, batch_size, *out_seq_len, *hidden_size)
                    }
                }
            }
            _ => {
                return Err(sift_core::SiftError::Embedding(format!(
                    "Unexpected ONNX embedding output shape: {shape:?}"
                )));
            }
        };

        // Matryoshka truncation. The pooled vectors above are L2-normalized;
        // truncating breaks unit length, so we renormalize. Models trained
        // with MRL keep most semantic content in the leading dimensions, so
        // a truncated-then-renormalized vector remains a valid embedding.
        if let Some(target_dim) = self.truncate_dim {
            let truncated: Vec<Vec<f32>> = results
                .into_iter()
                .map(|mut v| {
                    v.truncate(target_dim);
                    l2_normalize_in_place(&mut v);
                    v
                })
                .collect();
            return Ok(truncated);
        }

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn search_prefix(&self) -> &'static str {
        self.search_prefix
    }

    fn document_prefix(&self) -> &'static str {
        self.document_prefix
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

    #[test]
    fn cls_pooling_uses_first_token_per_batch() {
        #[rustfmt::skip]
        let token_embeddings: Vec<f32> = vec![
            3.0, 4.0, // batch 0, token 0 -> normalized [0.6, 0.8]
            9.0, 9.0, // batch 0, token 1 ignored
            0.0, 5.0, // batch 1, token 0 -> normalized [0.0, 1.0]
            7.0, 7.0, // batch 1, token 1 ignored
        ];

        let results = OnnxEmbedder::cls_pooling(&token_embeddings, 2, 2, 2);

        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 0.6).abs() < 1e-6);
        assert!((results[0][1] - 0.8).abs() < 1e-6);
        assert!((results[1][0] - 0.0).abs() < 1e-6);
        assert!((results[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_2d_embeddings_normalizes_each_row() {
        let embeddings = vec![3.0, 4.0, 0.0, 0.0, 0.0, 2.0];
        let results = OnnxEmbedder::normalize_2d_embeddings(&embeddings, 2, 3);

        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 0.6).abs() < 1e-6);
        assert!((results[0][1] - 0.8).abs() < 1e-6);
        assert!((results[0][2] - 0.0).abs() < 1e-6);
        assert!((results[1][0] - 0.0).abs() < 1e-6);
        assert!((results[1][1] - 0.0).abs() < 1e-6);
        assert!((results[1][2] - 1.0).abs() < 1e-6);
    }
}
