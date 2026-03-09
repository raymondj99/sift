//! Vision embedding module for image-to-vector conversion.
//!
//! Uses an ONNX vision model (Nomic Embed Vision) to produce 768-dimensional
//! embedding vectors in the same embedding space as the text model. This enables
//! cross-modal retrieval: text queries can match image embeddings and vice versa.

use image::imageops::FilterType;
use ort::session::Session;
use sift_core::{SiftError, SiftResult};
use std::path::Path;
use std::sync::Mutex;
use tracing::debug;

/// CLIP / ImageNet normalization constants.
///
/// These values are standard for models trained on ImageNet or using CLIP-style
/// preprocessing. They correspond to the per-channel mean and standard deviation
/// of the training dataset.
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];

/// Default input image resolution expected by CLIP-based vision encoders.
const DEFAULT_IMAGE_SIZE: u32 = 224;

/// ONNX Runtime-based vision embedder for image-to-vector conversion.
///
/// Wraps an ONNX vision model (e.g., Nomic Embed Vision) and produces
/// L2-normalized embedding vectors from raw image bytes. The session is
/// guarded by a [`Mutex`] for thread safety.
///
/// # Example
///
/// ```no_run
/// # use std::path::Path;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use sift_embed::vision::VisionEmbedder;
///
/// let embedder = VisionEmbedder::load(
///     Path::new("/models/nomic-embed-vision"),
///     "nomic-embed-vision-v1.5",
///     768,
/// )?;
///
/// let image_bytes = std::fs::read("photo.jpg")?;
/// let embedding = embedder.embed_image(&image_bytes)?;
/// assert_eq!(embedding.len(), 768);
/// # Ok(())
/// # }
/// ```
pub struct VisionEmbedder {
    /// ONNX Runtime inference session, mutex-guarded for thread safety.
    session: Mutex<Session>,
    /// Output embedding dimensionality (typically 768 for Nomic Vision).
    dimensions: usize,
    /// Human-readable model identifier.
    model_name: String,
    /// Expected input image resolution (width and height in pixels).
    image_size: u32,
}

impl VisionEmbedder {
    /// Load a vision embedding model from an ONNX file on disk.
    ///
    /// Expects a `model.onnx` file inside `model_dir`. No tokenizer is required
    /// for vision models since the input is pixel data rather than text.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing `model.onnx`.
    /// * `model_name` - Descriptive name for logging and identification.
    /// * `dimensions` - Expected output dimensionality (e.g., 768).
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::Embedding`] if the ONNX session cannot be created
    /// or the model file cannot be read.
    pub fn load(model_dir: &Path, model_name: &str, dimensions: usize) -> SiftResult<Self> {
        let model_path = model_dir.join("model.onnx");

        let session = Session::builder()
            .map_err(|e| SiftError::Embedding(format!("ONNX session builder error: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| SiftError::Embedding(format!("ONNX thread config error: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| {
                SiftError::Embedding(format!(
                    "Failed to load ONNX vision model from {}: {e}",
                    model_path.display(),
                ))
            })?;

        debug!(
            model_name,
            dimensions,
            path = %model_path.display(),
            "Loaded vision embedding model"
        );

        Ok(Self {
            session: Mutex::new(session),
            dimensions,
            model_name: model_name.to_string(),
            image_size: DEFAULT_IMAGE_SIZE,
        })
    }

    /// Embed a single image into a 768-dimensional vector.
    ///
    /// The image is decoded from raw bytes (supports PNG, JPEG, GIF, WebP, BMP),
    /// resized to the model's expected input resolution, normalized using CLIP
    /// constants, and passed through the ONNX vision encoder.
    ///
    /// The output vector is L2-normalized so it can be used directly for cosine
    /// similarity comparisons with text embeddings from the same model family.
    ///
    /// # Arguments
    ///
    /// * `image_bytes` - Raw image file bytes (any supported format).
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::Embedding`] if image decoding, preprocessing, or
    /// ONNX inference fails.
    pub fn embed_image(&self, image_bytes: &[u8]) -> SiftResult<Vec<f32>> {
        // Decode the image from raw bytes.
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| SiftError::Embedding(format!("Failed to decode image: {e}")))?;

        // Resize to the expected input dimensions using high-quality Lanczos3 filter.
        let resized = img.resize_exact(self.image_size, self.image_size, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        // Preprocess: convert to normalized CHW float tensor.
        let pixel_data = preprocess_image(&rgb, self.image_size);

        // Build the ONNX input tensor with shape [1, 3, H, W].
        let input_array = ndarray::Array4::from_shape_vec(
            (1, 3, self.image_size as usize, self.image_size as usize),
            pixel_data,
        )
        .map_err(|e| SiftError::Embedding(format!("Tensor shape error: {e}")))?;

        // Run inference.
        let session = self
            .session
            .lock()
            .map_err(|e| SiftError::Embedding(format!("Session lock poisoned: {e}")))?;

        let outputs = session
            .run(
                ort::inputs! {
                    "pixel_values" => input_array,
                }
                .map_err(|e| SiftError::Embedding(format!("ONNX input error: {e}")))?,
            )
            .map_err(|e| SiftError::Embedding(format!("ONNX vision inference failed: {e}")))?;

        // Extract the first output tensor.
        let first_output = &outputs[0];
        let output_tensor = first_output
            .try_extract_tensor::<f32>()
            .map_err(|e| SiftError::Embedding(format!("Failed to extract output tensor: {e}")))?;

        // Take the first `dimensions` values from the output.
        let raw: Vec<f32> = output_tensor
            .iter()
            .copied()
            .take(self.dimensions)
            .collect();

        if raw.len() < self.dimensions {
            return Err(SiftError::Embedding(format!(
                "Model output has {} values, expected at least {}",
                raw.len(),
                self.dimensions,
            )));
        }

        // L2 normalize the embedding vector.
        let embedding = l2_normalize(&raw);

        debug!(dimensions = embedding.len(), "Generated vision embedding");

        Ok(embedding)
    }

    /// Embed multiple images, returning one vector per image.
    ///
    /// Images are processed sequentially since the ONNX session is mutex-guarded.
    /// Each image is independently decoded, preprocessed, and embedded.
    ///
    /// # Arguments
    ///
    /// * `images` - Slice of raw image byte slices.
    ///
    /// # Errors
    ///
    /// Returns [`SiftError::Embedding`] if any image fails to embed. Processing
    /// stops at the first error.
    pub fn embed_image_batch(&self, images: &[&[u8]]) -> SiftResult<Vec<Vec<f32>>> {
        debug!(batch_size = images.len(), "Embedding image batch");

        let mut results = Vec::with_capacity(images.len());
        for (i, image_bytes) in images.iter().enumerate() {
            let embedding = self.embed_image(image_bytes).map_err(|e| {
                SiftError::Embedding(format!("Failed to embed image {i} in batch: {e}"))
            })?;
            results.push(embedding);
        }
        Ok(results)
    }

    /// Returns the output embedding dimensionality.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the expected input image size in pixels.
    pub fn image_size(&self) -> u32 {
        self.image_size
    }
}

/// Convert an RGB image to a normalized CHW float tensor.
///
/// Applies CLIP/ImageNet normalization:
///   `pixel = (pixel / 255.0 - mean) / std`
///
/// The output layout is CHW (channels-first), which is the standard format
/// for ONNX vision models. The returned vector has length `3 * size * size`.
///
/// # Arguments
///
/// * `rgb` - An RGB image, expected to already be the correct dimensions.
/// * `size` - The spatial dimension (both width and height).
pub fn preprocess_image(rgb: &image::RgbImage, size: u32) -> Vec<f32> {
    let (w, h) = rgb.dimensions();
    debug_assert_eq!(w, size, "Image width must match expected size");
    debug_assert_eq!(h, size, "Image height must match expected size");

    let npixels = (size * size) as usize;
    let mut chw = vec![0.0f32; 3 * npixels];

    for y in 0..size {
        for x in 0..size {
            let pixel = rgb.get_pixel(x, y);
            let idx = (y * size + x) as usize;

            // Channel 0: Red
            chw[idx] = (pixel[0] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
            // Channel 1: Green
            chw[npixels + idx] = (pixel[1] as f32 / 255.0 - CLIP_MEAN[1]) / CLIP_STD[1];
            // Channel 2: Blue
            chw[2 * npixels + idx] = (pixel[2] as f32 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2];
        }
    }

    chw
}

/// L2-normalize a vector in place, returning the normalized copy.
///
/// If the vector has zero magnitude, it is returned unchanged to avoid
/// division by zero.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `preprocess_image` produces a tensor with the correct shape
    /// (3 * size * size) and that values fall within the expected normalized range.
    #[test]
    fn test_preprocess_image_shape_and_range() {
        let size = 224u32;
        let img = image::RgbImage::from_fn(size, size, |x, y| {
            // Create a gradient pattern for variety.
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });

        let tensor = preprocess_image(&img, size);

        // Check total length: 3 channels * 224 * 224 = 150528
        assert_eq!(tensor.len(), 3 * (size as usize) * (size as usize));

        // Normalized values should be roughly in [-2, 3] for CLIP normalization.
        // Minimum: (0/255 - max_mean) / min_std ~ (0 - 0.48) / 0.26 ~ -1.85
        // Maximum: (255/255 - min_mean) / min_std ~ (1 - 0.408) / 0.26 ~ 2.28
        for &val in &tensor {
            assert!(
                val > -3.0 && val < 4.0,
                "Normalized value {val} outside expected range"
            );
        }
    }

    /// Verify that a small known image produces the expected normalized values.
    #[test]
    fn test_preprocess_image_known_values() {
        let size = 2u32;
        // All-black image: pixel value 0.
        let img = image::RgbImage::from_pixel(size, size, image::Rgb([0, 0, 0]));
        let tensor = preprocess_image(&img, size);

        // For a black pixel (value 0): normalized = (0/255 - mean) / std = -mean/std
        let expected_r = -CLIP_MEAN[0] / CLIP_STD[0];
        let expected_g = -CLIP_MEAN[1] / CLIP_STD[1];
        let expected_b = -CLIP_MEAN[2] / CLIP_STD[2];

        let npixels = (size * size) as usize;
        for i in 0..npixels {
            assert!(
                (tensor[i] - expected_r).abs() < 1e-5,
                "Red channel mismatch at pixel {i}"
            );
            assert!(
                (tensor[npixels + i] - expected_g).abs() < 1e-5,
                "Green channel mismatch at pixel {i}"
            );
            assert!(
                (tensor[2 * npixels + i] - expected_b).abs() < 1e-5,
                "Blue channel mismatch at pixel {i}"
            );
        }

        // Also test all-white (255).
        let img_white = image::RgbImage::from_pixel(size, size, image::Rgb([255, 255, 255]));
        let tensor_white = preprocess_image(&img_white, size);

        let expected_r_white = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        for (i, val) in tensor_white.iter().take(npixels).enumerate() {
            assert!(
                (val - expected_r_white).abs() < 1e-5,
                "White red channel mismatch at pixel {i}"
            );
        }
    }

    /// Verify that `preprocess_image` produces CHW layout (not HWC).
    #[test]
    fn test_preprocess_image_chw_layout() {
        let size = 4u32;
        // Create an image where R=100, G=150, B=200 everywhere.
        let img = image::RgbImage::from_pixel(size, size, image::Rgb([100, 150, 200]));
        let tensor = preprocess_image(&img, size);

        let npixels = (size * size) as usize;

        // All values in the R plane should be identical.
        let r_val = tensor[0];
        for (i, val) in tensor.iter().take(npixels).enumerate() {
            assert!((val - r_val).abs() < 1e-6, "R plane inconsistency at {i}");
        }

        // All values in the G plane should be identical and different from R.
        let g_val = tensor[npixels];
        for (i, val) in tensor[npixels..2 * npixels].iter().enumerate() {
            assert!((val - g_val).abs() < 1e-6, "G plane inconsistency at {i}");
        }

        // R and G should differ since the input pixel values are different.
        assert!(
            (r_val - g_val).abs() > 0.01,
            "R and G planes should have different values for different input channels"
        );
    }

    /// Verify L2 normalization produces a unit vector.
    #[test]
    fn test_l2_normalize_unit_vector() {
        let v = vec![3.0, 4.0];
        let normalized = l2_normalize(&v);

        // L2 norm of [3, 4] is 5, so normalized is [0.6, 0.8].
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Verify the result is a unit vector.
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Normalized vector should have unit norm, got {norm}"
        );
    }

    /// Verify L2 normalization with a higher-dimensional vector.
    #[test]
    fn test_l2_normalize_high_dimensional() {
        let v: Vec<f32> = (0..768).map(|i| (i as f32) * 0.01).collect();
        let normalized = l2_normalize(&v);

        assert_eq!(normalized.len(), 768);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "768-dim normalized vector should have unit norm, got {norm}"
        );
    }

    /// Verify L2 normalization handles the zero vector without panicking.
    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&v);

        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    /// Verify L2 normalization handles a single-element vector.
    #[test]
    fn test_l2_normalize_single_element() {
        let v = vec![5.0];
        let normalized = l2_normalize(&v);

        assert!((normalized[0] - 1.0).abs() < 1e-6);
    }

    /// Verify that invalid image bytes produce a descriptive error.
    #[test]
    fn test_embed_image_invalid_bytes() {
        // We cannot construct a full VisionEmbedder without a model file,
        // but we can verify that `image::load_from_memory` fails gracefully
        // on garbage bytes, which is the first step in `embed_image`.
        let garbage = b"this is not an image";
        let result = image::load_from_memory(garbage);
        assert!(result.is_err(), "Garbage bytes should fail to decode");
    }

    /// Verify that `embed_image` returns an error for empty input.
    #[test]
    fn test_embed_image_empty_bytes() {
        let result = image::load_from_memory(b"");
        assert!(result.is_err(), "Empty bytes should fail to decode");
    }

    /// Full integration test for `embed_image` -- requires a real model on disk.
    #[test]
    #[ignore]
    fn test_embed_image_integration() {
        // This test requires a Nomic Embed Vision ONNX model at the specified path.
        let model_dir = Path::new("/tmp/test-models/nomic-embed-vision");
        let embedder = VisionEmbedder::load(model_dir, "nomic-embed-vision-v1.5", 768)
            .expect("Failed to load vision model");

        // Create a minimal valid PNG in memory.
        let mut img = image::RgbImage::new(64, 64);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([128, 64, 192]);
        }
        let mut buf = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            encoder,
            img.as_raw(),
            64,
            64,
            image::ExtendedColorType::Rgb8,
        )
        .expect("Failed to encode test PNG");

        let embedding = embedder
            .embed_image(&buf)
            .expect("Failed to embed test image");
        assert_eq!(embedding.len(), 768);

        // Verify L2 normalization.
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedding should be L2-normalized, got norm {norm}"
        );
    }

    /// Batch embedding integration test -- requires a real model on disk.
    #[test]
    #[ignore]
    fn test_embed_image_batch_integration() {
        let model_dir = Path::new("/tmp/test-models/nomic-embed-vision");
        let embedder = VisionEmbedder::load(model_dir, "nomic-embed-vision-v1.5", 768)
            .expect("Failed to load vision model");

        // Create two distinct test images.
        let mut images_encoded = Vec::new();
        for color in &[[255u8, 0, 0], [0, 0, 255]] {
            let img = image::RgbImage::from_pixel(32, 32, image::Rgb(*color));
            let mut buf = Vec::new();
            let encoder = image::codecs::png::PngEncoder::new(&mut buf);
            image::ImageEncoder::write_image(
                encoder,
                img.as_raw(),
                32,
                32,
                image::ExtendedColorType::Rgb8,
            )
            .expect("Failed to encode test PNG");
            images_encoded.push(buf);
        }

        let image_refs: Vec<&[u8]> = images_encoded.iter().map(|b| b.as_slice()).collect();
        let embeddings = embedder
            .embed_image_batch(&image_refs)
            .expect("Batch embedding failed");

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
        assert_eq!(embeddings[1].len(), 768);

        // Different-colored images should produce different embeddings.
        let dot: f32 = embeddings[0]
            .iter()
            .zip(embeddings[1].iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot < 0.999,
            "Red and blue images should produce different embeddings, cosine similarity = {dot}"
        );
    }
}
