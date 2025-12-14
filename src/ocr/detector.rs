//! Text detection using DBNet++ model.
//!
//! DBNet++ (Differentiable Binarization Network) is a text detection model
//! that produces a probability map indicating text regions.

use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

use super::config::OcrConfig;
use super::error::{OcrError, OcrResult};
use super::postprocessor::{extract_boxes, DetectedBox};
use super::preprocessor::preprocess_for_detection;

/// Text detector using DBNet++ ONNX model.
pub struct TextDetector {
    /// ONNX Runtime session (Mutex for thread-safe mutable access)
    session: Mutex<Option<Session>>,
    /// Model bytes for deferred loading
    #[allow(dead_code)]
    model_bytes: Option<Vec<u8>>,
    config: OcrConfig,
}

impl TextDetector {
    /// Create a new text detector from model file path.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the DBNet++ ONNX model file
    /// * `config` - OCR configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::ocr::{TextDetector, OcrConfig};
    ///
    /// let detector = TextDetector::new("models/det.onnx", OcrConfig::default())?;
    /// ```
    pub fn new(model_path: impl AsRef<Path>, config: OcrConfig) -> OcrResult<Self> {
        let model_bytes = std::fs::read(model_path.as_ref())
            .map_err(|e| OcrError::ModelLoadError(format!("Failed to read model file: {}", e)))?;

        Self::from_bytes(&model_bytes, config)
    }

    /// Create a new text detector from model bytes (for bundled models).
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - ONNX model data as bytes
    /// * `config` - OCR configuration
    pub fn from_bytes(model_bytes: &[u8], config: OcrConfig) -> OcrResult<Self> {
        // Build session with optimization
        let session = Session::builder()
            .map_err(|e| {
                OcrError::ModelLoadError(format!("Failed to create session builder: {}", e))
            })?
            .with_intra_threads(config.num_threads)
            .map_err(|e| OcrError::ModelLoadError(format!("Failed to set threads: {}", e)))?
            .commit_from_memory(model_bytes)
            .map_err(|e| OcrError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            session: Mutex::new(Some(session)),
            model_bytes: Some(model_bytes.to_vec()),
            config,
        })
    }

    /// Detect text regions in an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    ///
    /// # Returns
    ///
    /// Vector of detected text boxes with coordinates and confidence scores.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let boxes = detector.detect(&image)?;
    /// for box in boxes {
    ///     println!("Found text at {:?} (confidence: {})", box.polygon, box.confidence);
    /// }
    /// ```
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<DetectedBox>> {
        // Preprocess image
        let (input_tensor, scale) = preprocess_for_detection(image, self.config.det_max_side)?;

        // Run inference
        let prob_map = self.run_inference(&input_tensor)?;

        // Extract boxes from probability map
        let boxes = extract_boxes(
            prob_map.view(),
            self.config.det_threshold,
            self.config.box_threshold,
            self.config.max_candidates,
            self.config.unclip_ratio,
            scale,
        )?;

        Ok(boxes)
    }

    /// Run ONNX model inference.
    fn run_inference(&self, input: &ndarray::Array4<f32>) -> OcrResult<Array2<f32>> {
        let mut session_guard = self.session.lock().map_err(|e| {
            OcrError::InferenceError(format!("Failed to acquire session lock: {}", e))
        })?;

        let session = session_guard
            .as_mut()
            .ok_or_else(|| OcrError::InferenceError("Model session not initialized".to_string()))?;

        // Create input tensor reference from ndarray
        let input_tensor = TensorRef::from_array_view(input).map_err(|e| {
            OcrError::InferenceError(format!("Failed to create input tensor: {}", e))
        })?;

        // Run inference
        // DBNet++ typically has input named "x" or "images" and output named "sigmoid_0.tmp_0" or "output"
        let outputs = session
            .run(ort::inputs!["x" => input_tensor])
            .map_err(|e| OcrError::InferenceError(format!("Inference failed: {}", e)))?;

        // Extract output - DBNet++ outputs [N, 1, H, W] probability map
        // Get the first output (models typically have one output)
        let (_, output_tensor) = outputs
            .iter()
            .next()
            .ok_or_else(|| OcrError::InferenceError("No output tensor found".to_string()))?;

        let output_array = output_tensor
            .try_extract_array::<f32>()
            .map_err(|e| OcrError::InferenceError(format!("Failed to extract output: {}", e)))?;

        // Convert from [N, 1, H, W] to [H, W]
        let shape = output_array.shape();
        if shape.len() != 4 {
            return Err(OcrError::InferenceError(format!(
                "Unexpected output shape: {:?}, expected 4D tensor",
                shape
            )));
        }

        let height = shape[2];
        let width = shape[3];

        // Extract the probability map (first batch, first channel)
        let mut prob_map = Array2::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                prob_map[[y, x]] = output_array[[0, 0, y, x]];
            }
        }

        Ok(prob_map)
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.session.lock().map(|s| s.is_some()).unwrap_or(false)
    }
}

// TextDetector is Send + Sync because it uses Mutex<Session>

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_config() {
        // Test that config is properly stored
        let config = OcrConfig::builder()
            .det_threshold(0.4)
            .box_threshold(0.6)
            .build();

        assert!((config.det_threshold - 0.4).abs() < f32::EPSILON);
        assert!((config.box_threshold - 0.6).abs() < f32::EPSILON);
    }

    // Note: Integration tests with actual models will be in tests/ocr/
    // These require model files to be present
}
