//! Text recognition using SVTR model.
//!
//! SVTR (Scene Text Recognition with Visual and Linguistic Transformation)
//! recognizes text from cropped text region images.

use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use super::config::OcrConfig;
use super::error::{OcrError, OcrResult};
use super::preprocessor::preprocess_for_recognition;

/// Result of text recognition for a single text region.
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Recognized text
    pub text: String,
    /// Overall confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Per-character confidence scores
    pub char_confidences: Vec<f32>,
}

/// Text recognizer using SVTR ONNX model.
pub struct TextRecognizer {
    /// ONNX Runtime session (Mutex for thread-safe mutable access)
    session: Mutex<Option<Session>>,
    /// Model bytes for reference
    #[allow(dead_code)]
    model_bytes: Option<Vec<u8>>,
    dictionary: Vec<char>,
    config: OcrConfig,
}

impl TextRecognizer {
    /// Create a new text recognizer from model file and dictionary.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the SVTR ONNX model file
    /// * `dict_path` - Path to the character dictionary file
    /// * `config` - OCR configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pdf_oxide::ocr::{TextRecognizer, OcrConfig};
    ///
    /// let recognizer = TextRecognizer::new(
    ///     "models/rec.onnx",
    ///     "models/en_dict.txt",
    ///     OcrConfig::default()
    /// )?;
    /// ```
    pub fn new(
        model_path: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        config: OcrConfig,
    ) -> OcrResult<Self> {
        let model_bytes = std::fs::read(model_path.as_ref())
            .map_err(|e| OcrError::ModelLoadError(format!("Failed to read model file: {}", e)))?;

        let dict_content = std::fs::read_to_string(dict_path.as_ref())
            .map_err(|e| OcrError::DictionaryError(format!("Failed to read dictionary: {}", e)))?;

        Self::from_bytes(&model_bytes, &dict_content, config)
    }

    /// Create a new text recognizer from model bytes and dictionary string.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - ONNX model data as bytes
    /// * `dict_content` - Character dictionary as string (one char per line)
    /// * `config` - OCR configuration
    pub fn from_bytes(
        model_bytes: &[u8],
        dict_content: &str,
        config: OcrConfig,
    ) -> OcrResult<Self> {
        let dictionary = Self::parse_dictionary(dict_content)?;

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
            dictionary,
            config,
        })
    }

    /// Parse character dictionary from string.
    fn parse_dictionary(content: &str) -> OcrResult<Vec<char>> {
        let mut chars: Vec<char> = content
            .lines()
            .filter(|line| !line.is_empty())
            .filter_map(|line| line.chars().next())
            .collect();

        if chars.is_empty() {
            return Err(OcrError::DictionaryError("Dictionary is empty".to_string()));
        }

        // Add blank character at the end (for CTC decoding)
        chars.push('\0');

        Ok(chars)
    }

    /// Recognize text from a single cropped text region.
    ///
    /// # Arguments
    ///
    /// * `crop` - Cropped image of a text region
    ///
    /// # Returns
    ///
    /// Recognition result with text, confidence, and per-character scores.
    pub fn recognize(&self, crop: &DynamicImage) -> OcrResult<RecognitionResult> {
        // Preprocess the crop
        let input_tensor = preprocess_for_recognition(crop, self.config.rec_target_height)?;

        // Run inference
        self.run_inference(&input_tensor)
    }

    /// Recognize text from multiple cropped regions (batched).
    ///
    /// # Arguments
    ///
    /// * `crops` - Vector of cropped text region images
    ///
    /// # Returns
    ///
    /// Vector of recognition results.
    pub fn recognize_batch(&self, crops: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        // For now, process sequentially
        // TODO: Implement true batch processing for better performance
        crops.iter().map(|crop| self.recognize(crop)).collect()
    }

    /// Run ONNX model inference.
    fn run_inference(&self, input: &Array4<f32>) -> OcrResult<RecognitionResult> {
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
        // SVTR typically has input named "x" and output named "softmax_0.tmp_0" or similar
        let outputs = session
            .run(ort::inputs!["x" => input_tensor])
            .map_err(|e| OcrError::InferenceError(format!("Inference failed: {}", e)))?;

        // Extract output - SVTR outputs [N, W, num_classes] softmax scores
        // Get the first output (models typically have one output)
        let (_, output_tensor) = outputs
            .iter()
            .next()
            .ok_or_else(|| OcrError::InferenceError("No output tensor found".to_string()))?;

        let output_array = output_tensor
            .try_extract_array::<f32>()
            .map_err(|e| OcrError::InferenceError(format!("Failed to extract output: {}", e)))?;

        // Decode using CTC greedy decoding
        self.ctc_greedy_decode(&output_array)
    }

    /// CTC greedy decoding.
    ///
    /// Takes softmax output [N, T, C] and produces text by:
    /// 1. Taking argmax at each timestep
    /// 2. Removing consecutive duplicates
    /// 3. Removing blank tokens
    fn ctc_greedy_decode(&self, output: &ndarray::ArrayViewD<f32>) -> OcrResult<RecognitionResult> {
        let shape = output.shape();

        // Handle different output shapes
        let (seq_len, num_classes) = match shape.len() {
            2 => (shape[0], shape[1]),
            3 => (shape[1], shape[2]),
            _ => {
                return Err(OcrError::InferenceError(format!(
                    "Unexpected output shape: {:?}, expected 2D or 3D tensor",
                    shape
                )));
            },
        };

        let blank_idx = self.dictionary.len() - 1;
        let mut text = String::new();
        let mut char_confidences = Vec::new();
        let mut prev_idx = blank_idx;

        for t in 0..seq_len {
            // Find argmax and max confidence for this timestep
            let mut max_idx = 0;
            let mut max_conf = f32::MIN;

            for c in 0..num_classes {
                let prob = if shape.len() == 3 {
                    output[[0, t, c]]
                } else {
                    output[[t, c]]
                };

                if prob > max_conf {
                    max_conf = prob;
                    max_idx = c;
                }
            }

            // Skip if same as previous or blank
            if max_idx != prev_idx && max_idx != blank_idx && max_idx < self.dictionary.len() {
                let ch = self.dictionary[max_idx];
                if ch != '\0' {
                    text.push(ch);
                    char_confidences.push(max_conf);
                }
            }

            prev_idx = max_idx;
        }

        // Calculate overall confidence as geometric mean of character confidences
        let confidence = if char_confidences.is_empty() {
            0.0
        } else {
            let log_sum: f32 = char_confidences.iter().map(|c| c.ln()).sum();
            (log_sum / char_confidences.len() as f32).exp()
        };

        Ok(RecognitionResult {
            text,
            confidence,
            char_confidences,
        })
    }

    /// Get the character dictionary.
    pub fn dictionary(&self) -> &[char] {
        &self.dictionary
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.session.lock().map(|s| s.is_some()).unwrap_or(false)
    }
}

// TextRecognizer is Send + Sync because it uses Mutex<Session>

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dictionary() {
        let dict_content = "a\nb\nc\n1\n2\n3";
        let dict = TextRecognizer::parse_dictionary(dict_content).unwrap();

        // Should have 6 chars + 1 blank
        assert_eq!(dict.len(), 7);
        assert_eq!(dict[0], 'a');
        assert_eq!(dict[5], '3');
        assert_eq!(dict[6], '\0'); // Blank at end
    }

    #[test]
    fn test_parse_dictionary_empty() {
        let result = TextRecognizer::parse_dictionary("");
        assert!(result.is_err());
    }

    #[test]
    fn test_recognition_result() {
        let result = RecognitionResult {
            text: "Hello".to_string(),
            confidence: 0.95,
            char_confidences: vec![0.99, 0.98, 0.92, 0.93, 0.95],
        };

        assert_eq!(result.text, "Hello");
        assert!(result.confidence > 0.9);
        assert_eq!(result.char_confidences.len(), 5);
    }
}
