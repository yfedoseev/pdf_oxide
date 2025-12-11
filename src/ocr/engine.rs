//! Main OCR engine combining detection and recognition.
//!
//! The OcrEngine provides a high-level interface for performing OCR on images,
//! coordinating the detection and recognition pipelines.

use std::path::Path;

use image::DynamicImage;

use super::config::OcrConfig;
use super::detector::TextDetector;
use super::error::OcrResult;
use super::preprocessor::crop_text_region;
use super::recognizer::TextRecognizer;

/// Recognized text span with position and confidence.
#[derive(Debug, Clone)]
pub struct OcrSpan {
    /// Recognized text
    pub text: String,
    /// Quadrilateral bounding box [top-left, top-right, bottom-right, bottom-left]
    pub polygon: [[f32; 2]; 4],
    /// Overall confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Per-character confidence scores
    pub char_confidences: Vec<f32>,
}

impl OcrSpan {
    /// Convert OCR span to a TextSpan for integration with existing text extraction.
    ///
    /// This creates a TextSpan with:
    /// - Bounding box converted from polygon
    /// - Font size estimated from text height
    /// - Default styling (font name "OCR", normal weight, black color)
    ///
    /// # Arguments
    ///
    /// * `sequence` - Sequence number for reading order
    /// * `scale` - Scale factor to convert from image coordinates to PDF coordinates
    ///            (typically image_dpi / 72.0 to convert to points)
    pub fn to_text_span(&self, sequence: usize, scale: f32) -> crate::layout::text_block::TextSpan {
        use crate::geometry::Rect;
        use crate::layout::text_block::{Color, FontWeight, TextSpan};

        // Convert polygon to axis-aligned bounding box
        let min_x = self.polygon.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let max_x = self.polygon.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let min_y = self.polygon.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let max_y = self.polygon.iter().map(|p| p[1]).fold(f32::MIN, f32::max);

        // Apply scale to convert image coordinates to PDF coordinates
        let bbox = Rect::new(min_x / scale, min_y / scale, max_x / scale, max_y / scale);

        // Estimate font size from text height
        let height_pixels = max_y - min_y;
        let font_size = self.estimate_font_size(height_pixels, scale);

        TextSpan {
            text: self.text.clone(),
            bbox,
            font_name: "OCR".to_string(),
            font_size,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        }
    }

    /// Estimate font size in points from text box height.
    ///
    /// Uses heuristic: font_size ≈ height * 0.75 (accounting for descenders/ascenders)
    fn estimate_font_size(&self, height_pixels: f32, scale: f32) -> f32 {
        // Convert pixel height to points and apply heuristic
        // Typical text boxes include space for ascenders/descenders
        // so actual font size is about 75% of the box height
        let height_points = height_pixels / scale;
        (height_points * 0.75).clamp(6.0, 72.0) // Clamp to reasonable font sizes
    }

    /// Get the axis-aligned bounding box of the polygon.
    pub fn bounding_rect(&self) -> crate::geometry::Rect {
        use crate::geometry::Rect;

        let min_x = self.polygon.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        let max_x = self.polygon.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
        let min_y = self.polygon.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let max_y = self.polygon.iter().map(|p| p[1]).fold(f32::MIN, f32::max);

        Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }
}

/// Result of OCR processing on an image.
#[derive(Debug, Clone)]
pub struct OcrOutput {
    /// All recognized text spans
    pub spans: Vec<OcrSpan>,
    /// Average confidence across all spans
    pub total_confidence: f32,
}

impl OcrOutput {
    /// Get all text concatenated with spaces.
    pub fn text(&self) -> String {
        self.spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get text spans sorted by reading order (top-to-bottom, left-to-right).
    pub fn text_in_reading_order(&self) -> String {
        let mut spans: Vec<_> = self.spans.iter().collect();

        // Sort by Y position (top to bottom), then X (left to right)
        spans.sort_by(|a, b| {
            let y_a = a.polygon[0][1];
            let y_b = b.polygon[0][1];

            // If Y positions are similar (within 10 pixels), sort by X
            if (y_a - y_b).abs() < 10.0 {
                let x_a = a.polygon[0][0];
                let x_b = b.polygon[0][0];
                x_a.partial_cmp(&x_b).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                y_a.partial_cmp(&y_b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Convert all OCR spans to TextSpans for integration with layout analysis.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factor to convert from image coordinates to PDF coordinates
    ///            (typically image_dpi / 72.0 to convert to points)
    ///
    /// # Returns
    ///
    /// Vector of TextSpans sorted in reading order.
    pub fn to_text_spans(&self, scale: f32) -> Vec<crate::layout::text_block::TextSpan> {
        let mut spans_with_pos: Vec<_> = self.spans.iter().enumerate().collect();

        // Sort by reading order (top to bottom, left to right)
        spans_with_pos.sort_by(|(_, a), (_, b)| {
            let y_a = a.polygon[0][1];
            let y_b = b.polygon[0][1];

            if (y_a - y_b).abs() < 10.0 {
                let x_a = a.polygon[0][0];
                let x_b = b.polygon[0][0];
                x_a.partial_cmp(&x_b).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                y_a.partial_cmp(&y_b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Convert to TextSpans with sequence numbers
        spans_with_pos
            .iter()
            .enumerate()
            .map(|(seq, (_, ocr_span))| ocr_span.to_text_span(seq, scale))
            .collect()
    }
}

/// Main OCR engine for text extraction from images.
///
/// Combines text detection (DBNet++) and recognition (SVTR) models
/// for end-to-end OCR.
///
/// # Example
///
/// ```ignore
/// use pdf_oxide::ocr::{OcrEngine, OcrConfig};
/// use image::open;
///
/// let engine = OcrEngine::new(
///     "models/det.onnx",
///     "models/rec.onnx",
///     "models/en_dict.txt",
///     OcrConfig::default()
/// )?;
///
/// let image = open("document.png")?;
/// let result = engine.ocr_image(&image)?;
///
/// println!("Extracted text: {}", result.text());
/// ```
pub struct OcrEngine {
    detector: TextDetector,
    recognizer: TextRecognizer,
    config: OcrConfig,
}

impl OcrEngine {
    /// Create a new OCR engine from model file paths.
    ///
    /// # Arguments
    ///
    /// * `det_model_path` - Path to DBNet++ detection model
    /// * `rec_model_path` - Path to SVTR recognition model
    /// * `dict_path` - Path to character dictionary
    /// * `config` - OCR configuration
    pub fn new(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        config: OcrConfig,
    ) -> OcrResult<Self> {
        let detector = TextDetector::new(det_model_path, config.clone())?;
        let recognizer = TextRecognizer::new(rec_model_path, dict_path, config.clone())?;

        Ok(Self {
            detector,
            recognizer,
            config,
        })
    }

    /// Create a new OCR engine from model bytes (for bundled models).
    ///
    /// # Arguments
    ///
    /// * `det_model_bytes` - Detection model ONNX bytes
    /// * `rec_model_bytes` - Recognition model ONNX bytes
    /// * `dict_content` - Character dictionary content
    /// * `config` - OCR configuration
    pub fn from_bytes(
        det_model_bytes: &[u8],
        rec_model_bytes: &[u8],
        dict_content: &str,
        config: OcrConfig,
    ) -> OcrResult<Self> {
        let detector = TextDetector::from_bytes(det_model_bytes, config.clone())?;
        let recognizer = TextRecognizer::from_bytes(rec_model_bytes, dict_content, config.clone())?;

        Ok(Self {
            detector,
            recognizer,
            config,
        })
    }

    /// Perform OCR on an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    ///
    /// # Returns
    ///
    /// OCR result containing all recognized text spans with positions.
    pub fn ocr_image(&self, image: &DynamicImage) -> OcrResult<OcrOutput> {
        // Step 1: Detect text regions
        let boxes = self.detector.detect(image)?;

        if boxes.is_empty() {
            return Ok(OcrOutput {
                spans: Vec::new(),
                total_confidence: 0.0,
            });
        }

        // Step 2: Recognize text in each region
        let mut spans = Vec::new();
        let mut total_confidence = 0.0;

        for detected_box in &boxes {
            // Crop the text region
            let crop = crop_text_region(image, &detected_box.polygon)?;

            // Recognize text in the crop
            let recognition = self.recognizer.recognize(&crop)?;

            // Filter out low-confidence results
            if recognition.confidence >= self.config.rec_threshold
                && !recognition.text.trim().is_empty()
            {
                total_confidence += recognition.confidence;

                spans.push(OcrSpan {
                    text: recognition.text,
                    polygon: detected_box.polygon,
                    confidence: recognition.confidence,
                    char_confidences: recognition.char_confidences,
                });
            }
        }

        // Calculate average confidence
        let avg_confidence = if spans.is_empty() {
            0.0
        } else {
            total_confidence / spans.len() as f32
        };

        Ok(OcrOutput {
            spans,
            total_confidence: avg_confidence,
        })
    }

    /// Get reference to the detector.
    pub fn detector(&self) -> &TextDetector {
        &self.detector
    }

    /// Get reference to the recognizer.
    pub fn recognizer(&self) -> &TextRecognizer {
        &self.recognizer
    }

    /// Get the configuration.
    pub fn config(&self) -> &OcrConfig {
        &self.config
    }
}

// OcrEngine is Send + Sync because its components use Mutex

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_output_text() {
        let result = OcrOutput {
            spans: vec![
                OcrSpan {
                    text: "Hello".to_string(),
                    polygon: [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]],
                    confidence: 0.95,
                    char_confidences: vec![],
                },
                OcrSpan {
                    text: "World".to_string(),
                    polygon: [[60.0, 0.0], [110.0, 0.0], [110.0, 20.0], [60.0, 20.0]],
                    confidence: 0.92,
                    char_confidences: vec![],
                },
            ],
            total_confidence: 0.935,
        };

        assert_eq!(result.text(), "Hello World");
    }

    #[test]
    fn test_ocr_output_reading_order() {
        let result = OcrOutput {
            spans: vec![
                // Second line
                OcrSpan {
                    text: "Line2".to_string(),
                    polygon: [[0.0, 50.0], [50.0, 50.0], [50.0, 70.0], [0.0, 70.0]],
                    confidence: 0.9,
                    char_confidences: vec![],
                },
                // First line
                OcrSpan {
                    text: "Line1".to_string(),
                    polygon: [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]],
                    confidence: 0.9,
                    char_confidences: vec![],
                },
            ],
            total_confidence: 0.9,
        };

        // Should sort by Y position (top to bottom)
        assert_eq!(result.text_in_reading_order(), "Line1 Line2");
    }

    #[test]
    fn test_ocr_span() {
        let span = OcrSpan {
            text: "Test".to_string(),
            polygon: [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]],
            confidence: 0.98,
            char_confidences: vec![0.99, 0.97, 0.98, 0.99],
        };

        assert_eq!(span.text, "Test");
        assert!(span.confidence > 0.9);
    }
}
