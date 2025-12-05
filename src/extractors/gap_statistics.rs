//! Statistical analysis of gaps between text spans.
//!
//! This module provides tools for analyzing the distribution of horizontal gaps
//! between consecutive text spans in a PDF document. Gap analysis is a fundamental
//! heuristic for detecting word boundaries, table structures, and column layouts.
//!
//! # Statistical Approach
//!
//! Instead of using fixed thresholds for gap detection, this module computes robust
//! statistics from the actual gap distribution in a document:
//!
//! - **Mean and Standard Deviation**: Overall spacing trends
//! - **Median and Percentiles**: Robust to outliers
//! - **IQR (Interquartile Range)**: Robust spread measure
//!
//! # Adaptive Thresholding
//!
//! The adaptive threshold is computed as a multiple of the median gap size,
//! optionally using IQR instead. This allows the threshold to automatically
//! adapt to different documents and font sizes.
//!
//! # Examples
//!
//! ```ignore
//! use pdf_oxide::extractors::gap_statistics::{
//!     analyze_document_gaps, AdaptiveThresholdConfig
//! };
//! use pdf_oxide::layout::TextSpan;
//!
//! let spans = vec![/* text spans from document */];
//!
//! // Use default adaptive threshold
//! let result = analyze_document_gaps(&spans, None);
//! println!("Threshold: {}pt", result.threshold_pt);
//!
//! // Use aggressive threshold for tight spacing
//! let config = AdaptiveThresholdConfig::aggressive();
//! let result = analyze_document_gaps(&spans, Some(config));
//! ```
//!
//! Phase 5.1

use crate::layout::TextSpan;
use log::debug;

/// Statistical summary of gaps between text spans.
///
/// All percentile values and gap measurements are in PDF points (1/72 inch).
/// This struct captures the complete distribution of horizontal spacing.
#[derive(Debug, Clone)]
pub struct GapStatistics {
    /// All measured gaps between consecutive spans (in points)
    pub gaps: Vec<f32>,
    /// Number of gaps measured
    pub count: usize,
    /// Minimum gap size (in points)
    pub min: f32,
    /// Maximum gap size (in points)
    pub max: f32,
    /// Mean (average) gap size (in points)
    pub mean: f32,
    /// Median gap size (50th percentile) (in points)
    pub median: f32,
    /// Standard deviation of gaps (in points)
    pub std_dev: f32,
    /// 25th percentile (first quartile) (in points)
    pub p25: f32,
    /// 75th percentile (third quartile) (in points)
    pub p75: f32,
    /// 10th percentile (in points)
    pub p10: f32,
    /// 90th percentile (in points)
    pub p90: f32,
}

impl GapStatistics {
    /// Get the interquartile range (IQR = p75 - p25).
    ///
    /// IQR is a robust measure of spread that is less sensitive to outliers
    /// than standard deviation.
    pub fn iqr(&self) -> f32 {
        self.p75 - self.p25
    }

    /// Get the range (max - min).
    pub fn range(&self) -> f32 {
        self.max - self.min
    }

    /// Calculate the coefficient of variation (std_dev / mean).
    ///
    /// Useful for understanding relative variability in gap sizes.
    /// Returns 0.0 if mean is 0 or negative.
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mean > 0.0 {
            self.std_dev / self.mean
        } else {
            0.0
        }
    }
}

/// Configuration for adaptive threshold calculation.
///
/// Determines how the threshold is computed from gap statistics.
/// All point values assume PDF points (1/72 inch).
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveThresholdConfig {
    /// Multiplier applied to median gap when computing threshold.
    ///
    /// **Default**: 1.5
    /// **Range**: 0.5 - 3.0 (values outside this may be unreasonable)
    ///
    /// Higher values → more conservative (fewer gaps marked as word boundaries)
    /// Lower values → more aggressive (more gaps marked as word boundaries)
    pub median_multiplier: f32,

    /// Minimum threshold in PDF points (floor).
    ///
    /// **Default**: 0.05pt (very small, close to tracking/kerning)
    /// **Range**: 0.01 - 0.2pt
    ///
    /// Prevents threshold from becoming too small even with small median gaps.
    pub min_threshold_pt: f32,

    /// Maximum threshold in PDF points (ceiling).
    ///
    /// **Default**: 1.0pt (about 1/72 inch, reasonable word spacing)
    /// **Range**: 0.5 - 2.0pt
    ///
    /// Prevents threshold from becoming unreasonably large.
    pub max_threshold_pt: f32,

    /// Use IQR instead of median for robust threshold calculation.
    ///
    /// **Default**: false
    ///
    /// When true: `threshold = (p75 - p25) * multiplier`
    /// When false: `threshold = median * multiplier`
    ///
    /// IQR-based approach is more robust to outliers but may require
    /// different multiplier values (typically 0.3 - 0.7).
    pub use_iqr: bool,

    /// Minimum number of samples required to compute meaningful statistics.
    ///
    /// **Default**: 10
    ///
    /// If fewer gaps exist, statistics cannot be reliably computed
    /// and the function returns a default threshold instead.
    pub min_samples: usize,
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 100.0, // Phase 7 FIX: Increased from 1.0pt to allow computed thresholds up to 100pt // Phase 7 FIX: Increased from 1.0pt (was clamping adaptive threshold too aggressively)
            use_iqr: false,
            min_samples: 10,
        }
    }
}

impl AdaptiveThresholdConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Balanced configuration (default multiplier: 1.5).
    ///
    /// Suitable for most PDF documents with standard spacing.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Aggressive configuration (lower multiplier: 1.2).
    ///
    /// Marks more gaps as word boundaries. Useful when:
    /// - Text has tight spacing
    /// - You want to break up large blocks more aggressively
    /// - False negatives (missed gaps) are worse than false positives
    pub fn aggressive() -> Self {
        Self {
            median_multiplier: 1.2,
            min_threshold_pt: 0.05,
            max_threshold_pt: 100.0, // Phase 7 FIX: Increased from 1.0pt to allow computed thresholds up to 100pt // Phase 7 FIX: Increased from 1.0pt
            use_iqr: false,
            min_samples: 10,
        }
    }

    /// Conservative configuration (higher multiplier: 2.0).
    ///
    /// Marks fewer gaps as word boundaries. Useful when:
    /// - Text has loose spacing
    /// - You want to avoid breaking up tightly-kerned text
    /// - False positives (extra gaps) are worse than false negatives
    pub fn conservative() -> Self {
        Self {
            median_multiplier: 2.0,
            min_threshold_pt: 0.05,
            max_threshold_pt: 100.0, // Phase 7 FIX: Increased from 1.0pt to allow computed thresholds up to 100pt // Phase 7 FIX: Increased from 1.0pt
            use_iqr: false,
            min_samples: 10,
        }
    }

    /// Optimized for policy documents with tight spacing (multiplier: 1.3).
    ///
    /// Policy documents often have:
    /// - Narrow margins
    /// - Tight justified alignment
    /// - Minimal word spacing
    ///
    /// This configuration requires larger gaps to be detected as boundaries,
    /// and sets higher minimum threshold to avoid false positives.
    pub fn policy_documents() -> Self {
        Self {
            median_multiplier: 1.3,
            min_threshold_pt: 0.08,
            max_threshold_pt: 100.0, // Phase 7 FIX: Increased from 1.0pt to allow computed thresholds up to 100pt // Phase 7 FIX: Increased from 1.0pt
            use_iqr: false,
            min_samples: 10,
        }
    }

    /// Optimized for academic papers with standard spacing (multiplier: 1.6).
    ///
    /// Academic papers typically have:
    /// - Standard margins
    /// - Generous word spacing
    /// - Single or double-column layouts
    ///
    /// This configuration is slightly more conservative than balanced
    /// to handle the higher baseline spacing.
    pub fn academic() -> Self {
        Self {
            median_multiplier: 1.6,
            min_threshold_pt: 0.2,
            max_threshold_pt: 100.0, // Phase 7 FIX: Increased from 1.0pt to allow computed thresholds up to 100pt // Phase 7 FIX: Increased from 1.0pt
            use_iqr: false,
            min_samples: 10,
        }
    }

    /// Create a configuration with custom multiplier.
    ///
    /// # Arguments
    ///
    /// * `multiplier` - Multiplier for median or IQR (typically 0.5 - 3.0)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::extractors::gap_statistics::AdaptiveThresholdConfig;
    ///
    /// let config = AdaptiveThresholdConfig::with_multiplier(1.4);
    /// ```
    pub fn with_multiplier(multiplier: f32) -> Self {
        Self {
            median_multiplier: multiplier,
            ..Default::default()
        }
    }

    /// Enable or disable IQR-based calculation.
    ///
    /// # Arguments
    ///
    /// * `use_iqr` - If true, use IQR instead of median
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::extractors::gap_statistics::AdaptiveThresholdConfig;
    ///
    /// let config = AdaptiveThresholdConfig::default()
    ///     .with_iqr(true);
    /// ```
    pub fn with_iqr(mut self, use_iqr: bool) -> Self {
        self.use_iqr = use_iqr;
        self
    }

    /// Set minimum threshold floor.
    pub fn with_min_threshold(mut self, min_pt: f32) -> Self {
        self.min_threshold_pt = min_pt;
        self
    }

    /// Set maximum threshold ceiling.
    pub fn with_max_threshold(mut self, max_pt: f32) -> Self {
        self.max_threshold_pt = max_pt;
        self
    }

    /// Set minimum number of samples required.
    pub fn with_min_samples(mut self, count: usize) -> Self {
        self.min_samples = count;
        self
    }
}

/// Result of adaptive threshold analysis.
///
/// Contains the computed threshold, underlying statistics if available,
/// and a reason string explaining how the threshold was determined.
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdResult {
    /// The computed threshold in PDF points.
    ///
    /// Use this value to classify gaps:
    /// - If `gap >= threshold_pt`: likely a word boundary
    /// - If `gap < threshold_pt`: likely tight spacing/kerning
    pub threshold_pt: f32,

    /// Statistical summary if available.
    ///
    /// None if:
    /// - No spans provided
    /// - Fewer gaps than `min_samples` in config
    /// - All gaps are identical (no variation)
    pub stats: Option<GapStatistics>,

    /// Explanation of how threshold was determined.
    ///
    /// Examples:
    /// - "Computed from 245 gaps: median=0.15pt * 1.5 = 0.225pt"
    /// - "Insufficient samples: 3 gaps < min_samples (10), using default 0.1pt"
    /// - "Single span: no gaps to analyze, using default 0.1pt"
    pub reason: String,
}

/// Extract horizontal gaps from text spans.
///
/// Measures the distance from the right edge of each span to the left edge
/// of the next span. Negative values indicate overlapping text.
///
/// # Arguments
///
/// * `spans` - Text spans sorted in reading order (typically by position)
///
/// # Returns
///
/// Vector of gap sizes in PDF points. Empty if fewer than 2 spans.
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::extractors::gap_statistics::extract_gaps;
/// use pdf_oxide::layout::TextSpan;
/// use pdf_oxide::geometry::Rect;
///
/// let spans = vec![
///     TextSpan {
///         bbox: Rect::new(10.0, 10.0, 30.0, 12.0),  // right edge at 40.0
///         // ...other fields...
///     },
///     TextSpan {
///         bbox: Rect::new(45.0, 10.0, 30.0, 12.0),  // left edge at 45.0
///         // ...other fields...
///     },
/// ];
///
/// let gaps = extract_gaps(&spans);
/// assert_eq!(gaps[0], 5.0);  // 45.0 - 40.0 = 5.0
/// ```
pub fn extract_gaps(spans: &[TextSpan]) -> Vec<f32> {
    if spans.len() < 2 {
        return Vec::new();
    }

    let mut gaps = Vec::with_capacity(spans.len() - 1);

    for i in 0..spans.len() - 1 {
        let current_right = spans[i].bbox.right();
        let next_left = spans[i + 1].bbox.left();
        let gap = next_left - current_right;
        gaps.push(gap);
    }

    gaps
}

/// Calculate comprehensive statistics from a list of gaps.
///
/// Computes mean, median, standard deviation, and multiple percentiles.
/// Returns None if the input is empty.
///
/// # Arguments
///
/// * `gaps` - Raw gap measurements in points
///
/// # Returns
///
/// Some(GapStatistics) if gaps is non-empty, None otherwise.
///
/// # Percentile Calculation
///
/// Uses linear interpolation between sorted values (NIST recommended method).
/// This provides smooth estimates even with small samples.
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::extractors::gap_statistics::calculate_statistics;
///
/// let gaps = vec![0.1, 0.2, 0.15, 0.25, 0.3, 0.18, 0.22];
/// let stats = calculate_statistics(gaps).unwrap();
///
/// println!("Mean: {}pt", stats.mean);
/// println!("Median: {}pt", stats.median);
/// println!("Std Dev: {}pt", stats.std_dev);
/// ```
pub fn calculate_statistics(mut gaps: Vec<f32>) -> Option<GapStatistics> {
    if gaps.is_empty() {
        return None;
    }

    let count = gaps.len();

    // Compute min and max
    let min = gaps.iter().copied().fold(f32::INFINITY, f32::min);
    let max = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute mean
    let sum: f32 = gaps.iter().sum();
    let mean = sum / count as f32;

    // Compute standard deviation
    let variance: f32 = gaps.iter().map(|&g| (g - mean).powi(2)).sum::<f32>() / count as f32;
    let std_dev = variance.sqrt();

    // Sort for percentile calculations
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate percentiles using linear interpolation
    let p10 = percentile(&gaps, 0.10);
    let p25 = percentile(&gaps, 0.25);
    let median = percentile(&gaps, 0.50);
    let p75 = percentile(&gaps, 0.75);
    let p90 = percentile(&gaps, 0.90);

    Some(GapStatistics {
        gaps,
        count,
        min,
        max,
        mean,
        median,
        std_dev,
        p25,
        p75,
        p10,
        p90,
    })
}

/// Determine adaptive threshold from gap statistics.
///
/// Uses the configuration to compute a threshold based on median or IQR,
/// then clamps the result to the configured bounds.
///
/// # Arguments
///
/// * `stats` - Gap statistics from the document
/// * `config` - Threshold configuration parameters
///
/// # Returns
///
/// Threshold value in PDF points.
///
/// # Calculation
///
/// If `config.use_iqr` is false:
/// ```text
/// base_threshold = stats.median * config.median_multiplier
/// ```
///
/// If `config.use_iqr` is true:
/// ```text
/// base_threshold = (stats.p75 - stats.p25) * config.median_multiplier
/// ```
///
/// Then clamped:
/// ```text
/// final_threshold = clamp(base_threshold, min_threshold_pt, max_threshold_pt)
/// ```
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::extractors::gap_statistics::{
///     calculate_statistics, determine_adaptive_threshold, AdaptiveThresholdConfig
/// };
///
/// let gaps = vec![0.1, 0.15, 0.2, 0.25, 0.3];
/// let stats = calculate_statistics(gaps).unwrap();
///
/// let config = AdaptiveThresholdConfig::balanced();
/// let threshold = determine_adaptive_threshold(&stats, &config);
///
/// println!("Threshold: {}pt", threshold);
/// ```
pub fn determine_adaptive_threshold(
    stats: &GapStatistics,
    config: &AdaptiveThresholdConfig,
) -> f32 {
    let base_threshold = if config.use_iqr {
        stats.iqr() * config.median_multiplier
    } else {
        stats.median * config.median_multiplier
    };

    // Clamp to configured bounds
    base_threshold
        .max(config.min_threshold_pt)
        .min(config.max_threshold_pt)
}

/// Detect word boundary threshold using percentile-based analysis.
///
/// Uses the 75th percentile of positive gaps as the word spacing threshold.
/// This naturally falls at the boundary between letter-spacing (tight, ~70% of gaps)
/// and word-spacing (wider, ~25% of gaps).
///
/// # Algorithm
///
/// 1. Filter gaps to only positive values (negative gaps indicate overlaps/kerning)
/// 2. Sort positive gaps
/// 3. Compute 75th percentile: approximately where letter-spacing ends, word-spacing begins
/// 4. Return percentile if within reasonable bounds (2-10pt)
///
/// # Returns
///
/// `Some(threshold)` if percentile falls in reasonable range (2-10pt)
/// `None` if insufficient data or percentile out of bounds
///
/// # Rationale
///
/// In typical documents:
/// - ~75% of gaps are letter-spacing (tight, 2-4pt)
/// - ~25% of gaps are word-spacing (wider, 4-10pt)
/// - P75 naturally marks the transition
///
/// This is more robust than looking for the "largest jump" because:
/// - Handles diverse PDF structures uniformly
/// - Adapts to document's actual gap distribution
/// - Avoids detecting layout breaks (which are far beyond word-spacing)
fn detect_word_boundary_threshold(spans: &[TextSpan]) -> Option<f32> {
    // Extract gaps
    let mut gaps: Vec<f32> = spans.windows(2)
        .map(|w| w[1].bbox.left() - w[0].bbox.right())
        .filter(|g| *g > 0.0)  // Only positive gaps
        .collect();

    if gaps.len() < 10 {
        return None; // Not enough data for percentile
    }

    // Sort gaps
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute 75th percentile using linear interpolation
    let p75 = percentile(&gaps, 0.75);

    // Accept if threshold is in reasonable range for word spacing
    if p75 >= 2.0 && p75 <= 10.0 {
        debug!("Percentile-based threshold: P75 = {:.4}pt", p75);
        Some(p75)
    } else {
        debug!("Percentile-based threshold: P75 = {:.4}pt (out of bounds 2-10pt)", p75);
        None
    }
}

/// Analyze gap statistics for an entire document and compute adaptive threshold.
///
/// This is the main entry point for gap analysis. It:
/// 1. Extracts gaps from consecutive spans
/// 2. Attempts bimodal detection first
/// 3. Falls back to adaptive threshold computation
/// 4. Computes statistics if sufficient gaps exist
/// 5. Provides detailed reasoning in the result
///
/// # Arguments
///
/// * `spans` - Text spans from the document (should be sorted by position)
/// * `config` - Configuration (uses default if None)
///
/// # Returns
///
/// AdaptiveThresholdResult containing:
/// - `threshold_pt`: The computed threshold for gap detection
/// - `stats`: Optional statistics (None if insufficient data)
/// - `reason`: Explanation of how threshold was determined
///
/// # Edge Cases
///
/// - **No spans or single span**: Returns default threshold of 0.1pt
/// - **Insufficient gaps**: Returns default threshold if gaps < min_samples
/// - **All identical gaps**: Computes threshold normally (std_dev = 0)
/// - **Very tight spacing**: Threshold is clamped to max_threshold_pt
/// - **Very loose spacing**: Threshold is clamped to max_threshold_pt
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::extractors::gap_statistics::{
///     analyze_document_gaps, AdaptiveThresholdConfig
/// };
/// use pdf_oxide::layout::TextSpan;
///
/// let spans = vec![/* extracted text spans */];
///
/// // With default config
/// let result = analyze_document_gaps(&spans, None);
/// println!("Threshold: {}pt ({})", result.threshold_pt, result.reason);
///
/// // With custom config
/// let config = AdaptiveThresholdConfig::aggressive();
/// let result = analyze_document_gaps(&spans, Some(config));
/// ```
pub fn analyze_document_gaps(
    spans: &[TextSpan],
    config: Option<AdaptiveThresholdConfig>,
) -> AdaptiveThresholdResult {
    let config = config.unwrap_or_default();

    debug!(
        "Analyzing {} spans with config: multiplier={}, min={}pt, max={}pt, iqr={}",
        spans.len(),
        config.median_multiplier,
        config.min_threshold_pt,
        config.max_threshold_pt,
        config.use_iqr
    );

    // Handle edge case: no spans or single span
    if spans.len() < 2 {
        let reason = if spans.is_empty() {
            "No spans provided".to_string()
        } else {
            "Single span: no gaps to analyze".to_string()
        };

        debug!("{}, using default threshold", reason);

        return AdaptiveThresholdResult {
            threshold_pt: 0.1,
            stats: None,
            reason,
        };
    }

    // Try bimodal detection first (more robust for complex PDFs)
    if let Some(bimodal_threshold) = detect_word_boundary_threshold(spans) {
        let reason =
            format!("Bimodal detection: identified word boundary at {:.4}pt", bimodal_threshold);
        debug!("Using bimodal threshold: {}", reason);

        return AdaptiveThresholdResult {
            threshold_pt: bimodal_threshold,
            stats: None,
            reason,
        };
    }

    // Fallback to adaptive threshold computation
    // Extract gaps
    let gaps = extract_gaps(spans);

    debug!("Extracted {} gaps from {} spans", gaps.len(), spans.len());

    // Check if we have sufficient samples
    if gaps.len() < config.min_samples {
        let reason = format!(
            "Insufficient samples: {} gaps < min_samples ({}), using default",
            gaps.len(),
            config.min_samples
        );

        debug!("{}", reason);

        return AdaptiveThresholdResult {
            threshold_pt: 0.1,
            stats: None,
            reason,
        };
    }

    // Filter out negative gaps before computing statistics
    // (negative gaps represent text overlaps/kerning, not word boundaries)
    let positive_gaps: Vec<f32> = gaps.iter().filter(|g| **g > 0.0).copied().collect();

    let gaps_to_analyze = if positive_gaps.len() >= 10 {
        debug!(
            "Filtered to {} positive gaps (from {} total gaps)",
            positive_gaps.len(),
            gaps.len()
        );
        positive_gaps
    } else {
        debug!("Not enough positive gaps ({}) to filter, using all gaps", positive_gaps.len());
        gaps
    };

    // Calculate statistics
    let stats = match calculate_statistics(gaps_to_analyze) {
        Some(s) => s,
        None => {
            let reason = "Failed to calculate statistics".to_string();
            debug!("{}", reason);

            return AdaptiveThresholdResult {
                threshold_pt: 0.1,
                stats: None,
                reason,
            };
        },
    };

    // Determine threshold
    let threshold_pt = determine_adaptive_threshold(&stats, &config);

    let base_value = if config.use_iqr {
        format!("IQR={:.3}pt", stats.iqr())
    } else {
        format!("median={:.3}pt", stats.median)
    };

    let reason = format!(
        "Computed from {} gaps: {} * {:.1} = {:.3}pt (clamped to {:.3}pt)",
        stats.count,
        base_value,
        config.median_multiplier,
        if config.use_iqr {
            stats.iqr() * config.median_multiplier
        } else {
            stats.median * config.median_multiplier
        },
        threshold_pt
    );

    debug!("Threshold analysis: {}", reason);

    AdaptiveThresholdResult {
        threshold_pt,
        stats: Some(stats),
        reason,
    }
}

/// Document type classification based on gap statistics.
///
/// Classifies PDF documents into types based on inter-word gap distribution.
/// This enables adaptive thresholding to handle different document layouts effectively.
///
/// # Document Types
///
/// - **Academic**: Research papers and technical documents
///   - Typical median gap: 1.5-3.0pt (standard word spacing)
///   - Gap variance: high (due to column boundaries)
///   - Examples: ArXiv papers, conference proceedings
///
/// - **Policy**: Legal, policy, and formal documents
///   - Typical median gap: <0.8pt (tight justified spacing)
///   - Gap variance: low (consistent margins and alignment)
///   - Examples: Code of Conduct, legal agreements, contracts
///
/// - **Mixed**: Variable layouts with multiple structure types
///   - Typical median gap: 0.8-1.5pt (variable)
///   - Gap variance: high relative to median
///   - Examples: Reports with tables, forms with mixed content
///
/// # References
///
/// Research findings from LA-PDFText, pdfminer.six, PDFBox, and iText
/// all indicate that document-type-aware adaptive thresholds outperform fixed thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentType {
    /// Academic papers: wide word gaps, multi-column layout
    Academic,
    /// Policy documents: tight justified layout
    Policy,
    /// Mixed documents: variable layout
    Mixed,
}

impl DocumentType {
    /// Detect document type from inter-word gap statistics.
    ///
    /// Uses gap distribution analysis to classify documents into types.
    /// This enables adaptive thresholding tailored to each document type.
    ///
    /// # Algorithm
    ///
    /// 1. **Tight spacing detection** (Policy): median_gap < 0.8pt indicates justified text
    /// 2. **High variance detection** (Academic): CV > 0.8 indicates column boundaries
    /// 3. **Variable spacing** (Mixed): High relative variance or unclear patterns
    ///
    /// # Arguments
    ///
    /// * `gaps` - Inter-word gaps in points (pt)
    ///
    /// # Returns
    ///
    /// The detected DocumentType, defaults to Mixed if insufficient data
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let gaps = vec![0.2, 0.25, 0.3, 0.25, 0.2]; // Tight gaps
    /// let doc_type = DocumentType::detect(&gaps);
    /// assert_eq!(doc_type, DocumentType::Policy);
    /// ```
    pub fn detect(gaps: &[f32]) -> Self {
        // Need sufficient samples for reliable detection
        if gaps.is_empty() || gaps.len() < 5 {
            debug!("Insufficient gaps ({}) for document type detection, using Mixed", gaps.len());
            return Self::Mixed;
        }

        // Calculate basic statistics
        let median = Self::median(gaps);
        let std_dev = Self::standard_deviation(gaps);
        let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;

        // Avoid division by zero
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        debug!(
            "Document detection: {} gaps, median={:.3}pt, mean={:.3}pt, cv={:.3}",
            gaps.len(),
            median,
            mean,
            coefficient_of_variation
        );

        // Heuristic 1: Tight median gap suggests policy document (justified text)
        // Policy documents typically have median gap < 0.8pt with tight std_dev
        if median < 0.8 && std_dev < 0.5 {
            debug!(
                "Detected Policy type: median {:.3}pt < 0.8pt and std_dev {:.3}pt < 0.5pt",
                median, std_dev
            );
            return Self::Policy;
        }

        // Heuristic 2: High gap variance suggests academic document (column boundaries)
        // Academic papers have multiple gaps: tight (within columns) and wide (between columns)
        // Result: high coefficient of variation (CV > 0.8)
        if coefficient_of_variation > 0.8 {
            debug!(
                "Detected Academic type: coefficient of variation {:.3} > 0.8",
                coefficient_of_variation
            );
            return Self::Academic;
        }

        // Heuristic 3: Variable relative variance (stddev relative to median)
        // Indicates mixed layouts with variable structure (tables, forms, etc.)
        let relative_variance = if median > 0.0 { std_dev / median } else { 0.0 };

        if relative_variance > 0.5 && median > 0.5 {
            debug!(
                "Detected Mixed type: relative variance {:.3} > 0.5 with median {:.3}pt",
                relative_variance, median
            );
            return Self::Mixed;
        }

        // Default: Mixed if no clear pattern emerges
        debug!(
            "Using Mixed type (default): median={:.3}pt, cv={:.3}, rel_var={:.3}",
            median, coefficient_of_variation, relative_variance
        );
        Self::Mixed
    }

    /// Get recommended threshold multiplier for this document type.
    ///
    /// Returns the ideal median multiplier for adaptive threshold calculation.
    /// These values are calibrated from research in LA-PDFText and pdfminer.six.
    ///
    /// # Returns
    ///
    /// - Academic: 1.6 (more conservative, handle wide column gaps)
    /// - Policy: 1.2 (more aggressive, detect tight justified spacing)
    /// - Mixed: 1.5 (balanced)
    pub fn threshold_multiplier(&self) -> f32 {
        match self {
            Self::Academic => 1.6,
            Self::Policy => 1.2,
            Self::Mixed => 1.5,
        }
    }

    /// Get minimum threshold floor for this document type (in points).
    ///
    /// Prevents threshold from becoming too small even with small median gaps.
    ///
    /// # Returns
    ///
    /// - Academic: 0.2pt (allows detection in dense academic layouts)
    /// - Policy: 0.05pt (catches very tight justified spacing)
    /// - Mixed: 0.1pt (balanced)
    pub fn min_threshold_pt(&self) -> f32 {
        match self {
            Self::Academic => 0.2,
            Self::Policy => 0.05,
            Self::Mixed => 0.1,
        }
    }

    /// Get profile-specific adaptive threshold configuration.
    ///
    /// Returns an AdaptiveThresholdConfig tuned for this document type.
    /// Recommended for use in `analyze_document_gaps()` and span merging.
    ///
    /// # Returns
    ///
    /// Configuration with thresholds and multipliers tailored to document type
    pub fn get_adaptive_config(&self) -> AdaptiveThresholdConfig {
        match self {
            Self::Academic => AdaptiveThresholdConfig {
                median_multiplier: 1.6,
                min_threshold_pt: 0.2,
                max_threshold_pt: 100.0,
                use_iqr: false,
                min_samples: 10,
            },
            Self::Policy => AdaptiveThresholdConfig {
                median_multiplier: 1.2,
                min_threshold_pt: 0.05,
                max_threshold_pt: 100.0,
                use_iqr: false,
                min_samples: 10,
            },
            Self::Mixed => AdaptiveThresholdConfig::default(),
        }
    }

    /// Get human-readable document type name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Academic => "Academic",
            Self::Policy => "Policy",
            Self::Mixed => "Mixed",
        }
    }

    /// Calculate median of a slice of f32 values.
    fn median(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate standard deviation of a slice of f32 values.
    fn standard_deviation(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
}

/// Document profile for adaptive threshold tuning.
///
/// Detects the document type based on gap statistics to apply profile-specific
/// threshold configurations. This pattern follows pdfminer.six's approach of
/// analyzing document characteristics and adapting extraction parameters accordingly.
///
/// # Profiles
///
/// - **Academic**: Papers with standard spacing and column layouts.
///   - Typical gap variance: high (columns create wide gaps)
///   - Threshold: slightly conservative (1.6x multiplier)
///
/// - **Policy**: Formal documents with tight, justified spacing.
///   - Typical median gap: very small (<0.5pt)
///   - Threshold: more aggressive (1.2x multiplier)
///
/// - **Default**: Mixed or uncertain document type.
///   - Balanced profile (1.5x multiplier)
///
/// # References
///
/// Based on patterns from pdfminer.six:
/// - [pdfminer.six LAParams](https://pdfminersix.readthedocs.io/en/latest/topic/converting_pdf_to_text.html)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentProfile {
    /// Academic papers: standard spacing, column layouts
    Academic,
    /// Policy documents: tight spacing, justified text
    Policy,
    /// Mixed/unknown: balanced defaults
    Default,
}

impl DocumentProfile {
    /// Detect document profile from gap statistics.
    ///
    /// Uses gap distribution analysis to infer document type:
    /// - Tight median gap (< 0.5pt) → Policy document
    /// - High gap variance (CV > 0.8) → Academic with columns
    /// - Otherwise → Default/balanced
    ///
    /// # Arguments
    ///
    /// * `spans` - Text spans from document (for analysis)
    /// * `existing_stats` - Optional pre-computed gap statistics (optimization)
    ///
    /// # Returns
    ///
    /// The detected DocumentProfile
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::extractors::gap_statistics::DocumentProfile;
    ///
    /// let profile = DocumentProfile::detect(&spans, None);
    /// println!("Detected profile: {:?}", profile);
    /// ```
    pub fn detect(spans: &[TextSpan], existing_stats: Option<&GapStatistics>) -> Self {
        // If stats provided, use them; otherwise analyze gaps
        let stats = if let Some(s) = existing_stats {
            s.clone()
        } else {
            // Quick analysis: extract gaps and compute basic stats
            let gaps = extract_gaps(spans);
            if gaps.len() < 10 {
                // Not enough data for reliable detection
                return Self::Default;
            }

            match calculate_statistics(gaps) {
                Some(s) => s,
                None => return Self::Default,
            }
        };

        // Heuristic 1: Tight median gap suggests policy document
        if stats.median < 0.5 {
            debug!("Detected Policy profile: median gap {:.3}pt < 0.5pt", stats.median);
            return Self::Policy;
        }

        // Heuristic 2: High gap variance suggests academic document (columns)
        let cv = stats.coefficient_of_variation();
        if cv > 0.8 {
            debug!("Detected Academic profile: coefficient of variation {:.3} > 0.8", cv);
            return Self::Academic;
        }

        debug!("Using Default profile: median={:.3}pt, CV={:.3}", stats.median, cv);
        Self::Default
    }

    /// Get profile-specific adaptive threshold configuration.
    ///
    /// Returns tuned thresholds for this document profile.
    /// Based on pdfminer.six's LAParams approach:
    /// - Aggressive for tight-spaced documents
    /// - Conservative for loose-spaced documents
    ///
    /// # Returns
    ///
    /// AdaptiveThresholdConfig optimized for this profile
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::extractors::gap_statistics::{DocumentProfile, AdaptiveThresholdConfig};
    ///
    /// let profile = DocumentProfile::Academic;
    /// let config = profile.get_config();
    /// assert_eq!(config.median_multiplier, 1.6);
    /// ```
    pub fn get_config(&self) -> AdaptiveThresholdConfig {
        match self {
            Self::Academic => AdaptiveThresholdConfig {
                median_multiplier: 1.6,
                min_threshold_pt: 0.1,
                max_threshold_pt: 100.0,
                use_iqr: false,
                min_samples: 10,
            },
            Self::Policy => AdaptiveThresholdConfig {
                median_multiplier: 1.2, // More aggressive (sensitive to gaps)
                min_threshold_pt: 0.05,
                max_threshold_pt: 100.0,
                use_iqr: false,
                min_samples: 10,
            },
            Self::Default => AdaptiveThresholdConfig::balanced(),
        }
    }

    /// Get human-readable profile name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Academic => "Academic",
            Self::Policy => "Policy",
            Self::Default => "Default",
        }
    }
}

/// Helper function to compute percentiles using linear interpolation.
///
/// Uses the NIST-recommended method:
/// - For sorted array of length n, to compute percentile p (0.0 - 1.0):
///   - Calculate index: `i = p * (n - 1)`
///   - If i is not an integer, interpolate between adjacent values
///
/// # Arguments
///
/// * `sorted_values` - Values in ascending order
/// * `percentile` - Percentile to compute (0.0 - 1.0)
///
/// # Returns
///
/// Interpolated percentile value.
fn percentile(sorted_values: &[f32], percentile: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let index = percentile * (sorted_values.len() - 1) as f32;
    let lower_index = index.floor() as usize;
    let upper_index = (lower_index + 1).min(sorted_values.len() - 1);

    if lower_index == upper_index {
        sorted_values[lower_index]
    } else {
        let fraction = index - lower_index as f32;
        sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_single_value() {
        let values = vec![5.0];
        assert_eq!(percentile(&values, 0.5), 5.0);
    }

    #[test]
    fn test_percentile_two_values() {
        let values = vec![1.0, 3.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 3.0);
        assert_eq!(percentile(&values, 0.5), 2.0);
    }

    #[test]
    fn test_percentile_many_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 10.0);
        assert_eq!(percentile(&values, 0.5), 5.5);
    }

    #[test]
    fn test_extract_gaps() {
        use crate::geometry::Rect;

        let spans = vec![
            TextSpan {
                text: "Hello".to_string(),
                bbox: Rect::new(0.0, 0.0, 30.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
            },
            TextSpan {
                text: "World".to_string(),
                bbox: Rect::new(35.0, 0.0, 30.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
            },
        ];

        let gaps = extract_gaps(&spans);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], 5.0); // 35.0 - 30.0
    }

    #[test]
    fn test_extract_gaps_empty() {
        let gaps = extract_gaps(&[]);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_calculate_statistics() {
        let gaps = vec![0.1, 0.2, 0.15, 0.25, 0.3];
        let stats = calculate_statistics(gaps).unwrap();

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 0.1);
        assert_eq!(stats.max, 0.3);
        assert!(stats.mean > 0.19 && stats.mean < 0.21); // approx 0.20
    }

    #[test]
    fn test_calculate_statistics_empty() {
        let gaps = vec![];
        assert!(calculate_statistics(gaps).is_none());
    }

    #[test]
    fn test_gap_statistics_iqr() {
        let gaps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_statistics(gaps).unwrap();
        let iqr = stats.iqr();
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_adaptive_threshold_config_defaults() {
        let config = AdaptiveThresholdConfig::default();
        assert_eq!(config.median_multiplier, 1.5);
        assert_eq!(config.min_threshold_pt, 0.05);
        // Phase 7 FIX: max_threshold_pt was increased from 1.0 to 100.0
        // to allow computed thresholds for documents with larger word spacing
        assert_eq!(config.max_threshold_pt, 100.0);
        assert!(!config.use_iqr);
        assert_eq!(config.min_samples, 10);
    }

    #[test]
    fn test_adaptive_threshold_config_aggressive() {
        let config = AdaptiveThresholdConfig::aggressive();
        assert_eq!(config.median_multiplier, 1.2);
    }

    #[test]
    fn test_adaptive_threshold_config_conservative() {
        let config = AdaptiveThresholdConfig::conservative();
        assert_eq!(config.median_multiplier, 2.0);
    }

    #[test]
    fn test_determine_threshold_clamping() {
        let gaps = vec![0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        let stats = calculate_statistics(gaps).unwrap();
        let config = AdaptiveThresholdConfig::default();

        let threshold = determine_adaptive_threshold(&stats, &config);
        assert!(threshold >= config.min_threshold_pt);
        assert!(threshold <= config.max_threshold_pt);
    }

    #[test]
    fn test_analyze_document_gaps_empty() {
        let result = analyze_document_gaps(&[], None);
        assert_eq!(result.threshold_pt, 0.1);
        assert!(result.stats.is_none());
    }

    #[test]
    fn test_analyze_document_gaps_insufficient_samples() {
        use crate::geometry::Rect;

        let spans = vec![
            TextSpan {
                text: "A".to_string(),
                bbox: Rect::new(0.0, 0.0, 10.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: 0,
                split_boundary_before: false,
            },
            TextSpan {
                text: "B".to_string(),
                bbox: Rect::new(15.0, 0.0, 10.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: 1,
                split_boundary_before: false,
            },
        ];

        let result = analyze_document_gaps(&spans, None);
        assert_eq!(result.threshold_pt, 0.1);
        assert!(result.stats.is_none());
    }

    // ========== DocumentType Tests ==========

    #[test]
    fn test_document_type_policy_detection() {
        // Policy documents: tight gaps (< 0.8pt) with low variance
        let gaps = vec![0.2, 0.25, 0.3, 0.25, 0.2, 0.28, 0.22, 0.26, 0.24, 0.23];
        let doc_type = DocumentType::detect(&gaps);
        assert_eq!(doc_type, DocumentType::Policy, "Expected Policy type for tight spacing");
    }

    #[test]
    fn test_document_type_academic_detection() {
        // Academic documents: high variance from column boundaries
        // Tight gaps (within columns): ~1pt
        // Wide gaps (between columns): ~20pt
        // Result: high coefficient of variation
        let gaps = vec![
            1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.8, 1.2,  // Column 1: tight spacing
            25.0, // Column boundary
            1.1, 0.9, 1.2, 1.0, 1.1, 0.9, 1.2,  // Column 2: tight spacing
            24.0, // Column boundary
            1.0, 1.1, 0.95, 1.15, // Column 3: tight spacing
        ];
        let doc_type = DocumentType::detect(&gaps);
        assert_eq!(
            doc_type,
            DocumentType::Academic,
            "Expected Academic type for high gap variance"
        );
    }

    #[test]
    fn test_document_type_mixed_detection() {
        // Mixed documents: moderate gaps with variable patterns
        // Not tight enough for policy, not variable enough for academic
        let gaps = vec![
            0.8, 1.2, 0.9, 1.5, 1.1, 0.85, 1.3, 0.95, 1.2, 1.0, 1.1, 0.9, 1.2, 1.05, 0.88,
        ];
        let doc_type = DocumentType::detect(&gaps);
        assert_eq!(doc_type, DocumentType::Mixed, "Expected Mixed type for variable spacing");
    }

    #[test]
    fn test_document_type_insufficient_samples() {
        // Insufficient samples should default to Mixed
        let gaps = vec![0.5, 0.6];
        let doc_type = DocumentType::detect(&gaps);
        assert_eq!(doc_type, DocumentType::Mixed, "Expected Mixed type for insufficient samples");
    }

    #[test]
    fn test_document_type_empty() {
        let gaps = vec![];
        let doc_type = DocumentType::detect(&gaps);
        assert_eq!(doc_type, DocumentType::Mixed, "Expected Mixed type for empty gaps");
    }

    #[test]
    fn test_document_type_threshold_multiplier() {
        assert_eq!(DocumentType::Academic.threshold_multiplier(), 1.6);
        assert_eq!(DocumentType::Policy.threshold_multiplier(), 1.2);
        assert_eq!(DocumentType::Mixed.threshold_multiplier(), 1.5);
    }

    #[test]
    fn test_document_type_min_threshold_pt() {
        assert_eq!(DocumentType::Academic.min_threshold_pt(), 0.2);
        assert_eq!(DocumentType::Policy.min_threshold_pt(), 0.05);
        assert_eq!(DocumentType::Mixed.min_threshold_pt(), 0.1);
    }

    #[test]
    fn test_document_type_adaptive_config() {
        let academic_config = DocumentType::Academic.get_adaptive_config();
        assert_eq!(academic_config.median_multiplier, 1.6);
        assert_eq!(academic_config.min_threshold_pt, 0.2);

        let policy_config = DocumentType::Policy.get_adaptive_config();
        assert_eq!(policy_config.median_multiplier, 1.2);
        assert_eq!(policy_config.min_threshold_pt, 0.05);

        let mixed_config = DocumentType::Mixed.get_adaptive_config();
        assert_eq!(mixed_config.median_multiplier, 1.5);
        assert_eq!(mixed_config.min_threshold_pt, 0.05);
    }

    #[test]
    fn test_document_type_name() {
        assert_eq!(DocumentType::Academic.name(), "Academic");
        assert_eq!(DocumentType::Policy.name(), "Policy");
        assert_eq!(DocumentType::Mixed.name(), "Mixed");
    }

    #[test]
    fn test_document_type_very_tight_policy() {
        // Extremely tight spacing (< 0.3pt median)
        let gaps = vec![0.1, 0.15, 0.2, 0.15, 0.1, 0.18, 0.12, 0.16, 0.14, 0.13];
        let doc_type = DocumentType::detect(&gaps);
        assert_eq!(doc_type, DocumentType::Policy, "Expected Policy type for very tight spacing");
    }

    #[test]
    fn test_document_type_edge_case_boundary() {
        // Test edge case: median exactly at 0.8pt boundary
        let gaps = vec![0.75, 0.8, 0.79, 0.81, 0.77, 0.83, 0.78, 0.82, 0.76, 0.8];
        let doc_type = DocumentType::detect(&gaps);
        // Should still classify as Policy since variance is low
        assert_eq!(
            doc_type,
            DocumentType::Policy,
            "Expected Policy type for median near 0.8pt threshold"
        );
    }

    #[test]
    fn test_policy_profile_detection() {
        use crate::geometry::Rect;

        // Create spans with very tight gaps (< 0.5pt) - typical for policy documents
        // Gap = right edge of span N - left edge of span N+1
        // Span 0: left=0, width=8, right=8
        // Span 1: left=8.1 (tight gap of 0.1pt), width=8, right=16.1
        let spans: Vec<TextSpan> = (0..15)
            .map(|i| TextSpan {
                text: format!("word{}", i),
                bbox: Rect::new((i as f32) * 8.1, 0.0, 8.0, 12.0), // 0.1pt gaps
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: i,
                split_boundary_before: false,
            })
            .collect();

        let profile = DocumentProfile::detect(&spans, None);
        assert_eq!(profile, DocumentProfile::Policy, "Expected Policy profile for tight spacing");
    }

    #[test]
    fn test_academic_profile_detection() {
        use crate::geometry::Rect;

        // Create spans with high variance in gaps (columns):
        // First column: tight spacing (1-2pt)
        // Then gap (20pt - column boundary)
        // Second column: tight spacing (1-2pt)
        let mut spans = Vec::new();

        // First column: 10 words with 1-2pt gaps
        for i in 0..10 {
            spans.push(TextSpan {
                text: format!("word{}", i),
                bbox: Rect::new((i as f32) * 10.0, 100.0, 8.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: i,
                split_boundary_before: false,
            });
        }

        // Large gap (column boundary) - 20pt
        // Second column: 10 words with 1-2pt gaps
        for i in 10..20 {
            spans.push(TextSpan {
                text: format!("word{}", i),
                bbox: Rect::new(150.0 + ((i - 10) as f32) * 10.0, 100.0, 8.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: i,
                split_boundary_before: false,
            });
        }

        let profile = DocumentProfile::detect(&spans, None);
        assert_eq!(
            profile,
            DocumentProfile::Academic,
            "Expected Academic profile for high gap variance"
        );
    }

    #[test]
    fn test_default_profile_fallback() {
        use crate::geometry::Rect;

        // Create spans with moderate, consistent spacing
        let spans: Vec<TextSpan> = (0..15)
            .map(|i| TextSpan {
                text: format!("word{}", i),
                bbox: Rect::new((i as f32) * 15.0, 0.0, 8.0, 12.0), // Consistent 7pt gap
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: i,
                split_boundary_before: false,
            })
            .collect();

        let profile = DocumentProfile::detect(&spans, None);
        assert_eq!(
            profile,
            DocumentProfile::Default,
            "Expected Default profile for balanced spacing"
        );
    }

    #[test]
    fn test_profile_config_values() {
        // Academic profile
        let academic_config = DocumentProfile::Academic.get_config();
        assert_eq!(academic_config.median_multiplier, 1.6);
        assert_eq!(academic_config.min_threshold_pt, 0.1);

        // Policy profile
        let policy_config = DocumentProfile::Policy.get_config();
        assert_eq!(policy_config.median_multiplier, 1.2);
        assert_eq!(policy_config.min_threshold_pt, 0.05);

        // Default profile
        let default_config = DocumentProfile::Default.get_config();
        assert_eq!(default_config.median_multiplier, 1.5);
        assert_eq!(default_config.min_threshold_pt, 0.05);
    }

    #[test]
    fn test_document_profile_name() {
        assert_eq!(DocumentProfile::Academic.name(), "Academic");
        assert_eq!(DocumentProfile::Policy.name(), "Policy");
        assert_eq!(DocumentProfile::Default.name(), "Default");
    }

    #[test]
    fn test_profile_detect_with_existing_stats() {
        use crate::geometry::Rect;

        let spans: Vec<TextSpan> = (0..5)
            .map(|i| TextSpan {
                text: format!("w{}", i),
                bbox: Rect::new((i as f32) * 15.0, 0.0, 8.0, 12.0),
                font_name: "Arial".to_string(),
                font_size: 12.0,
                font_weight: crate::layout::FontWeight::Normal,
                color: crate::layout::Color::new(0.0, 0.0, 0.0),
                mcid: None,
                sequence: i,
                split_boundary_before: false,
            })
            .collect();

        // Create stats with tight median (policy profile)
        let stats = GapStatistics {
            gaps: vec![0.2, 0.25, 0.3, 0.25, 0.2],
            count: 5,
            min: 0.2,
            max: 0.3,
            mean: 0.24,
            median: 0.25,
            std_dev: 0.04,
            p25: 0.2,
            p75: 0.3,
            p10: 0.2,
            p90: 0.3,
        };

        let profile = DocumentProfile::detect(&spans, Some(&stats));
        assert_eq!(profile, DocumentProfile::Policy, "Expected Policy profile when median < 0.5");
    }
}
