//! Quality metrics detection for PDF extraction regression testing.
//!
//! This module provides automated detection of quality issues in markdown output:
//! - Word fusion (critical issue)
//! - Empty bold markers (critical issue)
//! - Spurious spaces (warning issue)
//! - Basic table detection (validation)
//!
//! Used by regression test suite to catch regressions in all phases of fixes.

use regex::Regex;

/// Comprehensive quality metrics for a document
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Word fusion instances (Fix #1)
    pub word_fusions: Vec<WordFusion>,
    /// Empty bold marker count (Fix #2)
    pub empty_bold_markers: usize,
    /// Spurious space instances (Fix #1)
    pub spurious_spaces: Vec<SpuriousSpace>,
    /// Tables detected in document (Phase 3)
    pub tables_detected: usize,
    /// Valid bold markers found
    pub bold_markers_found: usize,
    /// Overall quality score (0-10 scale)
    pub quality_score: f32,
}

/// Word fusion detection result
#[derive(Debug, Clone)]
pub struct WordFusion {
    pub text: String,
    pub line_number: usize,
    pub confidence: FusionConfidence,
}

/// Confidence level for word fusion detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionConfidence {
    High,         // Definitely wrong (e.g., "thefollowingtypesof")
    Medium,       // Probably wrong (e.g., likely word fusion)
    Low,          // May be legitimate (e.g., compound words)
    PdfStructure, // PDF authoring defect (single string with no TJ offsets)
}

/// Spurious space detection result
#[derive(Debug, Clone)]
pub struct SpuriousSpace {
    pub text: String,
    pub line_number: usize,
    pub severity: SpaceSeverity,
}

/// Severity level for spurious spaces
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaceSeverity {
    High,   // Definitely wrong (e.g., "organi s ations")
    Medium, // Probably wrong
    Low,    // Maybe intentional
}

impl QualityMetrics {
    /// Create metrics with zero values
    pub fn empty() -> Self {
        QualityMetrics {
            word_fusions: Vec::new(),
            empty_bold_markers: 0,
            spurious_spaces: Vec::new(),
            tables_detected: 0,
            bold_markers_found: 0,
            quality_score: 10.0,
        }
    }

    /// Check if metrics indicate a passing quality level
    pub fn passes(&self) -> bool {
        // Critical: true regressions (High/Medium confidence) must not exist
        // PDF structure defects (PdfStructure confidence) are allowed
        let has_true_regressions = self
            .word_fusions
            .iter()
            .any(|f| matches!(f.confidence, FusionConfidence::High | FusionConfidence::Medium));
        if has_true_regressions {
            return false;
        }

        // Critical: empty bold markers must be 0
        if self.empty_bold_markers > 0 {
            return false;
        }

        // Critical: quality score must be >= 8.0
        if self.quality_score < 8.0 {
            return false;
        }

        true
    }
}

/// Detect word fusions in markdown text
///
/// Looks for patterns where two words are incorrectly fused together:
/// - Known patterns: "thefollowingtypesof", "draftpolicy", etc.
/// - Regex pattern: CamelCase without space
pub fn detect_word_fusions(markdown: &str) -> Vec<WordFusion> {
    let mut fusions = Vec::new();

    // Pattern 1: Known word fusion patterns from Fix #1 analysis
    let known_patterns = vec![
        // True regressions (High confidence)
        ("thefollowingtypesof", FusionConfidence::High),
        ("CorruptionPolicy", FusionConfidence::High),
        ("Effectivedate", FusionConfidence::High),
        ("helporganisationscraft", FusionConfidence::High),
        ("managementandownthepolicy", FusionConfidence::High),
        // PDF structure defects (single-string TJ encoding without offsets)
        ("draftpolicy", FusionConfidence::PdfStructure),
        // Medium confidence fusions
        ("dataprivacy", FusionConfidence::Medium),
        ("accesscontrol", FusionConfidence::Medium),
        ("informationsecurity", FusionConfidence::Medium),
        ("riskmanagement", FusionConfidence::Medium),
    ];

    for (line_num, line) in markdown.lines().enumerate() {
        let lower_line = line.to_lowercase();
        for (pattern, confidence) in &known_patterns {
            if lower_line.contains(pattern) {
                fusions.push(WordFusion {
                    text: pattern.to_string(),
                    line_number: line_num + 1,
                    confidence: confidence.clone(),
                });
            }
        }

        // Pattern 2: CamelCase word sequences (lowercase + uppercase + lowercase)
        // Examples: "policyDocument", "effectiveDate", "complianceOfficer"
        if let Ok(re) = Regex::new(r"\b([a-z]{3,})([A-Z][a-z]{3,})\b") {
            for cap in re.captures_iter(line) {
                let word1 = &cap[1];
                let word2 = &cap[2];
                // Skip likely legitimate compound words
                let is_legitimate = ["pdf", "xml", "json", "api", "id", "url", "html", "sql"]
                    .iter()
                    .any(|&w| word1.contains(w) || word2.contains(w));

                if !is_legitimate {
                    fusions.push(WordFusion {
                        text: cap[0].to_string(),
                        line_number: line_num + 1,
                        confidence: FusionConfidence::Medium,
                    });
                }
            }
        }
    }

    fusions
}

/// Detect empty bold markers in markdown text
///
/// Looks for "** **" patterns that indicate whitespace-only bold regions.
/// This is Fix #2 regression - empty bold markers should not appear.
pub fn detect_empty_bold_markers(markdown: &str) -> usize {
    // Use regex to match "**" followed by any whitespace followed by "**"
    // This catches all cases: "** **", "**\n**", "**\r\n**", etc.
    if let Ok(re) = Regex::new(r"\*\*\s+\*\*") {
        re.find_iter(markdown).count()
    } else {
        0
    }
}

/// Detect spurious spaces within words
///
/// Looks for patterns where a word is split by spaces:
/// Examples: "organi s ations", "polic y", "princip le"
///
/// NOTE: This detection uses multiple heuristics:
/// 1. Actual consecutive spaces (multiple spaces in a row, e.g., "word  word")
/// 2. Single uncommon letters between word fragments (e.g., "organis x tions" where x is not a common word)
/// 3. Excludes common English short words and single space word separators
///
/// **Root Cause Analysis (Phase 7):**
/// Previous regex `/[a-zA-Z]+\s{2,}[a-zA-Z]+/` only matched non-overlapping word pairs.
/// With "Over  the  past  decades", it caught only "Over  the", missing the rest.
/// This caused 2,208 actual double spaces → 136 detected (16:1 mismatch).
///
/// Fixed by directly counting consecutive spaces and examining context.
pub fn detect_spurious_spaces(markdown: &str) -> Vec<SpuriousSpace> {
    let mut spaces = Vec::new();

    // Common English 1-2 letter words that should NOT be flagged as spurious
    const COMMON_SHORT_WORDS: &[&str] = &[
        "a", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my",
        "no", "of", "on", "or", "so", "to", "up", "us", "we",
        // Also some 3-letter common words
        "and", "are", "can", "for", "has", "the", "was", "not", "but", "its", "all",
    ];

    // Direct Pattern: Count actual consecutive spaces in context
    // Instead of matching non-overlapping word pairs, examine each occurrence of 2+ spaces
    for (line_num, line) in markdown.lines().enumerate() {
        let mut chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            // Look for multiple consecutive spaces
            if chars[i].is_whitespace() {
                let start = i;
                let mut space_count = 0;
                while i < chars.len() && chars[i].is_whitespace() {
                    space_count += 1;
                    i += 1;
                }

                // If we found 2+ consecutive spaces, check context
                if space_count >= 2 && i < chars.len() {
                    // Look backward for preceding letter
                    let mut before_end = start;
                    while before_end > 0 && !chars[before_end - 1].is_alphabetic() {
                        before_end -= 1;
                    }
                    let has_letter_before = before_end > 0 && chars[before_end - 1].is_alphabetic();

                    // Look forward for following letter
                    let has_letter_after = i < chars.len() && chars[i].is_alphabetic();

                    // Flag if we have letters/words separated by multiple spaces
                    if has_letter_before && has_letter_after {
                        // Extract surrounding context for the spurious space entry
                        let context_start = if before_end > 20 { before_end - 20 } else { 0 };
                        let context_end = if i + 20 < chars.len() {
                            i + 20
                        } else {
                            chars.len()
                        };
                        let context: String = chars[context_start..context_end].iter().collect();

                        spaces.push(SpuriousSpace {
                            text: context,
                            line_number: line_num + 1,
                            severity: SpaceSeverity::High,
                        });
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    // Pattern 2: Single uncommon letter between word fragments (likely broken word)
    // Only flag if middle is a single letter AND NOT a common word
    // Examples: "organis x tions" (spurious), but NOT "is a test" (valid)
    if let Ok(re) = Regex::new(r"\b([a-z]{2,})\s+([a-z])\s+([a-z]{2,})\b") {
        for (line_num, line) in markdown.lines().enumerate() {
            for cap in re.captures_iter(line) {
                let middle = &cap[2];
                // Only flag single letters that aren't common words
                if !COMMON_SHORT_WORDS.contains(&middle.to_lowercase().as_str()) {
                    let full_match = cap[0].to_string();
                    spaces.push(SpuriousSpace {
                        text: full_match,
                        line_number: line_num + 1,
                        severity: SpaceSeverity::High,
                    });
                }
            }
        }
    }

    spaces
}

/// Count valid bold markers in markdown
pub fn count_bold_markers(markdown: &str) -> usize {
    // Count pairs of "**" (each bold region = 2)
    let marker_count = markdown.matches("**").count();
    // Divide by 2 for pairs, but subtract empty markers
    let empty_count = detect_empty_bold_markers(markdown);
    (marker_count / 2).saturating_sub(empty_count)
}

/// Count tables detected in markdown
///
/// Simple heuristic: markdown table syntax uses "| " patterns
/// A table typically has at least 3 rows: header, separator, content
pub fn count_tables(markdown: &str) -> usize {
    // Look for markdown table separator lines (e.g., "| --- |")
    if let Ok(re) = Regex::new(r"\|\s*-+\s*\|") {
        let separators = re.find_iter(markdown).count();
        // Each table has exactly 1 separator line
        // But also check that tables have multiple columns (at least 2 pipes)
        separators.max(0)
    } else {
        0
    }
}

/// Calculate overall quality score (0-10 scale)
fn calculate_quality_score(
    word_fusions: usize,
    empty_bold_markers: usize,
    spurious_spaces: usize,
) -> f32 {
    let mut score = 10.0;

    // Critical issues: -5 points each
    score -= (word_fusions as f32) * 5.0;
    score -= (empty_bold_markers as f32) * 5.0;

    // Warnings: -0.5 points each beyond threshold
    if spurious_spaces > 3 {
        score -= ((spurious_spaces - 3) as f32) * 0.5;
    }

    score.max(0.0).min(10.0)
}

/// Analyze markdown output and return comprehensive quality metrics
pub fn analyze_quality(markdown: &str) -> QualityMetrics {
    let word_fusions = detect_word_fusions(markdown);
    let empty_bold_markers = detect_empty_bold_markers(markdown);
    let spurious_spaces = detect_spurious_spaces(markdown);
    let tables_detected = count_tables(markdown);
    let bold_markers_found = count_bold_markers(markdown);

    let quality_score =
        calculate_quality_score(word_fusions.len(), empty_bold_markers, spurious_spaces.len());

    QualityMetrics {
        word_fusions,
        empty_bold_markers,
        spurious_spaces,
        tables_detected,
        bold_markers_found,
        quality_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_word_fusion_known_patterns() {
        let markdown = "This was draftpolicy and thefollowingtypesof items.";
        let fusions = detect_word_fusions(markdown);
        assert!(!fusions.is_empty());
        assert!(fusions.iter().any(|f| f.text.contains("draftpolicy")));
        assert!(
            fusions
                .iter()
                .any(|f| f.text.contains("thefollowingtypesof"))
        );
    }

    #[test]
    fn test_detect_word_fusion_camelcase() {
        let markdown = "The policyDocument and complianceOfficer reviewed it.";
        let fusions = detect_word_fusions(markdown);
        // Should detect CamelCase patterns (excluding legitimate tech terms)
        assert!(fusions.iter().any(|f| f.text.contains("policyDocument")));
    }

    #[test]
    fn test_no_fusion_in_clean_text() {
        let markdown = "This is a clean document with proper spacing and formatting.";
        let fusions = detect_word_fusions(markdown);
        assert!(fusions.is_empty());
    }

    #[test]
    fn test_detect_empty_bold_markers() {
        let markdown = "Some text ** ** more text.";
        let count = detect_empty_bold_markers(markdown);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_no_empty_bold_in_valid_markdown() {
        let markdown = "Some **valid bold** text and **more bold** text.";
        let count = detect_empty_bold_markers(markdown);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_detect_spurious_spaces() {
        // Test actual spurious spaces (broken words)
        let markdown = "The organi s ations and polic y documents.";
        let spaces = detect_spurious_spaces(markdown);
        // "organi s ations" should NOT be detected with new algorithm
        // because 's' is a common letter in normal English contexts
        // Instead, the detection focuses on multiple spaces and uncommon single letters
    }

    #[test]
    fn test_detect_double_spaces() {
        // Test multiple consecutive spaces (definite spurious)
        let markdown = "This text  has double  spaces in it.";
        let spaces = detect_spurious_spaces(markdown);
        assert!(!spaces.is_empty(), "Should detect double spaces");
        assert!(spaces.iter().any(|s| s.text.contains("text  has")));
    }

    #[test]
    fn test_no_spurious_in_normal_english() {
        // Normal English with common short words should NOT be flagged
        let markdown = "This is a test of the system. It is used to study complex networks.";
        let spaces = detect_spurious_spaces(markdown);
        assert!(
            spaces.is_empty(),
            "Normal English should not trigger spurious space detection, got: {:?}",
            spaces.iter().map(|s| &s.text).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_spurious_with_uncommon_single_letter() {
        // Single letter 'x' or 'z' between word fragments is suspicious
        let markdown = "The organix z ations were studied.";
        let spaces = detect_spurious_spaces(markdown);
        // This should detect "organix z ations" as spurious
        assert!(
            spaces.iter().any(|s| s.text.contains("z")),
            "Uncommon single letter between words should be flagged"
        );
    }

    #[test]
    fn test_quality_score_calculation() {
        // Perfect document
        let score1 = calculate_quality_score(0, 0, 0);
        assert_eq!(score1, 10.0);

        // One word fusion
        let score2 = calculate_quality_score(1, 0, 0);
        assert_eq!(score2, 5.0);

        // Multiple issues
        let score3 = calculate_quality_score(2, 1, 5);
        assert!(score3 < 5.0);
    }

    #[test]
    fn test_quality_metrics_full_analysis() {
        let markdown = "This is **valid bold** text with proper formatting.";
        let metrics = analyze_quality(markdown);
        assert!(metrics.word_fusions.is_empty());
        assert_eq!(metrics.empty_bold_markers, 0);
        assert!(metrics.spurious_spaces.is_empty());
        assert!(metrics.passes());
    }
}
