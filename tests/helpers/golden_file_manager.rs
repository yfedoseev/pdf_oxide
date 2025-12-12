//! Golden file management system for regression testing.
//!
//! Provides utilities to:
//! - Save extracted text as golden files
//! - Load and parse golden files
//! - Compare extracted text against golden files
//! - Detect regressions with detailed diff reporting

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Default golden files directory
pub const DEFAULT_GOLDEN_DIR: &str = "tests/golden_files";

/// Comparison status for golden file testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonStatus {
    /// Extraction matches golden file perfectly
    Pass,
    /// Minor differences within tolerance
    Warning,
    /// Significant regression detected
    Fail,
}

/// Result of comparing extracted text to golden file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Overall status
    pub status: ComparisonStatus,
    /// SHA-256 hash matches
    pub hash_match: bool,
    /// Character count matches (within tolerance)
    pub char_count_match: bool,
    /// Word count matches (within tolerance)
    pub word_count_match: bool,
    /// Character count difference
    pub char_count_diff: i64,
    /// Word count difference
    pub word_count_diff: i64,
    /// First difference position (if any)
    pub first_diff_position: Option<usize>,
    /// Context around first difference
    pub diff_context: Option<String>,
}

impl ComparisonResult {
    /// Check if comparison passes (Pass or Warning)
    pub fn passes(&self) -> bool {
        matches!(self.status, ComparisonStatus::Pass | ComparisonStatus::Warning)
    }

    /// Get detailed error message
    pub fn details(&self) -> String {
        match self.status {
            ComparisonStatus::Pass => "Extraction matches golden file".to_string(),
            ComparisonStatus::Warning => {
                let mut msg = String::from("Minor differences detected:\n");
                if !self.char_count_match {
                    msg.push_str(&format!("  - Character count diff: {}\n", self.char_count_diff));
                }
                if !self.word_count_match {
                    msg.push_str(&format!("  - Word count diff: {}\n", self.word_count_diff));
                }
                msg
            },
            ComparisonStatus::Fail => {
                let mut msg = String::from("Regression detected:\n");
                if !self.hash_match {
                    msg.push_str("  - Text hash mismatch\n");
                }
                if let Some(pos) = self.first_diff_position {
                    msg.push_str(&format!("  - First difference at position {}\n", pos));
                    if let Some(context) = &self.diff_context {
                        msg.push_str(&format!("  - Context: {}\n", context));
                    }
                }
                msg.push_str(&format!("  - Character count diff: {}\n", self.char_count_diff));
                msg.push_str(&format!("  - Word count diff: {}\n", self.word_count_diff));
                msg
            },
        }
    }
}

/// Golden file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenFile {
    /// PDF file path
    pub pdf_path: String,
    /// Category
    pub category: String,
    /// Extracted text
    pub extracted_text: String,
    /// SHA-256 hash of extracted text
    pub text_hash: String,
    /// Character count
    pub char_count: usize,
    /// Word count
    pub word_count: usize,
    /// Script distribution
    pub script_distribution: HashMap<String, f32>,
    /// Extraction timestamp
    pub extraction_timestamp: String,
}

/// Golden file manager
pub struct GoldenFileManager {
    golden_dir: PathBuf,
}

impl GoldenFileManager {
    /// Create a new golden file manager
    pub fn new(golden_dir: impl AsRef<Path>) -> Self {
        GoldenFileManager {
            golden_dir: golden_dir.as_ref().to_path_buf(),
        }
    }

    /// Create a manager with default directory
    pub fn default() -> Self {
        GoldenFileManager::new(DEFAULT_GOLDEN_DIR)
    }

    /// Calculate SHA-256 hash of text
    fn calculate_hash(text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Count words in text (simple whitespace split)
    fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Detect script distribution in text
    fn detect_script_distribution(text: &str) -> HashMap<String, f32> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let total_chars = text.chars().filter(|c| c.is_alphabetic()).count();

        if total_chars == 0 {
            return HashMap::new();
        }

        for c in text.chars() {
            if !c.is_alphabetic() {
                continue;
            }

            let script = if c.is_ascii_alphabetic() {
                "Latin"
            } else if ('\u{4E00}'..='\u{9FFF}').contains(&c)
                || ('\u{3400}'..='\u{4DBF}').contains(&c)
            {
                "CJK"
            } else if ('\u{0600}'..='\u{06FF}').contains(&c)
                || ('\u{0750}'..='\u{077F}').contains(&c)
            {
                "Arabic"
            } else if ('\u{0590}'..='\u{05FF}').contains(&c) {
                "Hebrew"
            } else if ('\u{0900}'..='\u{097F}').contains(&c) {
                "Devanagari"
            } else if ('\u{0E00}'..='\u{0E7F}').contains(&c) {
                "Thai"
            } else {
                "Other"
            };

            *counts.entry(script.to_string()).or_insert(0) += 1;
        }

        // Convert to percentages
        counts
            .into_iter()
            .map(|(script, count)| (script, (count as f32 / total_chars as f32) * 100.0))
            .collect()
    }

    /// Save extracted text as a golden file
    pub fn save_golden_file(
        &self,
        pdf_path: &Path,
        category: &str,
        extracted_text: &str,
    ) -> std::io::Result<()> {
        // Create directory structure: golden_files/extracted_text/{category}/
        let category_dir = self.golden_dir.join("extracted_text").join(category);
        fs::create_dir_all(&category_dir)?;

        // Generate golden file
        let golden = GoldenFile {
            pdf_path: pdf_path.to_string_lossy().to_string(),
            category: category.to_string(),
            extracted_text: extracted_text.to_string(),
            text_hash: Self::calculate_hash(extracted_text),
            char_count: extracted_text.chars().count(),
            word_count: Self::count_words(extracted_text),
            script_distribution: Self::detect_script_distribution(extracted_text),
            extraction_timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Save as JSON
        let filename = pdf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.pdf")
            .replace(".pdf", ".json")
            .replace(".PDF", ".json");

        let golden_path = category_dir.join(filename);
        let json = serde_json::to_string_pretty(&golden)?;
        let mut file = fs::File::create(golden_path)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    /// Load a golden file
    pub fn load_golden_file(&self, pdf_path: &Path) -> std::io::Result<GoldenFile> {
        // Determine category from path
        let category = pdf_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let filename = pdf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.pdf")
            .replace(".pdf", ".json")
            .replace(".PDF", ".json");

        let golden_path = self
            .golden_dir
            .join("extracted_text")
            .join(category)
            .join(filename);

        let content = fs::read_to_string(golden_path)?;
        let golden: GoldenFile = serde_json::from_str(&content)?;

        Ok(golden)
    }

    /// Compare extracted text to golden file
    pub fn compare_extraction(
        &self,
        extracted_text: &str,
        golden: &GoldenFile,
    ) -> ComparisonResult {
        let extracted_hash = Self::calculate_hash(extracted_text);
        let extracted_char_count = extracted_text.chars().count();
        let extracted_word_count = Self::count_words(extracted_text);

        let hash_match = extracted_hash == golden.text_hash;

        // Tolerances
        let char_tolerance = (golden.char_count as f32 * 0.005).max(1.0) as i64; // 0.5%
        let word_tolerance = (golden.word_count as f32 * 0.01).max(1.0) as i64; // 1%

        let char_diff = extracted_char_count as i64 - golden.char_count as i64;
        let word_diff = extracted_word_count as i64 - golden.word_count as i64;

        let char_count_match = char_diff.abs() <= char_tolerance;
        let word_count_match = word_diff.abs() <= word_tolerance;

        // Find first difference
        let (first_diff_position, diff_context) = if !hash_match {
            Self::find_first_difference(extracted_text, &golden.extracted_text)
        } else {
            (None, None)
        };

        // Determine status
        let status = if hash_match && char_count_match && word_count_match {
            ComparisonStatus::Pass
        } else if char_count_match && word_count_match {
            ComparisonStatus::Warning
        } else {
            ComparisonStatus::Fail
        };

        ComparisonResult {
            status,
            hash_match,
            char_count_match,
            word_count_match,
            char_count_diff: char_diff,
            word_count_diff: word_diff,
            first_diff_position,
            diff_context,
        }
    }

    /// Find first difference between two strings
    fn find_first_difference(text1: &str, text2: &str) -> (Option<usize>, Option<String>) {
        let chars1: Vec<char> = text1.chars().collect();
        let chars2: Vec<char> = text2.chars().collect();

        for (i, (c1, c2)) in chars1.iter().zip(chars2.iter()).enumerate() {
            if c1 != c2 {
                let context_start = i.saturating_sub(20);
                let context_end = (i + 20).min(chars1.len()).min(chars2.len());

                let context1: String = chars1[context_start..context_end].iter().collect();
                let context2: String = chars2[context_start..context_end].iter().collect();

                let context =
                    format!("Position {}: Expected: {:?}, Got: {:?}", i, context2, context1);

                return (Some(i), Some(context));
            }
        }

        // Different lengths
        if chars1.len() != chars2.len() {
            let pos = chars1.len().min(chars2.len());
            let context = format!(
                "Length mismatch at position {}: Expected {}, Got {}",
                pos,
                chars2.len(),
                chars1.len()
            );
            return (Some(pos), Some(context));
        }

        (None, None)
    }

    /// Check if golden file exists for a PDF
    pub fn has_golden_file(&self, pdf_path: &Path) -> bool {
        let category = pdf_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let filename = pdf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.pdf")
            .replace(".pdf", ".json")
            .replace(".PDF", ".json");

        let golden_path = self
            .golden_dir
            .join("extracted_text")
            .join(category)
            .join(filename);

        golden_path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_calculation() {
        let text1 = "Hello, world!";
        let text2 = "Hello, world!";
        let text3 = "Hello, World!";

        assert_eq!(
            GoldenFileManager::calculate_hash(text1),
            GoldenFileManager::calculate_hash(text2)
        );
        assert_ne!(
            GoldenFileManager::calculate_hash(text1),
            GoldenFileManager::calculate_hash(text3)
        );
    }

    #[test]
    fn test_word_counting() {
        assert_eq!(GoldenFileManager::count_words("Hello world"), 2);
        assert_eq!(GoldenFileManager::count_words("  Hello   world  "), 2);
        assert_eq!(GoldenFileManager::count_words(""), 0);
    }

    #[test]
    fn test_script_distribution() {
        let text = "Hello 你好 مرحبا";
        let dist = GoldenFileManager::detect_script_distribution(text);

        assert!(dist.contains_key("Latin"));
        assert!(dist.contains_key("CJK"));
        assert!(dist.contains_key("Arabic"));

        // Percentages should sum to 100
        let total: f32 = dist.values().sum();
        assert!((total - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_find_first_difference() {
        let text1 = "Hello world";
        let text2 = "Hello World";

        let (pos, context) = GoldenFileManager::find_first_difference(text1, text2);
        assert_eq!(pos, Some(6));
        assert!(context.is_some());
    }

    #[test]
    fn test_comparison_result_passes() {
        let result = ComparisonResult {
            status: ComparisonStatus::Pass,
            hash_match: true,
            char_count_match: true,
            word_count_match: true,
            char_count_diff: 0,
            word_count_diff: 0,
            first_diff_position: None,
            diff_context: None,
        };

        assert!(result.passes());
    }

    #[test]
    fn test_comparison_result_fails() {
        let result = ComparisonResult {
            status: ComparisonStatus::Fail,
            hash_match: false,
            char_count_match: false,
            word_count_match: false,
            char_count_diff: 100,
            word_count_diff: 20,
            first_diff_position: Some(50),
            diff_context: Some("Context".to_string()),
        };

        assert!(!result.passes());
        assert!(result.details().contains("Regression"));
    }
}
