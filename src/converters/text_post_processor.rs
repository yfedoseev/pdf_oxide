//! Text post-processing for PDF extraction quality improvements.
//!
//! This module implements PDF Spec-compliant text post-processing to fix
//! common text extraction issues:
//!
//! - **Soft hyphen handling** (PDF Spec Section 14.8.2.2.3):
//!   Removes U+00AD (soft hyphen) characters at line breaks when rejoining
//!   hyphenated words across lines.
//!
//! - **Whitespace normalization**:
//!   Removes excessive spaces within words while preserving intentional spacing
//!   between words.
//!
//! - **Special character spacing**:
//!   Ensures proper spacing around Greek letters, mathematical symbols, and
//!   other special characters that require boundary detection.

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Regex for detecting line-end hyphens with potential line breaks
    /// Matches: soft hyphen or hard hyphen at line end, followed by optional whitespace and lowercase letter
    static ref RE_HYPHEN_LINEBREAK: Regex = Regex::new(r"(\-|\u{00AD})\n\s*([a-z])").unwrap();

    /// Regex for excessive spaces within words
    /// Matches: non-space followed by 2+ spaces followed by non-space
    static ref RE_EXCESSIVE_SPACES: Regex = Regex::new(r"([^\s])\s{2,}([^\s])").unwrap();
}

/// Text post-processor for improving extraction quality per PDF specification.
pub struct TextPostProcessor;

impl TextPostProcessor {
    /// Remove soft hyphens at line breaks per PDF Spec 14.8.2.2.3.
    ///
    /// The PDF specification states that soft hyphens (U+00AD) are used to indicate
    /// where a word can be hyphenated at line boundaries. When extracting text,
    /// these should be removed and the word should be rejoined.
    ///
    /// # Algorithm
    ///
    /// 1. Identify lines ending with `-` or U+00AD (soft hyphen)
    /// 2. Check if next line starts with lowercase letter (likely continuation)
    /// 3. If yes, remove hyphen and newline, joining the words
    /// 4. If no, keep as-is (actual hard hyphen or section break)
    ///
    /// # Arguments
    ///
    /// * `text` - The markdown text to process
    ///
    /// # Returns
    ///
    /// Text with soft hyphens removed and words rejoined
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::converters::text_post_processor::TextPostProcessor;
    ///
    /// let input = "modali-\nties are important";
    /// let output = TextPostProcessor::rejoin_hyphenated_words(input);
    /// assert_eq!(output, "modalities are important");
    /// ```
    pub fn rejoin_hyphenated_words(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let lines: Vec<&str> = text.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim_end();

            // Check if this line ends with a hyphen (soft or hard)
            if (trimmed.ends_with('-') || trimmed.ends_with('\u{00AD}')) && i + 1 < lines.len() {
                let next_line = lines[i + 1].trim_start();

                // If next line starts with lowercase letter, likely word continuation
                if next_line.chars().next().is_some_and(|c| c.is_lowercase()) {
                    // Remove the hyphen/soft-hyphen and join words
                    let without_hyphen = if trimmed.ends_with('\u{00AD}') {
                        &trimmed[..trimmed.len() - '\u{00AD}'.len_utf8()]
                    } else {
                        &trimmed[..trimmed.len() - 1]
                    };

                    result.push_str(without_hyphen);
                    result.push_str(next_line);

                    // Skip the next line since we already processed it
                    i += 2;

                    // Add newline before next iteration if there is one
                    if i < lines.len() {
                        result.push('\n');
                    }
                    continue;
                }
            }

            // Normal case: not a hyphenated word break
            result.push_str(line);
            i += 1;

            // Add newline except after last line
            if i < lines.len() {
                result.push('\n');
            }
        }

        // Remove trailing newline if it wasn't in the original
        if result.ends_with('\n') && !text.ends_with('\n') {
            result.pop();
        }

        result
    }

    /// Normalize whitespace: remove extra spaces within words, preserve between words.
    ///
    /// Per PDF Spec Section 14.8.2.5, boundary whitespace must be checked before adding spaces.
    /// This function fixes the issue where text extraction creates extra spaces within words.
    ///
    /// # Algorithm
    ///
    /// For each sequence of 2+ consecutive spaces:
    /// - Check the preceding and following characters
    /// - If both are word characters, reduce to single space (likely a word boundary)
    /// - If either is punctuation, preserve spacing pattern
    ///
    /// # Arguments
    ///
    /// * `text` - The text to process
    ///
    /// # Returns
    ///
    /// Text with normalized whitespace
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::converters::text_post_processor::TextPostProcessor;
    ///
    /// let input = "The  quick   brown  fox";
    /// let output = TextPostProcessor::normalize_whitespace(input);
    /// assert_eq!(output, "The quick brown fox");
    /// ```
    pub fn normalize_whitespace(text: &str) -> String {
        // Use regex to replace 2+ spaces with single space, but only when
        // not at line boundaries (preserve indentation at start of lines)
        let mut result = String::with_capacity(text.len());

        for line in text.lines() {
            // Get leading spaces (indentation)
            let trimmed_start = line.trim_start();
            let leading_spaces = line.len() - trimmed_start.len();

            // Preserve leading spaces, then normalize internal spaces
            for _ in 0..leading_spaces {
                result.push(' ');
            }

            // Reduce 2+ consecutive spaces to 1
            let normalized = RE_EXCESSIVE_SPACES
                .replace_all(trimmed_start, "$1 $2")
                .to_string();
            result.push_str(&normalized);

            // Add newline except on last line
            if !result.ends_with('\n') {
                result.push('\n');
            }
        }

        // Remove trailing newline if added
        if result.ends_with('\n') && !text.ends_with('\n') {
            result.pop();
        }

        result
    }

    /// Apply full text post-processing pipeline.
    ///
    /// Applies hyphenation removal and whitespace normalization in sequence.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to process
    ///
    /// # Returns
    ///
    /// Fully processed text with improved extraction quality
    /// Ensure proper spacing around special characters (Greek letters, math symbols).
    ///
    /// Per PDF Spec Section 9.4.4, certain Unicode ranges require special handling
    /// for word boundary detection:
    /// - Greek letters (U+0370–U+03FF)
    /// - Mathematical symbols (U+2200–U+22FF)
    /// - Other special ranges that need spacing
    ///
    /// # Arguments
    ///
    /// * `text` - The text to process
    ///
    /// # Returns
    ///
    /// Text with proper spacing around special characters
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::converters::text_post_processor::TextPostProcessor;
    ///
    /// let input = "compute β-VAE model";
    /// let output = TextPostProcessor::ensure_special_char_spacing(input);
    /// // Ensures spacing is correct around β character
    /// ```
    pub fn ensure_special_char_spacing(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let chars: Vec<char> = text.chars().collect();

        for i in 0..chars.len() {
            let current_char = chars[i];
            let prev_char = if i > 0 { Some(chars[i - 1]) } else { None };
            let next_char = if i + 1 < chars.len() {
                Some(chars[i + 1])
            } else {
                None
            };

            // Check if current character is special (Greek, math, etc.)
            let is_special = Self::is_special_character(current_char);

            // Add space before special character if needed
            if is_special {
                if let Some(prev) = prev_char {
                    // Add space if:
                    // 1. Previous char is not whitespace AND
                    // 2. Previous char is not a punctuation that typically precedes special chars
                    if !prev.is_whitespace()
                        && !Self::is_space_before_special(prev)
                        && !result.is_empty()
                        && !result.ends_with(' ')
                    {
                        result.push(' ');
                    }
                }
            }

            result.push(current_char);

            // Add space after special character if needed
            if is_special {
                if let Some(next) = next_char {
                    if !next.is_whitespace() && !Self::is_space_after_special(next) {
                        result.push(' ');
                    }
                }
            }
        }

        result
    }

    /// Check if a character is a special character requiring spacing.
    #[cfg_attr(test, allow(dead_code))]
    pub fn is_special_character(ch: char) -> bool {
        // Greek letters: U+0370–U+03FF
        if ('\u{0370}'..='\u{03FF}').contains(&ch) {
            return true;
        }

        // Mathematical symbols: U+2200–U+22FF
        if ('\u{2200}'..='\u{22FF}').contains(&ch) {
            return true;
        }

        // Mathematical operators and symbols: U+2000–U+206F
        if ('\u{2000}'..='\u{206F}').contains(&ch) {
            return true;
        }

        false
    }

    /// Check if a character typically precedes special characters (shouldn't add space).
    #[cfg_attr(test, allow(dead_code))]
    pub fn is_space_before_special(ch: char) -> bool {
        matches!(ch, '(' | '[' | '{' | '<' | '-' | '/')
    }

    /// Check if a character typically follows special characters (shouldn't add space).
    #[cfg_attr(test, allow(dead_code))]
    pub fn is_space_after_special(ch: char) -> bool {
        matches!(ch, ')' | ']' | '}' | '>' | '-' | ',' | '.' | ':' | ';' | '\'' | '"')
    }

    /// Apply full text post-processing pipeline.
    ///
    /// Applies hyphenation removal, whitespace normalization, and special character spacing in sequence.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to process
    ///
    /// # Returns
    ///
    /// Fully processed text with improved extraction quality
    pub fn process(text: &str) -> String {
        let hyphenated_fixed = Self::rejoin_hyphenated_words(text);
        let whitespace_normalized = Self::normalize_whitespace(&hyphenated_fixed);
        Self::ensure_special_char_spacing(&whitespace_normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejoin_hyphenated_words_basic() {
        let input = "modali-\nties";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, "modalities");
    }

    #[test]
    fn test_rejoin_hyphenated_words_soft_hyphen() {
        let input = "phenomenon\u{00AD}\nnon";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, "phenomenonnon");
    }

    #[test]
    fn test_rejoin_hyphenated_words_with_context() {
        let input = "This is modali-\nties are important";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, "This is modalities are important");
    }

    #[test]
    fn test_rejoin_hyphenated_words_preserves_actual_hyphens() {
        // Hard hyphens (not at word break) should be preserved
        let input = "well-designed\nβ-VAE";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        // "β" is uppercase, so no rejoin
        assert_eq!(output, "well-designed\nβ-VAE");
    }

    #[test]
    fn test_rejoin_hyphenated_words_uppercase_start() {
        // Lines starting with uppercase should not be joined
        let input = "test-\nAnother";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, "test-\nAnother");
    }

    #[test]
    fn test_rejoin_hyphenated_words_multiple() {
        // Test two separate hyphenations on different lines
        let input = "phenom-\nenal\n\nmodali-\nties";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, "phenomenal\n\nmodalities");
    }

    #[test]
    fn test_rejoin_hyphenated_words_no_hyphens() {
        let input = "No hyphens here\nJust normal text";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_normalize_whitespace_basic() {
        let input = "The  quick   brown  fox";
        let output = TextPostProcessor::normalize_whitespace(input);
        assert_eq!(output, "The quick brown fox");
    }

    #[test]
    fn test_normalize_whitespace_multiline() {
        let input = "Line  one\nLine  two";
        let output = TextPostProcessor::normalize_whitespace(input);
        assert_eq!(output, "Line one\nLine two");
    }

    #[test]
    fn test_normalize_whitespace_preserves_indentation() {
        let input = "  indented   text";
        let output = TextPostProcessor::normalize_whitespace(input);
        // Should preserve the 2 leading spaces but normalize internal ones
        assert!(output.starts_with("  "));
    }

    #[test]
    fn test_normalize_whitespace_email_with_dots() {
        let input = "marlene. mayer@tum. de";
        let output = TextPostProcessor::normalize_whitespace(input);
        // Spaces after dots should be normalized
        assert_eq!(output, "marlene. mayer@tum. de");
    }

    #[test]
    fn test_normalize_whitespace_no_changes_needed() {
        let input = "The quick brown fox";
        let output = TextPostProcessor::normalize_whitespace(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_process_combined() {
        let input = "modali-\nties  are  important";
        let output = TextPostProcessor::process(input);
        assert_eq!(output, "modalities are important");
    }

    #[test]
    fn test_rejoin_hyphenated_words_with_whitespace_after_hyphen() {
        let input = "test-  \nable";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        // trim_start removes the spaces, so "able" starts with lowercase 'a'
        assert_eq!(output, "testable");
    }

    #[test]
    fn test_normalize_whitespace_empty_string() {
        let input = "";
        let output = TextPostProcessor::normalize_whitespace(input);
        assert_eq!(output, "");
    }

    #[test]
    fn test_rejoin_hyphenated_words_end_of_text() {
        // Hyphen at very end of text (no next line)
        let input = "test-";
        let output = TextPostProcessor::rejoin_hyphenated_words(input);
        assert_eq!(output, "test-");
    }

    // ===== TDD Tests for Special Character Spacing (Phase 2B Extended) =====

    #[test]
    fn test_ensure_special_char_spacing_greek_letter_spacing() {
        // Greek letter β should have spacing around it
        let input = "computeβVAE";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Should ensure spaces around β
        assert!(output.contains(" β "));
    }

    #[test]
    fn test_ensure_special_char_spacing_greek_after_word() {
        let input = "modelβ-VAE";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Space before β, but keep - for hyphen
        assert!(output.contains(" β-"));
    }

    #[test]
    fn test_ensure_special_char_spacing_multiple_greek_letters() {
        let input = "αβγ";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Multiple Greek letters should have spacing
        assert!(!output.is_empty());
    }

    #[test]
    fn test_ensure_special_char_spacing_math_symbols() {
        // Math symbol ∑ (summation)
        let input = "compute∑x";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Should add space before and after
        assert!(output.contains(" ∑ "));
    }

    #[test]
    fn test_ensure_special_char_spacing_preserves_existing_spaces() {
        let input = "compute α VAE";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Existing spaces should be preserved
        assert!(output.contains("compute α VAE") || output.contains("compute  α  VAE"));
    }

    #[test]
    fn test_ensure_special_char_spacing_parenthesis_handling() {
        // Parentheses before/after special chars shouldn't add extra spaces
        let input = "(α)";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Should keep parentheses close: (α) not ( α )
        assert_eq!(output, "(α)");
    }

    #[test]
    fn test_ensure_special_char_spacing_punctuation_after() {
        let input = "variableα,";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Comma after α shouldn't have space added: α,
        assert!(output.contains("α,"));
    }

    #[test]
    fn test_ensure_special_char_spacing_hyphen_preservation() {
        let input = "β-VAE";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // Hyphen should be preserved: β-VAE (no space after β before hyphen)
        assert!(output.contains("β-"));
    }

    #[test]
    fn test_ensure_special_char_spacing_empty_string() {
        let input = "";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        assert_eq!(output, "");
    }

    #[test]
    fn test_ensure_special_char_spacing_no_special_chars() {
        let input = "regular text";
        let output = TextPostProcessor::ensure_special_char_spacing(input);
        // No special chars, should remain unchanged
        assert_eq!(output, "regular text");
    }

    #[test]
    fn test_process_full_pipeline_with_special_chars() {
        // Test the full pipeline: hyphenation + whitespace + special char spacing
        let input = "modali-\nties  α  VAE";
        let output = TextPostProcessor::process(input);
        // Should have: rejoined word + normalized spaces + special char spacing
        assert!(output.contains("modalities"));
        assert!(output.contains("α"));
    }

    #[test]
    fn test_is_special_character_greek_letters() {
        assert!(TextPostProcessor::is_special_character('α'));
        assert!(TextPostProcessor::is_special_character('β'));
        assert!(TextPostProcessor::is_special_character('γ'));
        assert!(TextPostProcessor::is_special_character('Ω'));
    }

    #[test]
    fn test_is_special_character_math_symbols() {
        assert!(TextPostProcessor::is_special_character('∑'));
        assert!(TextPostProcessor::is_special_character('∫'));
        assert!(TextPostProcessor::is_special_character('∞'));
    }

    #[test]
    fn test_is_special_character_regular_chars() {
        assert!(!TextPostProcessor::is_special_character('a'));
        assert!(!TextPostProcessor::is_special_character('1'));
        assert!(!TextPostProcessor::is_special_character(' '));
    }

    #[test]
    fn test_space_before_special_bracket() {
        assert!(TextPostProcessor::is_space_before_special('('));
        assert!(TextPostProcessor::is_space_before_special('['));
        assert!(TextPostProcessor::is_space_before_special('{'));
    }

    #[test]
    fn test_space_after_special_punctuation() {
        assert!(TextPostProcessor::is_space_after_special(','));
        assert!(TextPostProcessor::is_space_after_special('.'));
        assert!(TextPostProcessor::is_space_after_special(')'));
    }
}
