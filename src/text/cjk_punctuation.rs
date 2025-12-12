//! CJK Punctuation Detection for Word Boundary Analysis
//!
//! This module provides detection functions for CJK (Chinese, Japanese, Korean) punctuation
//! characters that affect word boundary determination. CJK punctuation includes both fullwidth
//! and halfwidth variants that have different boundary semantics than ASCII punctuation.
//!
//! # Punctuation Categories
//!
//! - **Sentence-ending**: Fullstop, question mark, exclamation - always create boundaries
//! - **Enumeration**: Ideographic comma, semicolon - create boundaries in most contexts
//! - **Brackets**: Parentheses, angle brackets, corner brackets - paired boundaries
//! - **Other**: Ellipsis, middle dot, wave dash - context-dependent
//!
//! # Unicode Ranges
//!
//! - CJK Symbols and Punctuation: U+3000-U+303F
//! - Halfwidth and Fullwidth Forms: U+FF00-U+FFEF

/// Check if a character code represents fullwidth CJK punctuation.
///
/// Fullwidth punctuation is typically used in CJK text and occupies the same
/// width as CJK ideographs (one em square). These characters create stronger
/// word boundaries than their ASCII equivalents.
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is fullwidth CJK punctuation
pub fn is_fullwidth_punctuation(code: u32) -> bool {
    is_sentence_ending_punctuation(code)
        || is_enumeration_punctuation(code)
        || is_bracket_punctuation(code)
        || is_other_cjk_punctuation(code)
}

/// Check if a character is sentence-ending punctuation.
///
/// These punctuation marks always create word boundaries and typically end
/// sentences or clauses in CJK text.
///
/// # Punctuation Characters
///
/// - U+3002: IDEOGRAPHIC FULL STOP (。)
/// - U+FF01: FULLWIDTH EXCLAMATION MARK (！)
/// - U+FF1F: FULLWIDTH QUESTION MARK (？)
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is sentence-ending punctuation
pub fn is_sentence_ending_punctuation(code: u32) -> bool {
    matches!(
        code,
        0x3002  // IDEOGRAPHIC FULL STOP (。)
        | 0xFF01 // FULLWIDTH EXCLAMATION MARK (！)
        | 0xFF1F // FULLWIDTH QUESTION MARK (？)
    )
}

/// Check if a character is enumeration punctuation.
///
/// These punctuation marks separate items in lists or clauses and create
/// boundaries when preceded by significant spacing signals (TJ offset or
/// geometric gap).
///
/// # Punctuation Characters
///
/// - U+3001: IDEOGRAPHIC COMMA (、)
/// - U+FF0C: FULLWIDTH COMMA (，)
/// - U+FF1B: FULLWIDTH SEMICOLON (；)
/// - U+FF1A: FULLWIDTH COLON (：)
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is enumeration punctuation
pub fn is_enumeration_punctuation(code: u32) -> bool {
    matches!(
        code,
        0x3001  // IDEOGRAPHIC COMMA (、)
        | 0xFF0C // FULLWIDTH COMMA (，)
        | 0xFF1B // FULLWIDTH SEMICOLON (；)
        | 0xFF1A // FULLWIDTH COLON (：)
    )
}

/// Check if a character is bracket/parenthesis punctuation.
///
/// These paired punctuation marks typically enclose content and create
/// boundaries at their opening and closing positions.
///
/// # Punctuation Characters
///
/// - U+3008-U+3011: Various angle and corner brackets
/// - U+3014-U+3015: Tortoise shell brackets
/// - U+FF08-U+FF09: Fullwidth parentheses
/// - U+FF3B-U+FF3D: Fullwidth square brackets
/// - U+FF5B-U+FF5D: Fullwidth curly brackets
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is bracket punctuation
pub fn is_bracket_punctuation(code: u32) -> bool {
    matches!(
        code,
        0x3008..=0x3011  // Angle and corner brackets
        | 0x3014..=0x3015 // Tortoise shell brackets
        | 0xFF08..=0xFF09 // Fullwidth parentheses (（）)
        | 0xFF3B..=0xFF3D // Fullwidth square brackets (［］)
        | 0xFF5B..=0xFF5D // Fullwidth curly brackets (｛｝)
    )
}

/// Check if a character is other CJK punctuation.
///
/// This includes miscellaneous CJK punctuation that may create boundaries
/// depending on context.
///
/// # Punctuation Characters
///
/// - U+3000: IDEOGRAPHIC SPACE
/// - U+3003: DITTO MARK
/// - U+30FB: KATAKANA MIDDLE DOT
/// - U+FF0E: FULLWIDTH FULL STOP (．)
/// - U+FF5E: FULLWIDTH TILDE (～)
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is other CJK punctuation
pub fn is_other_cjk_punctuation(code: u32) -> bool {
    matches!(
        code,
        0x3000  // IDEOGRAPHIC SPACE
        | 0x3003 // DITTO MARK
        | 0x30FB // KATAKANA MIDDLE DOT
        | 0xFF0E // FULLWIDTH FULL STOP (．)
        | 0xFF0D // FULLWIDTH HYPHEN-MINUS
        | 0xFF5E // FULLWIDTH TILDE (～)
    )
}

/// Get the boundary confidence score for CJK punctuation.
///
/// This function returns a confidence score (0.0-1.0) indicating how strongly
/// a punctuation character signals a word boundary. Higher scores mean stronger
/// boundary indicators.
///
/// # Confidence Levels
///
/// - **1.0**: Sentence-ending punctuation (。！？) - always creates boundary
/// - **0.9**: Enumeration punctuation (、，；：) - strong boundary signal
/// - **0.8**: Bracket punctuation - paired boundaries
/// - **0.7**: Other CJK punctuation - context-dependent
/// - **0.0**: Not CJK punctuation
///
/// # Arguments
///
/// * `code` - Unicode code point to evaluate
///
/// # Returns
///
/// Confidence score from 0.0 (no boundary) to 1.0 (definite boundary)
pub fn get_cjk_punctuation_boundary_score(code: u32) -> f32 {
    if is_sentence_ending_punctuation(code) {
        1.0 // Definite boundary
    } else if is_enumeration_punctuation(code) {
        0.9 // Strong boundary signal
    } else if is_bracket_punctuation(code) {
        0.8 // Paired boundary
    } else if is_other_cjk_punctuation(code) {
        0.7 // Context-dependent
    } else {
        0.0 // Not CJK punctuation
    }
}

/// Check if a character is opening bracket punctuation.
///
/// Opening brackets typically create a boundary before the enclosed content.
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is an opening bracket
pub fn is_opening_bracket(code: u32) -> bool {
    matches!(
        code,
        0x3008  // LEFT ANGLE BRACKET
        | 0x300A // LEFT DOUBLE ANGLE BRACKET
        | 0x300C // LEFT CORNER BRACKET
        | 0x300E // LEFT WHITE CORNER BRACKET
        | 0x3010 // LEFT BLACK LENTICULAR BRACKET
        | 0x3014 // LEFT TORTOISE SHELL BRACKET
        | 0xFF08 // FULLWIDTH LEFT PARENTHESIS
        | 0xFF3B // FULLWIDTH LEFT SQUARE BRACKET
        | 0xFF5B // FULLWIDTH LEFT CURLY BRACKET
    )
}

/// Check if a character is closing bracket punctuation.
///
/// Closing brackets typically create a boundary after the enclosed content.
///
/// # Arguments
///
/// * `code` - Unicode code point to check
///
/// # Returns
///
/// `true` if the character is a closing bracket
pub fn is_closing_bracket(code: u32) -> bool {
    matches!(
        code,
        0x3009  // RIGHT ANGLE BRACKET
        | 0x300B // RIGHT DOUBLE ANGLE BRACKET
        | 0x300D // RIGHT CORNER BRACKET
        | 0x300F // RIGHT WHITE CORNER BRACKET
        | 0x3011 // RIGHT BLACK LENTICULAR BRACKET
        | 0x3015 // RIGHT TORTOISE SHELL BRACKET
        | 0xFF09 // FULLWIDTH RIGHT PARENTHESIS
        | 0xFF3D // FULLWIDTH RIGHT SQUARE BRACKET
        | 0xFF5D // FULLWIDTH RIGHT CURLY BRACKET
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideographic_full_stop() {
        assert!(is_sentence_ending_punctuation(0x3002));
        assert!(is_fullwidth_punctuation(0x3002));
        assert_eq!(get_cjk_punctuation_boundary_score(0x3002), 1.0);
    }

    #[test]
    fn test_fullwidth_question_mark() {
        assert!(is_sentence_ending_punctuation(0xFF1F));
        assert!(is_fullwidth_punctuation(0xFF1F));
        assert_eq!(get_cjk_punctuation_boundary_score(0xFF1F), 1.0);
    }

    #[test]
    fn test_fullwidth_exclamation() {
        assert!(is_sentence_ending_punctuation(0xFF01));
        assert!(is_fullwidth_punctuation(0xFF01));
        assert_eq!(get_cjk_punctuation_boundary_score(0xFF01), 1.0);
    }

    #[test]
    fn test_ideographic_comma() {
        assert!(is_enumeration_punctuation(0x3001));
        assert!(is_fullwidth_punctuation(0x3001));
        assert_eq!(get_cjk_punctuation_boundary_score(0x3001), 0.9);
    }

    #[test]
    fn test_fullwidth_comma() {
        assert!(is_enumeration_punctuation(0xFF0C));
        assert!(is_fullwidth_punctuation(0xFF0C));
        assert_eq!(get_cjk_punctuation_boundary_score(0xFF0C), 0.9);
    }

    #[test]
    fn test_fullwidth_semicolon() {
        assert!(is_enumeration_punctuation(0xFF1B));
        assert!(is_fullwidth_punctuation(0xFF1B));
        assert_eq!(get_cjk_punctuation_boundary_score(0xFF1B), 0.9);
    }

    #[test]
    fn test_fullwidth_colon() {
        assert!(is_enumeration_punctuation(0xFF1A));
        assert!(is_fullwidth_punctuation(0xFF1A));
        assert_eq!(get_cjk_punctuation_boundary_score(0xFF1A), 0.9);
    }

    #[test]
    fn test_fullwidth_parentheses() {
        assert!(is_bracket_punctuation(0xFF08));
        assert!(is_opening_bracket(0xFF08));
        assert!(is_bracket_punctuation(0xFF09));
        assert!(is_closing_bracket(0xFF09));
        assert_eq!(get_cjk_punctuation_boundary_score(0xFF08), 0.8);
    }

    #[test]
    fn test_angle_brackets() {
        assert!(is_bracket_punctuation(0x3008));
        assert!(is_opening_bracket(0x3008));
        assert!(is_bracket_punctuation(0x3009));
        assert!(is_closing_bracket(0x3009));
    }

    #[test]
    fn test_ideographic_space() {
        assert!(is_other_cjk_punctuation(0x3000));
        assert!(is_fullwidth_punctuation(0x3000));
        assert_eq!(get_cjk_punctuation_boundary_score(0x3000), 0.7);
    }

    #[test]
    fn test_katakana_middle_dot() {
        assert!(is_other_cjk_punctuation(0x30FB));
        assert!(is_fullwidth_punctuation(0x30FB));
    }

    #[test]
    fn test_non_cjk_punctuation() {
        // ASCII period should not be CJK punctuation
        assert!(!is_fullwidth_punctuation(0x002E));
        assert_eq!(get_cjk_punctuation_boundary_score(0x002E), 0.0);

        // ASCII comma should not be CJK punctuation
        assert!(!is_fullwidth_punctuation(0x002C));
        assert_eq!(get_cjk_punctuation_boundary_score(0x002C), 0.0);
    }

    #[test]
    fn test_boundary_score_ordering() {
        // Sentence-ending > Enumeration > Bracket > Other
        assert!(
            get_cjk_punctuation_boundary_score(0x3002) > get_cjk_punctuation_boundary_score(0x3001)
        );
        assert!(
            get_cjk_punctuation_boundary_score(0x3001) > get_cjk_punctuation_boundary_score(0xFF08)
        );
        assert!(
            get_cjk_punctuation_boundary_score(0xFF08) > get_cjk_punctuation_boundary_score(0x30FB)
        );
    }
}
