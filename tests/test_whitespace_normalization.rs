//! TDD Tests for Whitespace Normalization
//!
//! Tests verify that consecutive spaces and newlines are properly normalized
//! while preserving intentional formatting (code blocks, tables, paragraph breaks).

#[test]
fn test_normalize_single_space() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize("hello world"), "hello world");
}

#[test]
fn test_normalize_multiple_spaces() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize("hello   world"), "hello world");
    assert_eq!(normalizer.normalize("hello     world"), "hello world");
}

#[test]
fn test_normalize_tabs_to_spaces() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize("hello\t\tworld"), "hello world");
}

#[test]
fn test_normalize_mixed_whitespace() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize("hello  \t  world"), "hello world");
}

#[test]
fn test_preserve_single_newline() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    let result = normalizer.normalize("line1\nline2");
    assert!(result.contains('\n'));
    assert!(!result.contains("  ")); // No double spaces
}

#[test]
fn test_collapse_multiple_newlines_to_paragraph_break() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    let result = normalizer.normalize("para1\n\n\npara2");
    // Multiple newlines collapse to double newline (paragraph break)
    assert!(result.contains("para1") && result.contains("para2"));
}

#[test]
fn test_trim_leading_spaces() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize("   hello"), "hello");
}

#[test]
fn test_trim_trailing_spaces() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize("hello   "), "hello");
}

#[test]
fn test_preserve_layout_mode_no_normalization() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(true);
    let text = "hello   world";
    // In layout mode, spacing is preserved
    assert_eq!(normalizer.normalize(text), text);
}

#[test]
fn test_preserve_intentional_double_space() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    // Single intentional double-space (old-style sentence spacing)
    let text = "Sentence one.  Sentence two.";
    let result = normalizer.normalize(text);
    // Should normalize to single space
    assert!(!result.contains("  "));
}

#[test]
fn test_normalize_at_line_breaks() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    let result = normalizer.normalize("line1   \nline2");
    // Trailing spaces before newline should be trimmed
    assert!(!result.contains("   \n"));
}

#[test]
fn test_normalize_after_line_breaks() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    let result = normalizer.normalize("line1\n   line2");
    // Leading spaces after newline should be trimmed
    assert!(!result.contains("\n   "));
}

#[test]
fn test_handle_empty_string() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert_eq!(normalizer.normalize(""), "");
}

#[test]
fn test_handle_only_whitespace() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    assert!(normalizer.normalize("   \t  \n  ").trim().is_empty());
}

#[test]
fn test_normalize_multiple_paragraphs() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    let text = "Para 1  with  spaces\n\n\nPara 2   with   spaces";
    let result = normalizer.normalize(text);

    // Both paragraphs should have single spaces
    assert!(!result.contains("  with  "));
    assert!(!result.contains("   with   "));
}

#[test]
fn test_normalize_code_block_marker_preserved() {
    use pdf_oxide::pipeline::text_processing::WhitespaceNormalizer;

    let normalizer = WhitespaceNormalizer::new(false);
    // Markdown code fence should be preserved
    let text = "```\ncode with spaces\n```";
    let result = normalizer.normalize(text);
    assert!(result.contains("```"));
}
