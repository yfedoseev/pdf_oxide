//! Integration tests for hyphenation module in text extraction pipeline

use pdf_oxide::pipeline::config::TextPipelineConfig;
use pdf_oxide::text::hyphenation::HyphenationHandler;

#[test]
fn test_hyphenation_reconstruction_simple() {
    let handler = HyphenationHandler::new();

    // Simple case: word split across line
    // "Government" is in the common words list, so it will be properly joined
    let input = "The Govern-\nment of the United States";
    let output = handler.process_text(input);

    // The hyphen at line break should be removed, joining "Govern" and "ment"
    assert!(output.contains("Government"), "Word should be reconstructed as 'Government'");
}

#[test]
fn test_hyphenation_preserves_regular_hyphens() {
    let handler = HyphenationHandler::new();

    // Regular hyphen (not at line break) should be preserved
    let input = "well-known phrase in the text";
    let output = handler.process_text(input);

    // Should keep the hyphen
    assert!(output.contains("well-known"), "Regular hyphens should be preserved");
}

#[test]
fn test_hyphenation_with_config_enabled() {
    // Test that configuration properly enables hyphenation
    let config_enabled = TextPipelineConfig::default().with_hyphenation_reconstruction(true);
    assert!(config_enabled.enable_hyphenation_reconstruction);
}

#[test]
fn test_hyphenation_with_config_disabled() {
    // Test that configuration can disable hyphenation
    let config_disabled = TextPipelineConfig::default().with_hyphenation_reconstruction(false);
    assert!(!config_disabled.enable_hyphenation_reconstruction);
}

#[test]
fn test_hyphenation_default_enabled() {
    // Test that hyphenation is enabled by default
    let config = TextPipelineConfig::default();
    assert!(config.enable_hyphenation_reconstruction);
}

#[test]
fn test_hyphenation_multiple_continuations() {
    let handler = HyphenationHandler::new();

    // Multiple hyphenated words
    let input = "Govern-\nment issued a reorgan-\nization";
    let output = handler.process_text(input);

    // Should process the first hyphenation
    assert!(output.contains("Government"), "First continuation should be reconstructed");
}

#[test]
fn test_hyphenation_preserves_compound_words() {
    let handler = HyphenationHandler::new();

    // Compound words with hyphens should be preserved
    let input = "content-type is a technical term";
    let output = handler.process_text(input);

    // Should preserve the hyphenated compound
    assert!(output.contains("content-type"), "Compound words should preserve hyphens");
}

#[test]
fn test_hyphenation_edge_case_empty_string() {
    let handler = HyphenationHandler::new();

    // Empty string should be handled gracefully
    let input = "";
    let output = handler.process_text(input);

    assert_eq!(output, "", "Empty string should remain empty");
}

#[test]
fn test_hyphenation_edge_case_no_hyphen() {
    let handler = HyphenationHandler::new();

    // Text without hyphens should remain unchanged
    let input = "This is normal text\nwithout any hyphens\nat line ends.";
    let output = handler.process_text(input);

    assert_eq!(output, input, "Text without hyphens should remain unchanged");
}

#[test]
fn test_hyphenation_single_letter_word() {
    let handler = HyphenationHandler::new();

    // Single letters after hyphen shouldn't trigger joining
    let input = "word-\na";
    let output = handler.process_text(input);

    // Single letter after hyphen should not be joined (minimum continuation length = 2)
    assert!(!output.contains("worda"), "Single letter shouldn't be joined");
}

#[test]
fn test_hyphenation_preserves_formatting_characters() {
    let handler = HyphenationHandler::new();

    // Text with special formatting should be handled correctly
    let input = "**bold text** contain-\ning word";
    let output = handler.process_text(input);

    // Should preserve markdown formatting
    assert!(output.contains("**bold text**"), "Formatting should be preserved");
}

#[test]
fn test_hyphenation_with_newlines_only() {
    let handler = HyphenationHandler::new();

    // Multiple newlines
    let input = "text\n\nwith\n\ngaps";
    let output = handler.process_text(input);

    // Should preserve paragraph structure
    assert_eq!(output, input, "Paragraph structure should be preserved");
}

#[test]
fn test_hyphenation_compound_prefix_self() {
    let handler = HyphenationHandler::new();

    // "self-regulation" is a compound word
    let input = "self-regulation is important";
    let output = handler.process_text(input);

    assert!(output.contains("self-regulation"), "Compound prefix 'self' should be preserved");
}

#[test]
fn test_hyphenation_compound_prefix_non() {
    let handler = HyphenationHandler::new();

    // "non-linear" is a compound word
    let input = "non-linear systems";
    let output = handler.process_text(input);

    assert!(output.contains("non-linear"), "Compound prefix 'non' should be preserved");
}

#[test]
fn test_hyphenation_trailing_hyphen_only() {
    let handler = HyphenationHandler::new();

    // Edge case: trailing hyphen with nothing after
    let input = "word-";
    let output = handler.process_text(input);

    // Should handle gracefully
    assert!(!output.is_empty(), "Should handle trailing hyphen gracefully");
}

#[test]
fn test_hyphenation_builder_pattern() {
    // Test the builder pattern for hyphenation
    let handler = HyphenationHandler::new()
        .with_min_continuation_length(3)
        .with_preserve_compounds(false);

    let input = "text-\nab";
    let output = handler.process_text(input);

    // With min length 3, "ab" (2 chars) shouldn't be joined
    assert!(!output.contains("textab"), "Minimum length threshold should be respected");
}

#[test]
fn test_hyphenation_multiline_paragraph() {
    let handler = HyphenationHandler::new();

    // Multi-line paragraph with hyphenations
    let input = "The quick brown\nfox jumps over\nthe lazy dog which is\nan example sentence.";
    let output = handler.process_text(input);

    // Should preserve the text (no hyphens to process)
    assert_eq!(output, input, "Text without continuation hyphens unchanged");
}

#[test]
fn test_hyphenation_mixed_content() {
    let handler = HyphenationHandler::new();

    // Mix of hyphenated and normal text
    // "implementation" is in the common words list
    let input = "This is normal.\nThis requires careful implemen-\ntation of the solution.";
    let output = handler.process_text(input);

    // Should reconstruct "implementation"
    assert!(
        output.contains("implementation"),
        "Word should be reconstructed as 'implementation'"
    );
}
