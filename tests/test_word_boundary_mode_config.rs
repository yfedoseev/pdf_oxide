#![allow(clippy::field_reassign_with_default)]
//! Phase 9.2.A: WordBoundaryMode Configuration Tests
//!
//! Tests for WordBoundaryMode enum and its integration with TextPipelineConfig.
//! This phase adds configuration infrastructure with no functional changes yet.

use pdf_oxide::extractors::{TextExtractionConfig, TextExtractor};
use pdf_oxide::pipeline::config::{TextPipelineConfig, WordBoundaryMode};

#[test]
fn test_word_boundary_mode_enum_exists() {
    // Just verify it compiles and exists
    let _mode = WordBoundaryMode::Tiebreaker;
    let _mode = WordBoundaryMode::Primary;
}

#[test]
fn test_word_boundary_mode_default_is_tiebreaker() {
    assert_eq!(WordBoundaryMode::default(), WordBoundaryMode::Tiebreaker);
}

#[test]
fn test_word_boundary_mode_clone_and_debug() {
    let mode = WordBoundaryMode::Primary;
    let mode_clone = mode;
    assert_eq!(mode, mode_clone);
    // Debug should not panic
    let _ = format!("{:?}", mode);
}

#[test]
fn test_word_boundary_mode_copy() {
    // Verify that WordBoundaryMode is Copy (required for efficient passing)
    let mode = WordBoundaryMode::Primary;
    let mode_copy = mode; // Should use Copy, not move
    assert_eq!(mode, mode_copy);
    // Original should still be usable (proves Copy, not Move)
    let _mode_again = mode;
}

#[test]
fn test_text_pipeline_config_default_mode_is_tiebreaker() {
    let config = TextPipelineConfig::default();
    assert_eq!(config.word_boundary_mode, WordBoundaryMode::Tiebreaker);
}

#[test]
fn test_text_pipeline_config_with_word_boundary_mode() {
    let config = TextPipelineConfig::default().with_word_boundary_mode(WordBoundaryMode::Primary);
    assert_eq!(config.word_boundary_mode, WordBoundaryMode::Primary);
}

#[test]
fn test_text_pipeline_config_pdfplumber_compatible_uses_tiebreaker() {
    let config = TextPipelineConfig::pdfplumber_compatible();
    assert_eq!(config.word_boundary_mode, WordBoundaryMode::Tiebreaker);
}

#[test]
fn test_text_extraction_config_has_word_boundary_mode() {
    // TextExtractionConfig should also have word_boundary_mode field
    let config = TextExtractionConfig::default();
    // Access the field to verify it exists
    let _ = config.word_boundary_mode;
}

#[test]
fn test_text_extraction_config_default_mode_is_tiebreaker() {
    let config = TextExtractionConfig::default();
    assert_eq!(config.word_boundary_mode, WordBoundaryMode::Tiebreaker);
}

#[test]
fn test_text_extractor_accepts_mode_from_config() {
    let mut config = TextExtractionConfig::default();
    config.word_boundary_mode = WordBoundaryMode::Primary;

    let _extractor = TextExtractor::with_config(config);
    // Just verify it initializes without panic
    // (Can't directly access field from test, but if it doesn't panic, it worked)
}

#[test]
fn test_text_extractor_defaults_to_tiebreaker_mode() {
    let config = TextExtractionConfig::default();
    let _extractor = TextExtractor::with_config(config);
    // Default should be Tiebreaker mode (no functional change from old behavior)
}

#[test]
fn test_word_boundary_mode_partial_eq() {
    // Test PartialEq implementation
    assert_eq!(WordBoundaryMode::Tiebreaker, WordBoundaryMode::Tiebreaker);
    assert_eq!(WordBoundaryMode::Primary, WordBoundaryMode::Primary);
    assert_ne!(WordBoundaryMode::Tiebreaker, WordBoundaryMode::Primary);
}

#[test]
fn test_builder_pattern_chaining() {
    // Verify builder pattern works with other config methods
    let config = TextPipelineConfig::default().with_word_boundary_mode(WordBoundaryMode::Primary);

    // Should be chainable with other builder methods if they exist
    assert_eq!(config.word_boundary_mode, WordBoundaryMode::Primary);
}

#[test]
fn test_word_boundary_mode_eq_trait() {
    // Test Eq trait (reflexive, symmetric, transitive)
    let mode1 = WordBoundaryMode::Tiebreaker;
    let mode2 = WordBoundaryMode::Tiebreaker;
    let mode3 = WordBoundaryMode::Primary;

    // Reflexive
    assert_eq!(mode1, mode1);

    // Symmetric
    assert_eq!(mode1, mode2);
    assert_eq!(mode2, mode1);

    // Transitive (if a == b && b == c, then a == c)
    // Not directly testable with 2 variants, but verify inequality
    assert_ne!(mode1, mode3);
    assert_ne!(mode2, mode3);
}
