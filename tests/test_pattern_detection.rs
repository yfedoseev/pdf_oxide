#![allow(clippy::needless_range_loop)]
//! Test suite for Week 2 Day 7: Email/URL Pattern Preservation (2C)
//!
//! This test suite verifies that email addresses and URLs are detected and
//! protected from word boundary splitting during text extraction.
//!
//! The pattern detector marks email and URL characters with the
//! `protected_from_split` flag, which prevents WordBoundaryDetector
//! from creating boundaries within these patterns.

use pdf_oxide::extractors::pattern_detector::{PatternDetector, PatternPreservationConfig};
use pdf_oxide::text::{BoundaryContext, CharacterInfo, WordBoundaryDetector};

/// Helper: Create a CharacterInfo for testing
fn create_char_info(code: u32) -> CharacterInfo {
    CharacterInfo {
        code,
        glyph_id: Some(1),
        width: 0.5,
        x_position: 0.0,
        tj_offset: None,
        font_size: 12.0,
        is_ligature: false,
        original_ligature: None,
        protected_from_split: false,
    }
}

/// Helper: Convert string to CharacterInfo array
fn string_to_chars(s: &str) -> Vec<CharacterInfo> {
    s.chars().map(|ch| create_char_info(ch as u32)).collect()
}

#[test]
fn test_email_pattern_detection() {
    // Test that email pattern is detected
    let chars = string_to_chars("user@example.com");
    assert!(PatternDetector::has_email_pattern(&chars), "Should detect email pattern");
}

#[test]
fn test_email_pattern_with_subdomain() {
    // Test email with subdomain
    let chars = string_to_chars("user@mail.example.com");
    assert!(PatternDetector::has_email_pattern(&chars), "Should detect email with subdomain");
}

#[test]
fn test_email_pattern_with_plus() {
    // Test email with plus sign (common pattern)
    let chars = string_to_chars("user+tag@example.com");
    assert!(PatternDetector::has_email_pattern(&chars), "Should detect email with plus sign");
}

#[test]
fn test_email_pattern_no_domain() {
    // Test that @ without domain (no dot) is not detected as email
    let chars = string_to_chars("user@example");
    assert!(
        !PatternDetector::has_email_pattern(&chars),
        "Should not detect email without dot in domain"
    );
}

#[test]
fn test_email_protection_from_split() {
    // Test that email characters are protected after mark_pattern_contexts
    let mut chars = string_to_chars("user@example.com");
    let config = PatternPreservationConfig::default();

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // All characters should be protected
    for (i, ch) in chars.iter().enumerate() {
        assert!(ch.protected_from_split, "Email character {} should be protected", i);
    }
}

#[test]
fn test_url_pattern_http() {
    // Test http:// URL detection
    let chars = string_to_chars("http://example.com");
    assert!(PatternDetector::has_url_pattern(&chars), "Should detect http:// URL");
}

#[test]
fn test_url_pattern_https() {
    // Test https:// URL detection
    let chars = string_to_chars("https://example.com");
    assert!(PatternDetector::has_url_pattern(&chars), "Should detect https:// URL");
}

#[test]
fn test_url_pattern_ftp() {
    // Test ftp:// URL detection
    let chars = string_to_chars("ftp://ftp.example.com");
    assert!(PatternDetector::has_url_pattern(&chars), "Should detect ftp:// URL");
}

#[test]
fn test_url_pattern_mailto() {
    // Test mailto: URL detection
    let chars = string_to_chars("mailto:user@example.com");
    assert!(PatternDetector::has_url_pattern(&chars), "Should detect mailto: URL");
}

#[test]
fn test_url_protection_from_split() {
    // Test that URL characters are protected after mark_pattern_contexts
    let mut chars = string_to_chars("http://example.com");
    let config = PatternPreservationConfig::default();

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // All characters should be protected
    for (i, ch) in chars.iter().enumerate() {
        assert!(ch.protected_from_split, "URL character {} should be protected", i);
    }
}

#[test]
fn test_boundary_skip_in_email() {
    // Test that word boundaries are skipped within email addresses
    let mut chars = string_to_chars("user@example.com");
    let config = PatternPreservationConfig::default();

    // Mark patterns
    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // Try to detect boundaries - should find none because all chars are protected
    let context = BoundaryContext::new(12.0);
    let detector = WordBoundaryDetector::new();
    let boundaries = detector.detect_word_boundaries(&chars, &context);

    // No boundaries should be created within the email
    assert!(boundaries.is_empty(), "No boundaries should be created within protected email");
}

#[test]
fn test_boundary_skip_in_url() {
    // Test that word boundaries are skipped within URLs
    let mut chars = string_to_chars("http://example.com");
    let config = PatternPreservationConfig::default();

    // Mark patterns
    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // Try to detect boundaries - should find none because all chars are protected
    let context = BoundaryContext::new(12.0);
    let detector = WordBoundaryDetector::new();
    let boundaries = detector.detect_word_boundaries(&chars, &context);

    // No boundaries should be created within the URL
    assert!(boundaries.is_empty(), "No boundaries should be created within protected URL");
}

#[test]
fn test_false_positive_version_number() {
    // Test that "version 2.0" is NOT treated as email
    // (has . but no @)
    let chars = string_to_chars("version 2.0");
    assert!(
        !PatternDetector::has_email_pattern(&chars),
        "Version number should not be detected as email"
    );
}

#[test]
fn test_multiple_patterns_in_text() {
    // Test text with both email and URL
    let mut chars = string_to_chars("Contact user@example.com or visit http://example.com");
    let config = PatternPreservationConfig::default();

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // Extract portions
    let text = "Contact user@example.com or visit http://example.com";
    let email_start = text.find("user@").unwrap();
    let email_end = email_start + "user@example.com".len();
    let url_start = text.find("http://").unwrap();
    let url_end = url_start + "http://example.com".len();

    // Email characters should be protected
    for i in email_start..email_end {
        assert!(chars[i].protected_from_split, "Email character {} should be protected", i);
    }

    // URL characters should be protected
    for i in url_start..url_end {
        assert!(chars[i].protected_from_split, "URL character {} should be protected", i);
    }

    // Non-pattern characters should not be protected
    for i in 0..email_start {
        assert!(
            !chars[i].protected_from_split,
            "Non-pattern character {} should not be protected",
            i
        );
    }
}

#[test]
fn test_pattern_detection_config_flag() {
    // Test that preserve_patterns = false disables all pattern detection
    let mut chars = string_to_chars("user@example.com");
    let config = PatternPreservationConfig {
        preserve_patterns: false,
        detect_emails: true,
        detect_urls: true,
    };

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // No characters should be protected when disabled
    for ch in &chars {
        assert!(
            !ch.protected_from_split,
            "Characters should not be protected when pattern detection is disabled"
        );
    }
}

#[test]
fn test_email_detection_disabled() {
    // Test that detect_emails = false disables email detection
    let mut chars = string_to_chars("user@example.com");
    let config = PatternPreservationConfig {
        preserve_patterns: true,
        detect_emails: false,
        detect_urls: true,
    };

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // No characters should be protected when email detection is disabled
    for ch in &chars {
        assert!(
            !ch.protected_from_split,
            "Characters should not be protected when email detection is disabled"
        );
    }
}

#[test]
fn test_url_detection_disabled() {
    // Test that detect_urls = false disables URL detection
    let mut chars = string_to_chars("http://example.com");
    let config = PatternPreservationConfig {
        preserve_patterns: true,
        detect_emails: true,
        detect_urls: false,
    };

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    // No characters should be protected when URL detection is disabled
    for ch in &chars {
        assert!(
            !ch.protected_from_split,
            "Characters should not be protected when URL detection is disabled"
        );
    }
}

#[test]
fn test_mixed_content_partial_protection() {
    // Test mixed content where only patterns are protected
    let mut chars = string_to_chars("Email: user@example.com for info");
    let config = PatternPreservationConfig::default();

    PatternDetector::mark_pattern_contexts(&mut chars, &config).unwrap();

    let text = "Email: user@example.com for info";
    let email_start = text.find("user@").unwrap();
    let email_end = email_start + "user@example.com".len();

    // Email portion should be protected
    for i in email_start..email_end {
        assert!(chars[i].protected_from_split, "Email character {} should be protected", i);
    }

    // Text before email should not be protected
    for i in 0..email_start {
        assert!(
            !chars[i].protected_from_split,
            "Non-email character {} should not be protected",
            i
        );
    }

    // Text after email should not be protected
    for i in email_end..chars.len() {
        assert!(
            !chars[i].protected_from_split,
            "Non-email character {} should not be protected",
            i
        );
    }
}

#[test]
fn test_url_with_path() {
    // Test URL with path and query parameters
    let chars = string_to_chars("https://example.com/path?q=1");
    assert!(
        PatternDetector::has_url_pattern(&chars),
        "Should detect URL with path and query"
    );
}

#[test]
fn test_url_with_port() {
    // Test URL with port number
    let chars = string_to_chars("http://example.com:8080");
    assert!(PatternDetector::has_url_pattern(&chars), "Should detect URL with port");
}

#[test]
fn test_email_case_insensitive() {
    // Email addresses are case-insensitive
    let chars_upper = string_to_chars("USER@EXAMPLE.COM");
    assert!(
        PatternDetector::has_email_pattern(&chars_upper),
        "Should detect uppercase email"
    );

    let chars_mixed = string_to_chars("User@Example.Com");
    assert!(
        PatternDetector::has_email_pattern(&chars_mixed),
        "Should detect mixed-case email"
    );
}

#[test]
fn test_url_case_insensitive() {
    // URL schemes are case-insensitive
    let chars_upper = string_to_chars("HTTP://EXAMPLE.COM");
    assert!(
        PatternDetector::has_url_pattern(&chars_upper),
        "Should detect uppercase HTTP URL"
    );

    let chars_mixed = string_to_chars("HtTp://Example.Com");
    assert!(
        PatternDetector::has_url_pattern(&chars_mixed),
        "Should detect mixed-case HTTP URL"
    );
}
