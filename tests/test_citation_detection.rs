//! TDD Tests for Citation Reference Detection
//!
//! Tests verify that common citation formats are properly detected and preserved
//! in extracted text without corruption or fragmentation.

#[test]
fn test_detect_numeric_citation_brackets() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "This is a fact[1] with citation.";
    let citations = detector.detect_citations(text);

    assert!(!citations.is_empty());
    assert!(citations.iter().any(|c| c.text == "[1]"));
}

#[test]
fn test_detect_numeric_citation_parens() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "This is a fact(23) with citation.";
    let citations = detector.detect_citations(text);

    assert!(!citations.is_empty());
}

#[test]
fn test_detect_superscript_citation() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "This is a fact¹ with superscript.";
    let citations = detector.detect_citations(text);

    assert!(!citations.is_empty());
}

#[test]
fn test_detect_author_year_citation() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "As Smith (2020) noted in their work.";
    let citations = detector.detect_citations(text);

    assert!(!citations.is_empty());
}

#[test]
fn test_detect_author_year_with_comma() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "As Smith, 2020 noted in their work.";
    let citations = detector.detect_citations(text);

    // Should detect even with comma
    assert!(!citations.is_empty());
}

#[test]
fn test_detect_et_al_citation() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "Previous work [Smith et al., 2020] showed this.";
    let citations = detector.detect_citations(text);

    assert!(!citations.is_empty());
}

#[test]
fn test_preserve_citation_integrity() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "This fact[1] remains true.";
    let citations = detector.detect_citations(text);
    let preserved = detector.preserve_citations(text, &citations);

    // Citation should not be split
    assert!(preserved.contains("[1]"));
}

#[test]
fn test_no_false_positive_numbers() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "The year 2020 was significant.";
    let citations = detector.detect_citations(text);

    // Should not detect "2020" as a citation
    assert!(citations.is_empty());
}

#[test]
fn test_multiple_citations_detected() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "First[1] and second[2] citations.";
    let citations = detector.detect_citations(text);

    assert_eq!(citations.len(), 2);
}

#[test]
fn test_citation_with_adjacent_punctuation() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "This fact[1]. And another[2]!";
    let citations = detector.detect_citations(text);

    assert_eq!(citations.len(), 2);
}

#[test]
fn test_detect_range_citation() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "Multiple authors[1-3] contributed.";
    let citations = detector.detect_citations(text);

    assert!(!citations.is_empty());
}

#[test]
fn test_citation_position_accuracy() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "Text[1]more";
    let citations = detector.detect_citations(text);

    // Citation should be detected at correct position
    if let Some(citation) = citations.first() {
        assert!(citation.position >= 4); // After "Text"
        assert!(citation.position <= 5);
    }
}

#[test]
fn test_classify_citation_type() {
    use pdf_oxide::pipeline::text_processing::{CitationDetector, CitationType};

    let detector = CitationDetector::new();

    let numeric = detector.detect_citations("[1]");
    if let Some(c) = numeric.first() {
        assert_eq!(c.citation_type, CitationType::Numeric);
    }

    let author_year = detector.detect_citations("(Smith, 2020)");
    if let Some(c) = author_year.first() {
        assert_eq!(c.citation_type, CitationType::AuthorYear);
    }
}

#[test]
fn test_normalize_citation_formatting() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "Text[  1  ] more";
    let citations = detector.detect_citations(text);
    let preserved = detector.preserve_citations(text, &citations);

    // Should normalize spacing in citation
    assert!(preserved.contains("[1]") || !preserved.contains("[  1  ]"));
}

#[test]
fn test_mixed_citation_styles() {
    use pdf_oxide::pipeline::text_processing::CitationDetector;

    let detector = CitationDetector::new();
    let text = "Smith (2020)[1] showed [2] that (Jones, 2021) worked.";
    let citations = detector.detect_citations(text);

    // Should detect all citation styles
    assert!(citations.len() >= 3);
}
