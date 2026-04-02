use pdf_oxide::document::PdfDocument;

fn make_test_font(name: &str, subtype: &str) -> pdf_oxide::fonts::FontInfo {
    use std::collections::HashMap;
    pdf_oxide::fonts::FontInfo {
        base_font: name.to_string(),
        subtype: subtype.to_string(),
        encoding: pdf_oxide::fonts::Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
    }
}

#[test]
fn test_tf_buffer_flush_on_font_switch() {
    let stream = b"BT /F1 12 Tf 100 700 Td (AB) Tj /F2 12 Tf (CD) Tj ET";

    let mut extractor = pdf_oxide::extractors::text::TextExtractor::new();
    extractor.add_font("F1".to_string(), make_test_font("Helvetica", "Type1"));
    extractor.add_font("F2".to_string(), make_test_font("Courier", "Type1"));

    let chars = extractor.extract(stream).unwrap();
    let text: String = chars.iter().map(|c| c.char).collect();

    assert!(text.contains("AB"), "Text from first font F1 missing: got '{}'", text);
    assert!(text.contains("CD"), "Text from second font F2 missing: got '{}'", text);
    assert_eq!(chars.len(), 4, "Expected 4 chars, got {}: '{}'", chars.len(), text);
}

#[test]
fn test_tf_buffer_flush_across_three_fonts() {
    let stream = b"BT /F1 10 Tf 0 700 Td (One) Tj /F2 10 Tf (Two) Tj /F3 10 Tf (Three) Tj ET";

    let mut extractor = pdf_oxide::extractors::text::TextExtractor::new();
    extractor.add_font("F1".to_string(), make_test_font("Helvetica", "Type1"));
    extractor.add_font("F2".to_string(), make_test_font("Courier", "Type1"));
    extractor.add_font("F3".to_string(), make_test_font("Times-Roman", "Type1"));

    let chars = extractor.extract(stream).unwrap();
    let text: String = chars.iter().map(|c| c.char).collect();

    assert_eq!(
        chars.len(),
        11,
        "Expected 11 chars (One+Two+Three), got {}: '{}'",
        chars.len(),
        text
    );
}

#[test]
fn test_annotation_extraction_no_panic() {
    for fixture in &["tests/fixtures/simple.pdf", "tests/fixtures/outline.pdf"] {
        let mut doc = PdfDocument::open(fixture).unwrap();
        let pages = doc.page_count().unwrap();
        for p in 0..pages {
            let _text = doc.extract_text(p).unwrap();
        }
    }
}

#[test]
fn test_space_ratio_below_threshold() {
    let mut doc = PdfDocument::open("tests/fixtures/outline.pdf").unwrap();
    let pages = doc.page_count().unwrap();

    for p in 0..pages {
        let text = doc.extract_text(p).unwrap();
        if text.is_empty() {
            continue;
        }
        let spaces = text.chars().filter(|c| *c == ' ').count();
        let non_ws = text.chars().filter(|c| !c.is_whitespace()).count();
        if non_ws > 10 {
            let ratio = spaces as f64 / non_ws as f64;
            assert!(
                ratio < 0.5,
                "Page {}: space ratio {:.2} too high ({} spaces / {} chars)",
                p,
                ratio,
                spaces,
                non_ws
            );
        }
    }
}

#[test]
fn test_bt_et_block_produces_output() {
    let stream = b"BT /F1 12 Tf 72 700 Td (Hello World) Tj ET";

    let mut extractor = pdf_oxide::extractors::text::TextExtractor::new();
    extractor.add_font("F1".to_string(), make_test_font("Helvetica", "Type1"));

    let chars = extractor.extract(stream).unwrap();
    assert!(!chars.is_empty(), "Valid BT/ET block should produce non-empty output");
    let text: String = chars.iter().map(|c| c.char).collect();
    assert!(text.contains("Hello"), "Expected 'Hello' in output, got '{}'", text);
}

#[test]
fn test_multiple_bt_et_blocks_all_extracted() {
    let stream = b"BT /F1 12 Tf 72 700 Td (First) Tj ET BT /F1 12 Tf 72 680 Td (Second) Tj ET";

    let mut extractor = pdf_oxide::extractors::text::TextExtractor::new();
    extractor.add_font("F1".to_string(), make_test_font("Helvetica", "Type1"));

    let chars = extractor.extract(stream).unwrap();
    let text: String = chars.iter().map(|c| c.char).collect();

    assert!(text.contains("First"), "First BT/ET block text missing");
    assert!(text.contains("Second"), "Second BT/ET block text missing");
}

#[test]
fn test_overlapping_duplicate_chars_removed() {
    let stream = b"BT /F1 12 Tf 100 700 Td (AB) Tj ET BT /F1 12 Tf 100 700 Td (AB) Tj ET";

    let mut extractor = pdf_oxide::extractors::text::TextExtractor::new();
    extractor.add_font("F1".to_string(), make_test_font("Helvetica", "Type1"));

    let chars = extractor.extract(stream).unwrap();

    assert!(
        chars.len() <= 3,
        "Expected deduplication to reduce 4 overlapping chars, got {}",
        chars.len()
    );
}

#[test]
fn test_distinct_lines_not_falsely_deduplicated() {
    let stream = b"BT /F1 12 Tf 100 700 Td (AB) Tj ET BT /F1 12 Tf 100 680 Td (CD) Tj ET";

    let mut extractor = pdf_oxide::extractors::text::TextExtractor::new();
    extractor.add_font("F1".to_string(), make_test_font("Helvetica", "Type1"));

    let chars = extractor.extract(stream).unwrap();
    let text: String = chars.iter().map(|c| c.char).collect();

    assert!(
        text.contains("AB") && text.contains("CD"),
        "Text at different Y positions must not be deduplicated, got '{}'",
        text
    );
    assert_eq!(chars.len(), 4, "Expected 4 chars on 2 lines, got {}", chars.len());
}

#[test]
fn test_brotli_decode_roundtrip() {
    use pdf_oxide::decoders::BrotliDecoder;
    use pdf_oxide::decoders::StreamDecoder;

    let original = b"The quick brown fox jumps over the lazy dog. PDF stream data.";

    let mut compressed = Vec::new();
    {
        let mut writer = brotli::CompressorWriter::new(&mut compressed, 4096, 6, 22);
        std::io::Write::write_all(&mut writer, original).unwrap();
    }

    let decoder = BrotliDecoder;
    let decoded = decoder.decode(&compressed).unwrap();
    assert_eq!(decoded, original.to_vec());
}

#[test]
fn test_fixture_deterministic_output() {
    for fixture in &["tests/fixtures/simple.pdf", "tests/fixtures/outline.pdf"] {
        let mut doc1 = PdfDocument::open(fixture).unwrap();
        let mut doc2 = PdfDocument::open(fixture).unwrap();
        let pages = doc1.page_count().unwrap();
        for p in 0..pages {
            let t1 = doc1.extract_text(p).unwrap();
            let t2 = doc2.extract_text(p).unwrap();
            assert_eq!(t1, t2, "{} page {} not deterministic", fixture, p);
        }
    }
}

#[test]
fn test_fixture_no_panic() {
    for fixture in &["tests/fixtures/simple.pdf", "tests/fixtures/outline.pdf"] {
        let mut doc = PdfDocument::open(fixture).unwrap();
        let pages = doc.page_count().unwrap();
        assert!(pages > 0, "{} should have at least 1 page", fixture);
        for p in 0..pages {
            let _text = doc.extract_text(p).unwrap();
        }
    }
}
