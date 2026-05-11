//! Regression test for encrypted PDF with CID TrueType Identity-H fonts.
//!
//! Issue #202: `extract_text()` returns empty string for an RC4-encrypted PDF
//! (V=4 with CFM=V2) containing CID TrueType fonts and Identity-H encoding.
//!
//! Root causes:
//! 1. `EncryptDict::algorithm()` assumed V=4 always means AES-128, but the PDF
//!    uses `/CFM /V2` (RC4-128). AES decryption failed on non-16-aligned data.
//! 2. `CIDToGIDMap` stream was decoded but never decrypted, producing garbage
//!    GID mappings even when decryption is otherwise correct.

#[cfg(feature = "legacy-crypto")]
use pdf_oxide::document::PdfDocument;

/// Verify text extraction works for an encrypted PDF with V=4/CFM=V2 (RC4-128)
/// and CID TrueType Identity-H fonts (OpenPDF 1.3.26).
///
/// This test requires the `legacy-crypto` feature because the fixture uses RC4-128
/// (PDF Standard Security R=4), which relies on MD5 key derivation. Under the FIPS
/// build (`--no-default-features --features fips,icc`) that algorithm is deliberately
/// disabled, so the test is excluded from compilation in that configuration.
#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encrypted_cid_truetype_extract_text_non_empty() {
    let pdf_path = "tests/fixtures/encrypted_cid_truetype.pdf";
    if !std::path::Path::new(pdf_path).exists() {
        eprintln!("Skipping test: fixture not found at {}", pdf_path);
        return;
    }

    let doc = PdfDocument::open(pdf_path).expect("Failed to open encrypted PDF");

    let page_count = doc.page_count().expect("Failed to get page count");
    assert!(page_count > 0, "PDF should have at least one page");

    let mut total_text = String::new();
    for page_num in 0..page_count {
        let text = doc
            .extract_text(page_num)
            .unwrap_or_else(|e| panic!("Failed to extract text from page {}: {}", page_num, e));
        total_text.push_str(&text);
    }

    // The PDF is an invoice — it must contain extractable text
    assert!(
        !total_text.trim().is_empty(),
        "extract_text() returned empty for encrypted CID TrueType PDF (issue #202)"
    );

    // Verify we got a reasonable amount of text (not just a few stray characters)
    let char_count = total_text.chars().count();
    assert!(
        char_count > 50,
        "Expected substantial text from invoice PDF, got only {} characters: {:?}",
        char_count,
        &total_text[..total_text.len().min(200)]
    );
}
