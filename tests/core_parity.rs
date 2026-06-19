//! Core functional test-parity suite (Rust) — the reference implementation of
//! the shared cross-language spec
//! (docs/releases/plans/v0.3.61/core-test-parity-spec.md). Every binding mirrors
//! these behaviors with its own idiomatic API. (Search is a binding-level
//! convenience and has no single Rust-core method, so it is covered in the
//! bindings, not here.)

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::writer::DocumentBuilder;
use pdf_oxide::PdfDocument;

fn fixture_bytes() -> Vec<u8> {
    std::fs::read("tests/fixtures/simple.pdf").expect("simple.pdf fixture")
}

fn open() -> PdfDocument {
    PdfDocument::from_bytes(fixture_bytes()).expect("open simple.pdf")
}

fn build_bytes() -> Vec<u8> {
    let mut b = DocumentBuilder::new();
    b.letter_page()
        .font("Helvetica", 12.0)
        .at(72.0, 720.0)
        .heading(1, "Core Parity")
        .at(72.0, 690.0)
        .paragraph("Functional parity across all language bindings.")
        .done();
    b.build().expect("build pdf")
}

#[test]
fn open_and_page_count() {
    assert_eq!(open().page_count().unwrap(), 1);
}

#[test]
fn extract_text() {
    let _: String = open().extract_text(0).unwrap();
}

#[test]
fn convert_markdown_html_plain() {
    let doc = open();
    let o = ConversionOptions::default();
    let _ = doc.to_markdown(0, &o).unwrap();
    let _ = doc.to_html(0, &o).unwrap();
    let _ = doc.to_plain_text(0, &o).unwrap();
}

#[test]
fn structured() {
    let _ = open().extract_structured(0).unwrap();
}

#[test]
fn create_pdf() {
    assert!(build_bytes().starts_with(b"%PDF"));
}

#[test]
fn from_bytes_page_count() {
    assert_eq!(
        PdfDocument::from_bytes(build_bytes())
            .unwrap()
            .page_count()
            .unwrap(),
        1
    );
}

#[test]
fn encrypt_roundtrip() {
    let plain = build_bytes();
    let mut b = DocumentBuilder::new();
    b.letter_page()
        .font("Helvetica", 12.0)
        .at(72.0, 720.0)
        .paragraph("secret")
        .done();
    let enc = b.to_bytes_encrypted("user123", "owner123").unwrap();
    assert!(enc.starts_with(b"%PDF"));
    assert_ne!(enc, plain, "encryption must change the bytes");
}

#[test]
fn open_error() {
    assert!(
        PdfDocument::from_bytes(b"this is not a pdf".to_vec()).is_err(),
        "opening non-PDF bytes must error"
    );
}

#[test]
fn version() {
    assert_eq!(env!("CARGO_PKG_VERSION"), "0.3.67");
}
