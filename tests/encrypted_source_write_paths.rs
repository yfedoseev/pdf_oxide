//! Every write path that copies objects out of an already-open, encrypted
//! source document — `save()`/`save_to_bytes()`, `extract_pages()`,
//! `remove_page()` + save — must decrypt stream data before re-emitting it
//! into an output with no `/Encrypt` dictionary of its own. Before this
//! fix, they re-serialized the source's raw ciphertext verbatim: the
//! output was a structurally valid PDF that opened without a password, but
//! every copied content stream was still AES ciphertext behind
//! `/Filter /FlateDecode`, so a conforming reader failed to inflate it and
//! rendered a blank page — silently, with no error or warning (#1032).
//!
//! The fixture is built entirely in-code (no third-party PDF): a plain
//! single-page PDF with known text, encrypted AES-128 via this crate's own
//! authoring path (`DocumentEditor::save_with_options` +
//! `SaveOptions::with_encryption`), matching the exact shape of the
//! original report (a small AES-128 `/V 4 /R 4` document).
//!
//! `/V 4 /R 4` derives its key via MD5, which the FIPS `CryptoProvider`
//! refuses to build regardless of the stream cipher being AES (FIPS 140-3
//! forbids MD5 outright) — this whole file is gated to non-FIPS builds,
//! matching the same fixture-shape constraint already applied to
//! `permissions_some_on_encrypted_pdf` in
//! `tests/extraction_api_regression.rs`.
#![cfg(not(feature = "fips"))]

use pdf_oxide::editor::{
    DocumentEditor, EditableDocument, EncryptionAlgorithm, EncryptionConfig, SaveOptions,
};
use pdf_oxide::writer::{DocumentBuilder, DocumentMetadata, PageSize};
use pdf_oxide::PdfDocument;

const SECRET_TEXT: &str = "HELLO-FROM-PAGE-ONE";
const USER_PASSWORD: &str = "test-password";

/// A plain, single-page PDF containing `SECRET_TEXT`, then encrypted
/// AES-128 with `USER_PASSWORD`, matching the original report's fixture
/// shape (`/V 4 /R 4 /CFM /AESV2`).
fn build_encrypted_source() -> Vec<u8> {
    let mut builder = DocumentBuilder::new();
    builder = builder.metadata(DocumentMetadata::new().title("Encrypted Source"));
    {
        let page = builder.page(PageSize::Letter);
        page.at(72.0, 720.0).text(SECRET_TEXT).done();
    }
    let plain_pdf = builder.build().expect("build plain pdf");

    let mut editor = DocumentEditor::from_bytes(plain_pdf).expect("open plain pdf");
    let config = EncryptionConfig::new(USER_PASSWORD, "owner-password")
        .with_algorithm(EncryptionAlgorithm::Aes128);
    editor
        .save_to_bytes_with_options(SaveOptions::with_encryption(config))
        .expect("save encrypted pdf")
}

/// Open the encrypted fixture and authenticate, returning a `DocumentEditor`
/// ready to exercise a write path.
fn open_authenticated_editor() -> DocumentEditor {
    let doc = PdfDocument::from_bytes(build_encrypted_source()).expect("parse encrypted pdf");
    assert!(doc.is_encrypted(), "fixture must actually be encrypted");
    let authed = doc
        .authenticate(USER_PASSWORD.as_bytes())
        .expect("authenticate");
    assert!(authed, "authentication with the correct password must succeed");
    assert!(doc.is_authenticated(), "document must report authenticated");
    DocumentEditor::from_document(doc).expect("wrap authenticated document")
}

/// The output of a write path must contain the original plaintext (not
/// ciphertext/garbage), and — since we never asked for output encryption —
/// must carry no `/Encrypt` dictionary.
fn assert_output_is_readable_plaintext(output_bytes: &[u8]) {
    let out_doc = PdfDocument::from_bytes(output_bytes.to_vec()).expect("parse output pdf");
    assert!(
        !out_doc.is_encrypted(),
        "output must not carry an /Encrypt dictionary when none was requested"
    );
    let text = out_doc.extract_text(0).expect("extract_text on output");
    assert!(
        text.contains(SECRET_TEXT),
        "output must contain the original plaintext, got: {text:?}"
    );
}

#[test]
fn save_to_bytes_decrypts_encrypted_source() {
    let mut editor = open_authenticated_editor();
    let output = editor
        .save_to_bytes()
        .expect("save_to_bytes on authenticated source");
    assert_output_is_readable_plaintext(&output);
}

#[test]
fn extract_pages_decrypts_encrypted_source() {
    let mut editor = open_authenticated_editor();
    let output = editor
        .extract_pages_to_bytes(&[0])
        .expect("extract_pages_to_bytes on authenticated source");
    assert_output_is_readable_plaintext(&output);
}

#[test]
fn remove_page_then_save_decrypts_remaining_pages() {
    // Build a 2-page encrypted source so there's a page left after removal.
    let mut builder = DocumentBuilder::new();
    builder = builder.metadata(DocumentMetadata::new().title("Encrypted Source 2 pages"));
    {
        let page = builder.page(PageSize::Letter);
        page.at(72.0, 720.0).text("REMOVE ME").done();
    }
    {
        let page = builder.page(PageSize::Letter);
        page.at(72.0, 720.0).text(SECRET_TEXT).done();
    }
    let plain_pdf = builder.build().expect("build plain pdf");
    let mut plain_editor = DocumentEditor::from_bytes(plain_pdf).expect("open plain pdf");
    let config = EncryptionConfig::new(USER_PASSWORD, "owner-password")
        .with_algorithm(EncryptionAlgorithm::Aes128);
    let encrypted = plain_editor
        .save_to_bytes_with_options(SaveOptions::with_encryption(config))
        .expect("save encrypted 2-page pdf");

    let doc = PdfDocument::from_bytes(encrypted).expect("parse encrypted pdf");
    doc.authenticate(USER_PASSWORD.as_bytes())
        .expect("authenticate");
    let mut editor = DocumentEditor::from_document(doc).expect("wrap authenticated document");

    EditableDocument::remove_page(&mut editor, 0).expect("remove first page");
    let output = editor.save_to_bytes().expect("save after remove_page");
    assert_output_is_readable_plaintext(&output);
}

#[test]
fn write_incremental_refuses_encrypted_source() {
    let mut editor = open_authenticated_editor();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("incremental.pdf");
    let result = editor.save_with_options(
        &path,
        SaveOptions {
            incremental: true,
            ..SaveOptions::full_rewrite()
        },
    );
    assert!(
        result.is_err(),
        "incremental save on an encrypted source must fail closed, not silently corrupt"
    );
}
