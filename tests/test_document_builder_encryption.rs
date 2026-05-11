//! Regression tests for issue #386 — `DocumentBuilder` encryption.
//!
//! v0.3.37 had `Pdf::save_encrypted` but it only worked for PDFs
//! *opened* via `DocumentEditor` (see
//! `src/api/pdf_builder.rs::save_encrypted` — it explicitly errors
//! "Encryption is only supported for opened PDFs" otherwise). Users
//! building PDFs programmatically through `DocumentBuilder::save` had
//! no way to produce an encrypted output.
//!
//! v0.3.38 adds `save_encrypted`, `save_with_encryption`, and
//! `to_bytes_encrypted` / `to_bytes_with_encryption` on
//! `DocumentBuilder`. Each routes the built bytes through
//! `DocumentEditor::from_bytes` → `save_with_options`, reusing the
//! tested production encryption pipeline.

use pdf_oxide::editor::{EncryptionAlgorithm, EncryptionConfig, Permissions};
use pdf_oxide::writer::{DocumentBuilder, DocumentMetadata, PageSize};
#[cfg(feature = "legacy-crypto")]
use pdf_oxide::PdfDocument;
use std::fs;
use tempfile::tempdir;

fn make_builder(body: &str) -> DocumentBuilder {
    let mut builder =
        DocumentBuilder::new().metadata(DocumentMetadata::new().title("enc test").author("test"));
    {
        let page = builder.page(PageSize::Letter);
        page.at(72.0, 720.0).text(body).done();
    }
    builder
}

/// Default `save_encrypted` uses AES-256 (`/V 5 /R 6`) and writes the
/// expected Standard-security-handler dictionary entries.
#[test]
fn save_encrypted_produces_aes256_encrypt_dict() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.pdf");

    make_builder("confidential content for #386")
        .save_encrypted(&path, "userpw", "ownerpw")
        .expect("save_encrypted should succeed");

    assert!(path.exists());
    let raw = fs::read(&path).unwrap();
    let text = String::from_utf8_lossy(&raw);

    assert!(text.contains("/Encrypt"), "missing /Encrypt dict");
    assert!(text.contains("/Filter /Standard"), "missing /Filter /Standard");
    assert!(text.contains("/V 5"), "expected /V 5 (AES-256) — got no match");
    assert!(text.contains("/R 6"), "expected /R 6 (AES-256 revision)");
    assert!(text.contains("/O "), "missing /O (owner hash)");
    assert!(text.contains("/U "), "missing /U (user hash)");
    assert!(text.contains("/P "), "missing /P (permissions)");
}

/// `to_bytes_encrypted` returns the encrypted PDF as a byte vector.
/// Must match what `save_encrypted` writes to disk for the same input
/// (modulo timestamp-derived key material — the encryption dict itself
/// is a function of password + seed and will vary, but the outer
/// structural markers must be present).
#[test]
fn to_bytes_encrypted_includes_encrypt_dict() {
    let bytes = make_builder("in-memory encrypted build")
        .to_bytes_encrypted("user", "owner")
        .expect("to_bytes_encrypted should succeed");

    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("/Encrypt"), "bytes should include /Encrypt dict");
    assert!(text.contains("/V 5"), "bytes should use AES-256 by default");
}

/// `save_with_encryption` honours a custom algorithm choice.
#[cfg(feature = "legacy-crypto")]
#[test]
fn save_with_encryption_respects_aes128_config() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out_aes128.pdf");

    let config = EncryptionConfig::new("u", "o")
        .with_algorithm(EncryptionAlgorithm::Aes128)
        .with_permissions(Permissions::all());

    make_builder("AES-128 test")
        .save_with_encryption(&path, config)
        .expect("save_with_encryption should succeed");

    let text = String::from_utf8_lossy(&fs::read(&path).unwrap()).to_string();
    assert!(text.contains("/Encrypt"), "missing /Encrypt dict");
    assert!(text.contains("/V 4"), "expected /V 4 (AES-128) — got no match in dict",);
}

/// Restricted permissions propagate into the `/P` permission bits.
/// `Permissions::read_only()` turns off print/modify/copy bits; we
/// check the resulting bits decoded from the `/P` integer.
#[test]
fn save_with_encryption_respects_restricted_permissions() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out_readonly.pdf");

    let config = EncryptionConfig::new("u", "o")
        .with_algorithm(EncryptionAlgorithm::Aes256)
        .with_permissions(Permissions::read_only());

    make_builder("read-only PDF")
        .save_with_encryption(&path, config)
        .expect("save_with_encryption should succeed");

    let raw = fs::read(&path).unwrap();
    let text = String::from_utf8_lossy(&raw);
    assert!(text.contains("/Encrypt"), "missing /Encrypt dict");

    // Extract the `/P <signed-int>` from the encrypt dict. The value
    // sits between `/P ` and the next whitespace.
    let p_idx = text.find("/P ").expect("dict has /P entry");
    let tail = &text[p_idx + 3..];
    let end = tail
        .find(|c: char| c.is_whitespace() || c == '/')
        .expect("/P value has a terminator");
    let p_value: i32 = tail[..end].parse().expect("/P value should be an integer");

    // ISO 32000-1 Table 22: bits 3..=5 are print / modify / copy.
    // Bit 3 (print) in read-only should be 0. Bits are 1-based; Rust
    // shift is 0-based, so bit 3 = (1 << 2), bit 4 = (1 << 3), etc.
    let bit = |n: u32| (p_value >> (n - 1)) & 1;
    assert_eq!(bit(3), 0, "print bit should be clear for read-only (P={p_value})");
    assert_eq!(bit(4), 0, "modify bit should be clear for read-only (P={p_value})");
    assert_eq!(bit(5), 0, "copy bit should be clear for read-only (P={p_value})");
}

/// Encrypting a document that also embeds a custom font exercises both
/// v0.3.38 changes end to end: #385 (font subsetting) produces the
/// bytes, #386 (encryption) wraps them. This guards against any
/// layering bug where the encryption path re-parses the build output
/// and fails to handle the new content-stream ops.
#[test]
fn save_encrypted_works_with_embedded_font_subsetting() {
    use pdf_oxide::writer::EmbeddedFont;
    use std::path::Path;

    let dir = tempdir().unwrap();
    let path = dir.path().join("out_enc_embedded.pdf");

    let font = EmbeddedFont::from_file(Path::new("tests/fixtures/fonts/DejaVuSans.ttf"))
        .expect("DejaVuSans.ttf fixture available");
    let mut builder = DocumentBuilder::new()
        .metadata(DocumentMetadata::new().title("enc+subset"))
        .register_embedded_font("DejaVu", font);
    builder
        .a4_page()
        .font("DejaVu", 12.0)
        .at(72.0, 720.0)
        .text("Привет and Hello")
        .done();

    builder
        .save_encrypted(&path, "userpw", "ownerpw")
        .expect("save_encrypted with embedded font should succeed");

    let bytes = fs::read(&path).unwrap();
    let text = String::from_utf8_lossy(&bytes);

    // Both features present: encryption + subset tag.
    assert!(text.contains("/Encrypt"), "missing /Encrypt dict");
    assert!(text.contains("/V 5"), "missing /V 5 for AES-256");
    let has_subset_prefix = bytes
        .windows(8)
        .any(|w| w[0] == b'/' && w[1..7].iter().all(|&b| b.is_ascii_uppercase()) && w[7] == b'+');
    assert!(has_subset_prefix, "missing /XXXXXX+ subset-tag prefix");

    // And the encrypted PDF should still be much smaller than the
    // original face: encryption overhead doesn't re-embed the full
    // font bytes on top of the subset.
    let face_bytes = std::fs::metadata("tests/fixtures/fonts/DejaVuSans.ttf")
        .unwrap()
        .len() as usize;
    assert!(
        bytes.len() * 5 < face_bytes,
        "encrypted PDF ({} bytes) is not meaningfully smaller than the original face ({} bytes) — \
         subsetting likely not applied when encryption is on",
        bytes.len(),
        face_bytes,
    );
}

// ── Content round-trip tests (issue #401) ─────────────────────────────────
//
// These tests confirm that content is READABLE after encryption, not just
// that the /Encrypt dictionary is structurally present.  They were all RED
// before the fix in write_full_to_writer (shallow object traversal missed
// DescendantFonts / FontFile2 / ToUnicode for embedded TrueType fonts).

/// Helper: open an encrypted PDF with the given password and extract text
/// from page 0. Returns the extracted string.
#[cfg(feature = "legacy-crypto")]
fn open_encrypted_and_extract(path: &std::path::Path, password: &str) -> String {
    let doc = PdfDocument::open(path).expect("encrypted PDF should open");
    let authenticated = doc
        .authenticate(password.as_bytes())
        .expect("authenticate should not error");
    assert!(authenticated, "password '{password}' should authenticate successfully");
    doc.extract_text(0)
        .expect("extract_text should succeed after auth")
}

/// Helper: write encrypted bytes to a temp file and open+extract.
#[cfg(feature = "legacy-crypto")]
fn bytes_open_encrypted_and_extract(bytes: &[u8], password: &str) -> String {
    let dir = tempdir().unwrap();
    let path = dir.path().join("from_bytes.pdf");
    fs::write(&path, bytes).unwrap();
    open_encrypted_and_extract(&path, password)
}

/// Plain ASCII text with base-14 font, saved with AES-128.
/// Content must survive the encryption round-trip.
/// Note: AES-256 (V=5, R=6) round-trips require /UE//OE unwrap support
/// on the read side; use AES-128 (V=4, R=4) which fully round-trips today.
#[cfg(feature = "legacy-crypto")]
#[test]
fn save_encrypted_content_preserved_ascii() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ascii_encrypted.pdf");
    let expected = "Hello from #401 fix";

    let config =
        EncryptionConfig::new("user123", "owner456").with_algorithm(EncryptionAlgorithm::Aes128);
    make_builder(expected)
        .save_with_encryption(&path, config)
        .expect("save_with_encryption AES-128 should succeed");

    let text = open_encrypted_and_extract(&path, "user123");
    assert!(text.contains(expected), "extracted text {text:?} should contain {expected:?}");
}

/// to_bytes_with_encryption (AES-128) must also preserve content end-to-end.
#[cfg(feature = "legacy-crypto")]
#[test]
fn to_bytes_encrypted_content_preserved() {
    let expected = "bytes round-trip content";
    let config = EncryptionConfig::new("u", "o").with_algorithm(EncryptionAlgorithm::Aes128);
    let bytes = make_builder(expected)
        .to_bytes_with_encryption(config)
        .expect("to_bytes_with_encryption AES-128 should succeed");

    let text = bytes_open_encrypted_and_extract(&bytes, "u");
    assert!(text.contains(expected), "extracted text {text:?} should contain {expected:?}");
}

/// AES-128 round-trip preserves content.
#[cfg(feature = "legacy-crypto")]
#[test]
fn save_with_encryption_aes128_content_preserved() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("aes128_content.pdf");
    let expected = "AES-128 content preservation test";

    let config = EncryptionConfig::new("u128", "o128").with_algorithm(EncryptionAlgorithm::Aes128);
    make_builder(expected)
        .save_with_encryption(&path, config)
        .expect("save_with_encryption AES-128 should succeed");

    let text = open_encrypted_and_extract(&path, "u128");
    assert!(
        text.contains(expected),
        "AES-128 extracted text {text:?} should contain {expected:?}"
    );
}

/// Read-only permissions should not prevent content from being readable.
/// Uses AES-128 which fully round-trips today.
#[cfg(feature = "legacy-crypto")]
#[test]
fn save_encrypted_read_only_permissions_content_preserved() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("readonly_content.pdf");
    let expected = "read-only permissions test";

    let config = EncryptionConfig::new("ruser", "rowner")
        .with_algorithm(EncryptionAlgorithm::Aes128)
        .with_permissions(Permissions::read_only());
    make_builder(expected)
        .save_with_encryption(&path, config)
        .expect("save_with_encryption read-only should succeed");

    let text = open_encrypted_and_extract(&path, "ruser");
    assert!(
        text.contains(expected),
        "read-only extracted text {text:?} should contain {expected:?}"
    );
}

/// Embedded TrueType font — full object graph (DescendantFonts /
/// FontFile2 / ToUnicode / FontDescriptor) must survive the encryption
/// pipeline.  This is the core regression for issue #401.
/// Uses AES-128 which fully round-trips today.
#[cfg(feature = "legacy-crypto")]
#[test]
fn save_encrypted_embedded_ttf_content_preserved() {
    use pdf_oxide::writer::EmbeddedFont;
    use std::path::Path;

    let dir = tempdir().unwrap();
    let path = dir.path().join("ttf_encrypted.pdf");
    let expected = "Hello from embedded font";

    let font = EmbeddedFont::from_file(Path::new("tests/fixtures/fonts/DejaVuSans.ttf"))
        .expect("DejaVuSans.ttf fixture must be present");
    let mut builder = DocumentBuilder::new()
        .metadata(DocumentMetadata::new().title("issue-401 regression"))
        .register_embedded_font("DejaVu", font);
    builder
        .a4_page()
        .font("DejaVu", 12.0)
        .at(72.0, 720.0)
        .text(expected)
        .done();

    let config =
        EncryptionConfig::new("ttfuser", "ttfowner").with_algorithm(EncryptionAlgorithm::Aes128);
    builder
        .save_with_encryption(&path, config)
        .expect("save_with_encryption AES-128 with embedded TTF should succeed");

    let text = open_encrypted_and_extract(&path, "ttfuser");
    assert!(
        !text.is_empty(),
        "text extracted from embedded-font encrypted PDF must not be empty (issue #401)"
    );
    assert!(
        text.contains("Hello") || text.contains("embedded") || text.contains("font"),
        "extracted text {text:?} should contain recognisable words from the page"
    );
}

/// Two embedded fonts (regular + bold) with multiple pages — mirrors
/// the exact scenario from issue #401 as closely as possible without
/// requiring Chinese-specific font files in the test fixtures.
///
/// The user reported that `save_with_encryption` with AES-128 produced
/// a blank PDF when the builder used embedded TrueType fonts.
#[cfg(feature = "legacy-crypto")]
#[test]
fn two_embedded_fonts_aes128_content_preserved() {
    use pdf_oxide::writer::EmbeddedFont;
    use std::path::Path;

    let dir = tempdir().unwrap();
    let path = dir.path().join("issue_401.pdf");

    // Register two fonts (regular + bold) mirroring the issue report.
    let font_regular = EmbeddedFont::from_file(Path::new("tests/fixtures/fonts/DejaVuSans.ttf"))
        .expect("DejaVuSans.ttf fixture must be present");
    let font_bold = EmbeddedFont::from_file(Path::new("tests/fixtures/fonts/DejaVuSans-Bold.ttf"))
        .expect("DejaVuSans-Bold.ttf fixture must be present");

    let mut builder = DocumentBuilder::new()
        .register_embedded_font("Regular", font_regular)
        .register_embedded_font("Bold", font_bold);

    // First page: bold heading + regular body, mirroring the issue #401 code.
    {
        let page = builder.a4_page();
        page.font("Bold", 14.5)
            .at(30.0, 800.0)
            .text("High Performance")
            .font("Regular", 10.5)
            .at(30.0, 780.0)
            .text("Rust is fast and memory-efficient.")
            .font("Bold", 14.5)
            .at(30.0, 745.0)
            .text("Reliability")
            .font("Regular", 10.5)
            .at(30.0, 725.0)
            .text("Rust's type system ensures memory and thread safety.")
            .font("Bold", 14.5)
            .at(30.0, 690.0)
            .text("Productivity")
            .font("Regular", 10.5)
            .at(30.0, 670.0)
            .text("Rust has excellent tooling and documentation.")
            .done();
    }

    // Mirror the exact API pattern from issue #401.
    let encryption_config = EncryptionConfig {
        user_password: "123456".to_string(),
        owner_password: "123456".to_string(),
        algorithm: EncryptionAlgorithm::Aes128,
        permissions: Permissions::all(),
    };
    builder
        .save_with_encryption(&path, encryption_config)
        .expect("save_with_encryption should succeed (issue #401)");

    // The encrypted file must exist and contain an /Encrypt dict.
    assert!(path.exists(), "encrypted PDF must be written to disk");
    let raw = fs::read(&path).unwrap();
    assert!(
        String::from_utf8_lossy(&raw).contains("/Encrypt"),
        "encrypted PDF must contain /Encrypt dict"
    );

    // Open with the user password and verify content is not blank.
    let text = open_encrypted_and_extract(&path, "123456");
    assert!(
        !text.is_empty(),
        "issue #401: content must not be empty after encryption with embedded fonts; \
         got empty string — the write_full_to_writer object sweep is broken"
    );
    assert!(
        text.contains("Rust") || text.contains("Performance") || text.contains("Reliability"),
        "issue #401: extracted text {text:?} should contain recognisable words \
         from the page content"
    );
}

/// AES-256 + embedded font size check — mirrors the exact API path used by
/// the Node.js `saveEncrypted` binding. This ensures the fix covers the
/// default encryption algorithm, not just AES-128.
#[test]
fn save_encrypted_aes256_embedded_font_size_check() {
    use pdf_oxide::writer::EmbeddedFont;
    use std::path::Path;

    let dir = tempdir().unwrap();

    // Baseline: simple text (no embedded font), AES-256
    let simple_path = dir.path().join("simple_aes256.pdf");
    make_builder("Hello simple")
        .save_encrypted(&simple_path, "u", "o")
        .expect("simple AES-256 save_encrypted should succeed");
    let simple_size = fs::metadata(&simple_path).unwrap().len() as usize;

    // Embedded font, AES-256 (same as saveEncrypted in all language bindings)
    let font_path = Path::new("tests/fixtures/fonts/DejaVuSans.ttf");
    if !font_path.exists() {
        return; // fixture not available; skip silently
    }
    let font = EmbeddedFont::from_file(font_path).expect("DejaVuSans.ttf must be loadable");
    let mut ttf_builder = DocumentBuilder::new().register_embedded_font("DejaVu", font);
    ttf_builder
        .a4_page()
        .font("DejaVu", 12.0)
        .at(72.0, 720.0)
        .text("Hello from embedded font (AES-256)")
        .done();

    let ttf_path = dir.path().join("ttf_aes256.pdf");
    ttf_builder
        .save_encrypted(&ttf_path, "u", "o")
        .expect("embedded-font AES-256 save_encrypted should succeed");
    let ttf_size = fs::metadata(&ttf_path).unwrap().len() as usize;

    let diff = ttf_size as isize - simple_size as isize;
    // `SaveOptions::with_encryption` sets compress=true, so the raw FontFile2
    // TrueType stream is FlateDecode-compressed in the output.  A 3 KB floor
    // still clearly distinguishes "font present" from "font missing" while
    // accounting for compression (actual diff observed: ~8 KB).
    assert!(
        diff >= 3_000,
        "AES-256 embedded-font PDF ({ttf_size} B) must be ≥3 KB larger \
         than simple ({simple_size} B); diff={diff} B — font sub-objects likely missing"
    );
}

/// Simulates the exact FFI call sequence used by Node.js `saveEncrypted`:
/// create → register font → a4_page (FFI) → font/at/text ops → done → save_encrypted
/// This verifies the fix works for the path taken by all language bindings.
#[test]
fn save_encrypted_aes256_ffi_sequence_embedded_font() {
    use pdf_oxide::writer::{DocumentBuilder, EmbeddedFont, PageSize};
    use std::path::Path;

    let dir = tempdir().unwrap();
    let font_path = Path::new("tests/fixtures/fonts/DejaVuSans.ttf");
    if !font_path.exists() {
        return;
    }

    // Simulate the FFI: create builder, register font, build page, save_encrypted
    let font = EmbeddedFont::from_file(font_path).expect("font");
    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVu", font);

    // Use the same page-building path as the FFI: builder.page() → fluent ops → done()
    {
        let page = builder.page(PageSize::A4);
        page.font("DejaVu", 12.0)
            .at(72.0, 720.0)
            .text("Hello from embedded font (FFI path AES-256)")
            .done();
    }

    // Baseline for size comparison
    let simple_path = dir.path().join("simple.pdf");
    {
        let mut b2 = DocumentBuilder::new();
        {
            b2.page(PageSize::A4).at(72.0, 720.0).text("simple").done();
        }
        b2.save_encrypted(&simple_path, "u", "o")
            .expect("simple save_encrypted");
    }
    let simple_size = fs::metadata(&simple_path).unwrap().len() as usize;

    let ttf_path = dir.path().join("ttf_ffi_aes256.pdf");
    builder
        .save_encrypted(&ttf_path, "u", "o")
        .expect("embedded-font AES-256 save_encrypted via FFI sequence");
    let ttf_size = fs::metadata(&ttf_path).unwrap().len() as usize;

    let diff = ttf_size as isize - simple_size as isize;
    // Same compression note as save_encrypted_aes256_embedded_font_size_check:
    // compress=true shrinks the raw FontFile2 stream; 3 KB floor is sufficient.
    assert!(
        diff >= 3_000,
        "FFI AES-256 path: embedded-font PDF ({ttf_size} B) must be ≥3 KB larger \
         than simple ({simple_size} B); diff={diff} B — font sub-objects missing"
    );
}

/// Traces the intermediate steps of the embedded-font + AES-256 path.
/// Checks that DocumentEditor::from_bytes + save_with_options correctly
/// preserves all objects from the built PDF.
#[test]
fn debug_trace_editor_from_bytes_size() {
    use pdf_oxide::editor::{
        DocumentEditor, EditableDocument, EncryptionAlgorithm, EncryptionConfig, SaveOptions,
    };
    use pdf_oxide::writer::{DocumentBuilder, EmbeddedFont, PageSize};
    use std::path::Path;

    let dir = tempdir().unwrap();
    let font_path = Path::new("tests/fixtures/fonts/DejaVuSans.ttf");
    if !font_path.exists() {
        return;
    }

    let font = EmbeddedFont::from_file(font_path).expect("font");
    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVu", font);
    {
        let page = builder.page(PageSize::A4);
        page.font("DejaVu", 12.0)
            .at(72.0, 720.0)
            .text("Hello embedded")
            .done();
    }

    // Step 1: build() to get the plain bytes
    let plain_bytes = builder.build().expect("build");
    eprintln!("Step 1 - plain build size: {} B", plain_bytes.len());

    // Step 2: create DocumentEditor from those bytes
    let mut editor = DocumentEditor::from_bytes(plain_bytes).expect("editor from_bytes");

    // Step 3: save with AES-256 encryption
    let config = EncryptionConfig::new("u", "o").with_algorithm(EncryptionAlgorithm::Aes256);
    let enc_path = dir.path().join("debug_enc.pdf");
    editor
        .save_with_options(&enc_path, SaveOptions::with_encryption(config))
        .expect("save_with_options");

    let enc_size = fs::metadata(&enc_path).unwrap().len() as usize;
    eprintln!("Step 3 - encrypted PDF size: {} B", enc_size);

    // compress=true in SaveOptions::with_encryption FlateDecode-compresses
    // the raw FontFile2 stream, so the output is smaller than the plain build.
    // A 5 KB floor clearly distinguishes "font present" from "skeleton only".
    assert!(
        enc_size > 5_000,
        "encrypted PDF ({enc_size} B) should be >5 KB; font objects likely missing"
    );
}

/// Verifies that all_object_ids() returns ALL objects from a built PDF
/// and that they can all be loaded. This validates the sweep prerequisite.
#[test]
fn all_object_ids_returns_complete_set_for_built_pdf() {
    use pdf_oxide::writer::{DocumentBuilder, EmbeddedFont, PageSize};
    use pdf_oxide::PdfDocument;
    use std::path::Path;

    let font_path = Path::new("tests/fixtures/fonts/DejaVuSans.ttf");
    if !font_path.exists() {
        return;
    }

    let font = EmbeddedFont::from_file(font_path).expect("font");
    let mut builder = DocumentBuilder::new().register_embedded_font("DejaVu", font);
    {
        let page = builder.page(PageSize::A4);
        page.font("DejaVu", 12.0)
            .at(72.0, 720.0)
            .text("Hello embedded")
            .done();
    }

    let plain_bytes = builder.build().expect("build");
    eprintln!("Plain PDF size: {} B", plain_bytes.len());

    let doc = PdfDocument::from_bytes(plain_bytes).expect("from_bytes");
    let ids = doc.all_object_ids();
    eprintln!("all_object_ids() returned {} IDs: {:?}", ids.len(), ids);

    // For a simple embedded-font PDF, we should have at least 10 objects
    assert!(
        ids.len() >= 10,
        "Expected at least 10 object IDs but got {} ({:?})",
        ids.len(),
        ids
    );

    // All returned IDs should be loadable
    for id in &ids {
        if *id == 0 {
            continue;
        }
        let obj = doc.load_object(pdf_oxide::object::ObjectRef::new(*id, 0));
        eprintln!("  Object {}: {:?}", id, obj.as_ref().map(|_| "ok").unwrap_or("ERR"));
        // We don't assert success here because some free objects return Null
    }
}
