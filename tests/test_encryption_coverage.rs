//! Integration tests for encryption-related code paths
//!
//! Targets coverage gaps in encryption handling within document.rs

use pdf_oxide::document::PdfDocument;
use pdf_oxide::editor::{
    DocumentEditor, EditableDocument, EncryptionAlgorithm, EncryptionConfig, Permissions,
    SaveOptions,
};

fn build_minimal_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
    );
    let off4 = pdf.len();
    let content = b"BT /F1 12 Tf 72 720 Td (Encrypted test) Tj ET";
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );
    finalize_pdf(&mut pdf, &[0, off1, off2, off3, off4, off5]);
    pdf
}

fn finalize_pdf(pdf: &mut Vec<u8>, obj_offsets: &[usize]) {
    let xref_offset = pdf.len();
    let count = obj_offsets.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \r\n");
    for &off in &obj_offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off).as_bytes());
    }
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        count, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
}

fn write_temp_pdf(data: &[u8], name: &str) -> std::path::PathBuf {
    use std::io::Write;
    let dir = std::env::temp_dir().join("pdf_oxide_encryption_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(data).unwrap();
    path
}

// ===========================================================================
// Tests: EncryptionConfig combinations
// ===========================================================================

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_config_all_algorithms() {
    let algorithms = [
        EncryptionAlgorithm::Rc4_40,
        EncryptionAlgorithm::Rc4_128,
        EncryptionAlgorithm::Aes128,
        EncryptionAlgorithm::Aes256,
    ];

    for alg in &algorithms {
        let config = EncryptionConfig::new("user", "owner").with_algorithm(*alg);
        assert_eq!(config.algorithm, *alg);
    }
}

#[test]
fn test_encryption_config_debug() {
    let config = EncryptionConfig::new("user", "owner");
    let debug = format!("{:?}", config);
    assert!(debug.contains("EncryptionConfig"));
}

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_config_clone() {
    let config = EncryptionConfig::new("user", "owner")
        .with_algorithm(EncryptionAlgorithm::Aes128)
        .with_permissions(Permissions::read_only());
    let cloned = config.clone();
    assert_eq!(cloned.user_password, "user");
    assert_eq!(cloned.algorithm, EncryptionAlgorithm::Aes128);
    assert!(!cloned.permissions.print);
}

// ===========================================================================
// Tests: Permissions bit patterns
// ===========================================================================

#[test]
fn test_permissions_individual_bits() {
    // Test each permission flag individually
    let test_cases = [
        (
            Permissions {
                print: true,
                ..Default::default()
            },
            1 << 2,
        ),
        (
            Permissions {
                modify: true,
                ..Default::default()
            },
            1 << 3,
        ),
        (
            Permissions {
                copy: true,
                ..Default::default()
            },
            1 << 4,
        ),
        (
            Permissions {
                annotate: true,
                ..Default::default()
            },
            1 << 5,
        ),
        (
            Permissions {
                fill_forms: true,
                ..Default::default()
            },
            1 << 8,
        ),
        (
            Permissions {
                accessibility: true,
                ..Default::default()
            },
            1 << 9,
        ),
        (
            Permissions {
                assemble: true,
                ..Default::default()
            },
            1 << 10,
        ),
        (
            Permissions {
                print_high_quality: true,
                ..Default::default()
            },
            1 << 11,
        ),
    ];

    for (perms, expected_bit) in &test_cases {
        let bits = perms.to_bits();
        assert!(bits & expected_bit != 0, "Permission bit {} should be set", expected_bit);
    }
}

#[test]
fn test_permissions_reserved_bits() {
    // Per PDF spec, bits 7-8 and 13-32 must be 1
    let perms = Permissions::default();
    let bits = perms.to_bits();
    // Bits 6-7 (0-indexed) must be set
    assert!(bits & (1 << 6) != 0, "Bit 7 must be set");
    assert!(bits & (1 << 7) != 0, "Bit 8 must be set");
    // High bits (12-31) must be set
    assert!(bits & (1 << 12) != 0, "Bit 13 must be set");
}

#[test]
fn test_permissions_debug() {
    let perms = Permissions::all();
    let debug = format!("{:?}", perms);
    assert!(debug.contains("Permissions"));
    assert!(debug.contains("true"));
}

// ===========================================================================
// Tests: SaveOptions with encryption
// ===========================================================================

#[test]
fn test_save_options_encryption_roundtrip() {
    let config = EncryptionConfig::new("u", "o")
        .with_algorithm(EncryptionAlgorithm::Aes256)
        .with_permissions(Permissions::all());

    let opts = SaveOptions::with_encryption(config);
    let enc = opts.encryption.unwrap();
    assert_eq!(enc.user_password, "u");
    assert_eq!(enc.owner_password, "o");
    assert_eq!(enc.algorithm, EncryptionAlgorithm::Aes256);
    assert!(enc.permissions.print);
}

// ===========================================================================
// Tests: Editor save with encryption config
// ===========================================================================

#[test]
fn test_editor_save_with_encryption_options() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "enc_save_src.pdf");
    let out_path = std::env::temp_dir()
        .join("pdf_oxide_encryption_tests")
        .join("enc_save_out.pdf");

    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_title("Encrypted Doc");

    let config = EncryptionConfig::new("user123", "owner456");
    let opts = SaveOptions::with_encryption(config);
    editor.save_with_options(&out_path, opts).unwrap();

    // The saved file should exist and be non-empty
    let metadata = std::fs::metadata(&out_path).unwrap();
    assert!(metadata.len() > 0);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

// ===========================================================================
// Tests: Unencrypted PDF authentication
// ===========================================================================

#[test]
fn test_authenticate_unencrypted() {
    let pdf = build_minimal_pdf();
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let result = doc.authenticate(b"anypassword");
    // Unencrypted PDF: authentication should succeed or be a no-op
    let _ = result;
}

#[test]
fn test_authenticate_empty_password() {
    let pdf = build_minimal_pdf();
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let result = doc.authenticate(b"");
    let _ = result;
}

// ===========================================================================
// Tests: Algorithm equality and defaults
// ===========================================================================

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_algorithm_equality() {
    assert_eq!(EncryptionAlgorithm::Aes256, EncryptionAlgorithm::Aes256);
    assert_ne!(EncryptionAlgorithm::Aes128, EncryptionAlgorithm::Aes256);
    assert_ne!(EncryptionAlgorithm::Rc4_40, EncryptionAlgorithm::Rc4_128);
}

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_algorithm_copy() {
    let alg = EncryptionAlgorithm::Aes128;
    let copied = alg;
    assert_eq!(alg, copied);
}

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_algorithm_debug() {
    let debug = format!("{:?}", EncryptionAlgorithm::Rc4_40);
    assert!(debug.contains("Rc4_40"));
}
