//! Integration tests for PDF encryption on write.
//!
//! Tests encryption functionality including:
//! - Writing encrypted PDFs with various algorithms (RC4-40, RC4-128, AES-128, AES-256)
//! - Permission flags
//! - Round-trip with empty password (auto-authenticate)
//! - Encrypt dictionary structure validation

use pdf_oxide::editor::{
    DocumentEditor, EditableDocument, EncryptionAlgorithm, EncryptionConfig, Permissions,
    SaveOptions,
};
use pdf_oxide::writer::{DocumentBuilder, DocumentMetadata, PageSize};
use std::fs;
use tempfile::tempdir;

/// Helper to read file contents as string for assertions.
fn read_file_as_string(path: &std::path::Path) -> String {
    let bytes = fs::read(path).unwrap();
    String::from_utf8_lossy(&bytes).to_string()
}

/// Helper to create a simple test PDF with known content.
fn create_test_pdf_with_content(path: &str, content: &str) -> std::io::Result<()> {
    let mut builder = DocumentBuilder::new();
    builder = builder.metadata(
        DocumentMetadata::new()
            .title("Encryption Test Document")
            .author("pdf_oxide"),
    );

    {
        let page = builder.page(PageSize::Letter);
        page.at(72.0, 720.0).text(content).done();
    }

    let pdf_bytes = builder.build().expect("Failed to build test PDF");
    fs::write(path, pdf_bytes)
}

mod encryption_config_tests {
    use super::*;

    #[test]
    fn test_encryption_config_builder() {
        let config = EncryptionConfig::new("user", "owner");

        assert_eq!(config.user_password, "user");
        assert_eq!(config.owner_password, "owner");
        assert_eq!(config.algorithm, EncryptionAlgorithm::Aes256); // Default
    }

    #[test]
    fn test_encryption_config_with_algorithm() {
        let config =
            EncryptionConfig::new("user", "owner").with_algorithm(EncryptionAlgorithm::Aes128);

        assert_eq!(config.algorithm, EncryptionAlgorithm::Aes128);
    }

    #[test]
    fn test_encryption_config_with_permissions() {
        let perms = Permissions::read_only();
        let config = EncryptionConfig::new("user", "owner").with_permissions(perms.clone());

        assert!(!config.permissions.print);
        assert!(!config.permissions.modify);
        assert!(!config.permissions.copy);
        assert!(config.permissions.accessibility); // Always true for compliance
    }

    #[test]
    fn test_permissions_all() {
        let perms = Permissions::all();
        assert!(perms.print);
        assert!(perms.print_high_quality);
        assert!(perms.modify);
        assert!(perms.copy);
        assert!(perms.annotate);
        assert!(perms.fill_forms);
        assert!(perms.accessibility);
        assert!(perms.assemble);
    }

    #[test]
    fn test_permissions_read_only() {
        let perms = Permissions::read_only();
        assert!(!perms.print);
        assert!(!perms.print_high_quality);
        assert!(!perms.modify);
        assert!(!perms.copy);
        assert!(!perms.annotate);
        assert!(!perms.fill_forms);
        assert!(perms.accessibility); // Always true
        assert!(!perms.assemble);
    }

    #[test]
    fn test_permissions_to_bits() {
        let perms = Permissions::all();
        let bits = perms.to_bits();

        // Check that permission bits are set
        assert!(bits & (1 << 2) != 0); // print
        assert!(bits & (1 << 3) != 0); // modify
        assert!(bits & (1 << 4) != 0); // copy
        assert!(bits & (1 << 5) != 0); // annotate
        assert!(bits & (1 << 8) != 0); // fill_forms
        assert!(bits & (1 << 9) != 0); // accessibility
        assert!(bits & (1 << 10) != 0); // assemble
        assert!(bits & (1 << 11) != 0); // print_high_quality
    }

    #[test]
    fn test_permissions_to_bits_read_only() {
        let perms = Permissions::read_only();
        let bits = perms.to_bits();

        // Only accessibility should be set
        assert!(bits & (1 << 2) == 0); // no print
        assert!(bits & (1 << 3) == 0); // no modify
        assert!(bits & (1 << 4) == 0); // no copy
        assert!(bits & (1 << 5) == 0); // no annotate
        assert!(bits & (1 << 9) != 0); // accessibility always on
    }
}

mod save_options_encryption_tests {
    use super::*;

    #[test]
    fn test_save_options_with_encryption() {
        let config = EncryptionConfig::new("user123", "owner456");
        let options = SaveOptions::with_encryption(config);

        assert!(options.encryption.is_some());
        let enc = options.encryption.unwrap();
        assert_eq!(enc.user_password, "user123");
        assert_eq!(enc.owner_password, "owner456");
    }
}

mod aes256_encryption_tests {
    use super::*;

    #[test]
    fn test_encrypt_pdf_aes256() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted.pdf");

        // Create test PDF
        create_test_pdf_with_content(
            input_path.to_str().unwrap(),
            "Secret document content for AES-256 test",
        )
        .unwrap();

        // Open and save with encryption
        let mut editor = DocumentEditor::open(&input_path).unwrap();
        let config = EncryptionConfig::new("userpass", "ownerpass")
            .with_algorithm(EncryptionAlgorithm::Aes256)
            .with_permissions(Permissions::all());

        let options = SaveOptions::with_encryption(config);
        editor.save_with_options(&output_path, options).unwrap();

        // Verify output file exists and is different from input
        assert!(output_path.exists());
        let input_bytes = fs::read(&input_path).unwrap();
        let output_bytes = fs::read(&output_path).unwrap();
        assert_ne!(input_bytes, output_bytes);

        // Verify the encrypted PDF has encryption marker
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/V 5")); // V=5 for AES-256
    }

    #[test]
    fn test_encrypt_pdf_aes256_structure() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let encrypted_path = dir.path().join("encrypted.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "AES-256 structure test")
            .unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();
        let config = EncryptionConfig::new("user256", "owner256")
            .with_algorithm(EncryptionAlgorithm::Aes256);
        editor
            .save_with_options(&encrypted_path, SaveOptions::with_encryption(config))
            .unwrap();

        // Verify encryption dictionary structure
        let output_str = read_file_as_string(&encrypted_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/Filter /Standard"));
        assert!(output_str.contains("/V 5")); // AES-256 uses V=5
        assert!(output_str.contains("/R 6")); // Revision 6
        assert!(output_str.contains("/O ")); // Owner hash
        assert!(output_str.contains("/U ")); // User hash
        assert!(output_str.contains("/P ")); // Permissions
    }

    // Note: Round-trip tests with password authentication require
    // full password verification implementation on the read side.
    // For now, we test that encrypted PDFs are created with correct structure.
}

#[cfg(feature = "legacy-crypto")]
mod aes128_encryption_tests {
    use super::*;

    #[test]
    fn test_encrypt_pdf_aes128() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted_aes128.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "AES-128 test content").unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();
        let config =
            EncryptionConfig::new("user", "owner").with_algorithm(EncryptionAlgorithm::Aes128);

        editor
            .save_with_options(&output_path, SaveOptions::with_encryption(config))
            .unwrap();

        assert!(output_path.exists());
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/V 4")); // V=4 for AES-128
        assert!(output_str.contains("/R 4")); // R=4 for AES-128
    }

    // Note: Round-trip tests with password authentication require
    // full password verification implementation on the read side.
}

#[cfg(feature = "legacy-crypto")]
mod rc4_encryption_tests {
    use super::*;

    #[test]
    fn test_encrypt_pdf_rc4_128() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted_rc4_128.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "RC4-128 test content").unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();
        let config =
            EncryptionConfig::new("user", "owner").with_algorithm(EncryptionAlgorithm::Rc4_128);

        editor
            .save_with_options(&output_path, SaveOptions::with_encryption(config))
            .unwrap();

        assert!(output_path.exists());
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/V 2")); // V=2 for RC4-128
        assert!(output_str.contains("/R 3")); // R=3 for RC4-128
    }

    #[test]
    fn test_encrypt_pdf_rc4_40() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted_rc4_40.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "RC4-40 test content").unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();
        let config =
            EncryptionConfig::new("user", "owner").with_algorithm(EncryptionAlgorithm::Rc4_40);

        editor
            .save_with_options(&output_path, SaveOptions::with_encryption(config))
            .unwrap();

        assert!(output_path.exists());
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/V 1")); // V=1 for RC4-40
        assert!(output_str.contains("/R 2")); // R=2 for RC4-40
    }

    // Note: Round-trip tests with password authentication require
    // full password verification implementation on the read side.
}

mod permission_tests {
    use super::*;

    #[test]
    fn test_encrypt_pdf_read_only_permissions() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("readonly.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "Read-only document").unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();
        let config =
            EncryptionConfig::new("user", "owner").with_permissions(Permissions::read_only());

        editor
            .save_with_options(&output_path, SaveOptions::with_encryption(config))
            .unwrap();

        assert!(output_path.exists());
        // Verify permission bits are in the output
        let output_bytes = fs::read(&output_path).unwrap();
        let output_str = String::from_utf8_lossy(&output_bytes);
        assert!(output_str.contains("/P"));
    }

    #[test]
    fn test_encrypt_pdf_custom_permissions() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("custom_perms.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "Custom permissions test")
            .unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();

        // Allow only printing and copying
        let perms = Permissions {
            print: true,
            print_high_quality: true,
            modify: false,
            copy: true,
            annotate: false,
            fill_forms: false,
            accessibility: true,
            assemble: false,
        };

        let config = EncryptionConfig::new("user", "owner").with_permissions(perms);
        editor
            .save_with_options(&output_path, SaveOptions::with_encryption(config))
            .unwrap();

        assert!(output_path.exists());
    }
}

mod empty_password_tests {
    use super::*;

    #[test]
    fn test_encrypt_with_empty_user_password_creates_valid_pdf() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("owner_only.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "Owner password only test")
            .unwrap();

        let mut editor = DocumentEditor::open(&input_path).unwrap();
        // Empty user password - document opens without prompt but has owner restrictions
        let config =
            EncryptionConfig::new("", "ownerpass").with_permissions(Permissions::read_only());

        editor
            .save_with_options(&output_path, SaveOptions::with_encryption(config))
            .unwrap();

        assert!(output_path.exists());

        // Verify encryption structure
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/O ")); // Owner hash
        assert!(output_str.contains("/U ")); // User hash
        assert!(output_str.contains("/P ")); // Permissions (restricted)
    }
}

mod api_encryption_tests {
    use super::*;
    use pdf_oxide::api::Pdf;

    #[test]
    fn test_pdf_api_save_encrypted() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted_api.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "API encryption test").unwrap();

        let mut pdf = Pdf::open(&input_path).unwrap();
        pdf.save_encrypted(&output_path, "user", "owner").unwrap();

        assert!(output_path.exists());
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
    }

    #[cfg(feature = "legacy-crypto")]
    #[test]
    fn test_pdf_api_save_with_encryption_config() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted_config.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "Config encryption test")
            .unwrap();

        let mut pdf = Pdf::open(&input_path).unwrap();
        let config = EncryptionConfig::new("myuser", "myowner")
            .with_algorithm(EncryptionAlgorithm::Aes128)
            .with_permissions(Permissions::all());

        pdf.save_with_encryption(&output_path, config).unwrap();

        assert!(output_path.exists());
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
        assert!(output_str.contains("/V 4")); // AES-128
    }

    #[test]
    fn test_pdf_api_save_encrypted_with_empty_password() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.pdf");
        let output_path = dir.path().join("encrypted_empty.pdf");

        create_test_pdf_with_content(input_path.to_str().unwrap(), "API empty password test")
            .unwrap();

        // Use empty user password
        let mut pdf = Pdf::open(&input_path).unwrap();
        pdf.save_encrypted(&output_path, "", "owner").unwrap();

        // Verify encryption structure
        assert!(output_path.exists());
        let output_str = read_file_as_string(&output_path);
        assert!(output_str.contains("/Encrypt"));
    }
}
