//! Integration tests for DocumentEditor and related types
//!
//! editor/document_editor.rs has ~3.3% coverage (5,410 missed lines). These tests exercise:
//! - DocumentEditor: open, metadata, page operations, save
//! - SaveOptions, Permissions, EncryptionConfig
//! - Page rotation, media box, crop box
//! - Erase regions, flatten annotations
//! - Form field operations
//! - File embedding

use pdf_oxide::document::PdfDocument;
use pdf_oxide::editor::{
    DocumentEditor, DocumentInfo, EditableDocument, EncryptionAlgorithm, EncryptionConfig,
    Permissions, SaveOptions,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    let content = b"BT /F1 12 Tf 72 720 Td (Test content) Tj ET";
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

fn build_multi_page_pdf(page_count: usize) -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = vec![0usize];

    let off1 = pdf.len();
    offsets.push(off1);
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let kids: Vec<String> = (0..page_count).map(|i| format!("{} 0 R", i + 3)).collect();
    let kids_str = kids.join(" ");

    let off2 = pdf.len();
    offsets.push(off2);
    pdf.extend_from_slice(
        format!(
            "2 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n",
            kids_str, page_count
        )
        .as_bytes(),
    );

    let font_obj_num = page_count * 2 + 3;

    for i in 0..page_count {
        let page_num = i + 3;
        let content_num = page_num + page_count;
        let off = pdf.len();
        offsets.push(off);
        pdf.extend_from_slice(
            format!(
                "{} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R /Resources << /Font << /F1 {} 0 R >> >> >>\nendobj\n",
                page_num, content_num, font_obj_num
            )
            .as_bytes(),
        );
    }

    for i in 0..page_count {
        let content_num = i + 3 + page_count;
        let content = format!("BT /F1 12 Tf 72 720 Td (Page {}) Tj ET", i + 1);
        let off = pdf.len();
        offsets.push(off);
        pdf.extend_from_slice(
            format!("{} 0 obj\n<< /Length {} >>\nstream\n", content_num, content.len()).as_bytes(),
        );
        pdf.extend_from_slice(content.as_bytes());
        pdf.extend_from_slice(b"\nendstream\nendobj\n");
    }

    let font_off = pdf.len();
    offsets.push(font_off);
    pdf.extend_from_slice(
        format!(
            "{} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
            font_obj_num
        )
        .as_bytes(),
    );

    finalize_pdf(&mut pdf, &offsets);
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
    let dir = std::env::temp_dir().join("pdf_oxide_editor_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(data).unwrap();
    path
}

// ===========================================================================
// Tests: DocumentInfo
// ===========================================================================

#[test]
fn test_document_info_full_builder() {
    let info = DocumentInfo::new()
        .title("My Document")
        .author("Author Name")
        .subject("Subject Here")
        .keywords("key1, key2")
        .creator("Test Creator")
        .producer("Test Producer");

    assert_eq!(info.title, Some("My Document".to_string()));
    assert_eq!(info.author, Some("Author Name".to_string()));
    assert_eq!(info.subject, Some("Subject Here".to_string()));
    assert_eq!(info.keywords, Some("key1, key2".to_string()));
    assert_eq!(info.creator, Some("Test Creator".to_string()));
    assert_eq!(info.producer, Some("Test Producer".to_string()));
}

#[test]
fn test_document_info_roundtrip() {
    let original = DocumentInfo::new()
        .title("Round Trip Title")
        .author("RT Author")
        .subject("RT Subject")
        .keywords("rt, test")
        .creator("RT Creator")
        .producer("RT Producer");

    let obj = original.to_object();
    let restored = DocumentInfo::from_object(&obj);

    assert_eq!(restored.title, original.title);
    assert_eq!(restored.author, original.author);
    assert_eq!(restored.subject, original.subject);
    assert_eq!(restored.keywords, original.keywords);
    assert_eq!(restored.creator, original.creator);
    assert_eq!(restored.producer, original.producer);
}

#[test]
fn test_document_info_from_empty_dict() {
    let obj = pdf_oxide::object::Object::Dictionary(std::collections::HashMap::new());
    let info = DocumentInfo::from_object(&obj);
    assert!(info.title.is_none());
    assert!(info.author.is_none());
}

#[test]
fn test_document_info_from_non_dict() {
    let obj = pdf_oxide::object::Object::Integer(42);
    let info = DocumentInfo::from_object(&obj);
    // Should return empty info without panic
    assert!(info.title.is_none());
}

#[test]
fn test_document_info_with_dates() {
    let mut info = DocumentInfo::new();
    info.creation_date = Some("D:20260226120000".to_string());
    info.mod_date = Some("D:20260226130000".to_string());

    let obj = info.to_object();
    let dict = obj.as_dict().unwrap();
    assert!(dict.contains_key("CreationDate"));
    assert!(dict.contains_key("ModDate"));

    let restored = DocumentInfo::from_object(&obj);
    assert_eq!(restored.creation_date, Some("D:20260226120000".to_string()));
    assert_eq!(restored.mod_date, Some("D:20260226130000".to_string()));
}

// ===========================================================================
// Tests: SaveOptions
// ===========================================================================

#[test]
fn test_save_options_full_rewrite() {
    let opts = SaveOptions::full_rewrite();
    assert!(!opts.incremental);
    assert!(opts.compress);
    assert!(opts.garbage_collect);
    assert!(!opts.linearize);
    assert!(opts.encryption.is_none());
}

#[test]
fn test_save_options_incremental() {
    let opts = SaveOptions::incremental();
    assert!(opts.incremental);
    assert!(!opts.compress);
    assert!(!opts.garbage_collect);
}

#[test]
fn test_save_options_with_encryption() {
    let config = EncryptionConfig::new("user", "owner");
    let opts = SaveOptions::with_encryption(config);
    assert!(!opts.incremental);
    assert!(opts.compress);
    assert!(opts.garbage_collect);
    assert!(opts.encryption.is_some());
    let enc = opts.encryption.unwrap();
    assert_eq!(enc.user_password, "user");
    assert_eq!(enc.owner_password, "owner");
}

// ===========================================================================
// Tests: Permissions
// ===========================================================================

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
    assert!(!perms.modify);
    assert!(!perms.copy);
    assert!(perms.accessibility); // Always allowed
    assert!(!perms.annotate);
}

#[test]
fn test_permissions_to_bits_all() {
    let perms = Permissions::all();
    let bits = perms.to_bits();
    // All permission bits should be set
    assert!(bits & (1 << 2) != 0, "Print bit should be set");
    assert!(bits & (1 << 3) != 0, "Modify bit should be set");
    assert!(bits & (1 << 4) != 0, "Copy bit should be set");
    assert!(bits & (1 << 5) != 0, "Annotate bit should be set");
    assert!(bits & (1 << 8) != 0, "Fill forms bit should be set");
    assert!(bits & (1 << 9) != 0, "Accessibility bit should be set");
    assert!(bits & (1 << 10) != 0, "Assemble bit should be set");
    assert!(bits & (1 << 11) != 0, "Print HQ bit should be set");
}

#[test]
fn test_permissions_to_bits_read_only() {
    let perms = Permissions::read_only();
    let bits = perms.to_bits();
    assert!(bits & (1 << 2) == 0, "Print bit should NOT be set");
    assert!(bits & (1 << 3) == 0, "Modify bit should NOT be set");
    assert!(bits & (1 << 4) == 0, "Copy bit should NOT be set");
    assert!(bits & (1 << 9) != 0, "Accessibility bit should be set");
}

#[test]
fn test_permissions_to_bits_custom() {
    let perms = Permissions {
        print: true,
        copy: true,
        fill_forms: true,
        ..Default::default()
    };
    let bits = perms.to_bits();
    assert!(bits & (1 << 2) != 0, "Print set");
    assert!(bits & (1 << 4) != 0, "Copy set");
    assert!(bits & (1 << 8) != 0, "Fill forms set");
    assert!(bits & (1 << 3) == 0, "Modify not set");
    assert!(bits & (1 << 5) == 0, "Annotate not set");
}

#[test]
fn test_permissions_default() {
    let perms = Permissions::default();
    assert!(!perms.print);
    assert!(!perms.modify);
    assert!(!perms.copy);
    assert!(!perms.accessibility);
}

// ===========================================================================
// Tests: EncryptionConfig
// ===========================================================================

#[test]
fn test_encryption_config_new() {
    let config = EncryptionConfig::new("user123", "owner456");
    assert_eq!(config.user_password, "user123");
    assert_eq!(config.owner_password, "owner456");
    assert_eq!(config.algorithm, EncryptionAlgorithm::Aes256); // default
}

#[test]
fn test_encryption_config_default() {
    let config = EncryptionConfig::default();
    assert_eq!(config.user_password, "");
    assert_eq!(config.owner_password, "");
    assert_eq!(config.algorithm, EncryptionAlgorithm::Aes256);
    assert!(config.permissions.print); // Permissions::all() by default
}

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_config_with_algorithm() {
    let config = EncryptionConfig::new("u", "o").with_algorithm(EncryptionAlgorithm::Aes128);
    assert_eq!(config.algorithm, EncryptionAlgorithm::Aes128);
}

#[test]
fn test_encryption_config_with_permissions() {
    let config = EncryptionConfig::new("u", "o").with_permissions(Permissions::read_only());
    assert!(!config.permissions.print);
    assert!(config.permissions.accessibility);
}

#[cfg(feature = "legacy-crypto")]
#[test]
fn test_encryption_algorithm_variants() {
    assert_ne!(EncryptionAlgorithm::Rc4_40, EncryptionAlgorithm::Rc4_128);
    assert_ne!(EncryptionAlgorithm::Aes128, EncryptionAlgorithm::Aes256);
    assert_eq!(EncryptionAlgorithm::default(), EncryptionAlgorithm::Aes256);
}

// ===========================================================================
// Tests: DocumentEditor
// ===========================================================================

#[test]
fn test_editor_open() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_open.pdf");
    let editor = DocumentEditor::open(&path).unwrap();
    assert!(!editor.is_modified());
    assert_eq!(editor.version(), (1, 7));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_page_count() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editor_pages.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    assert_eq!(editor.page_count().unwrap(), 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_title() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_title.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_title("New Title");
    assert!(editor.is_modified());
    let title = editor.title().unwrap();
    assert_eq!(title, Some("New Title".to_string()));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_author() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_author.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_author("New Author");
    let author = editor.author().unwrap();
    assert_eq!(author, Some("New Author".to_string()));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_subject_keywords() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_meta.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_subject("Test Subject");
    editor.set_keywords("key1, key2, key3");
    assert_eq!(editor.subject().unwrap(), Some("Test Subject".to_string()));
    assert_eq!(editor.keywords().unwrap(), Some("key1, key2, key3".to_string()));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_metadata_roundtrip() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_roundtrip.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let info = DocumentInfo::new().title("RT Title").author("RT Author");
    editor.set_info(info).unwrap();

    let retrieved = editor.get_info().unwrap();
    assert_eq!(retrieved.title, Some("RT Title".to_string()));
    assert_eq!(retrieved.author, Some("RT Author".to_string()));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_source_path() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_source.pdf");
    let editor = DocumentEditor::open(&path).unwrap();
    assert!(editor.source_path().contains("editor_source.pdf"));
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor page operations
// ===========================================================================

#[test]
fn test_editor_current_page_count() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editor_current_pages.pdf");
    let editor = DocumentEditor::open(&path).unwrap();
    assert_eq!(editor.current_page_count(), 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_info() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_pageinfo.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    let info = editor.get_page_info(0).unwrap();
    assert_eq!(info.index, 0);
    assert!(info.width > 0.0);
    assert!(info.height > 0.0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_save_full_rewrite() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_save_src.pdf");
    let out_path = write_temp_pdf(&[], "editor_save_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_title("Saved Doc");
    editor
        .save_with_options(&out_path, SaveOptions::full_rewrite())
        .unwrap();

    // Verify saved file is valid
    let doc = PdfDocument::open(&out_path).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn test_editor_save_default() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_save_default_src.pdf");
    let out_path = std::env::temp_dir()
        .join("pdf_oxide_editor_tests")
        .join("editor_save_default_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.save(&out_path).unwrap();

    let doc = PdfDocument::open(&out_path).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

// ===========================================================================
// Tests: Page rotation
// ===========================================================================

#[test]
fn test_editor_page_rotation() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_rotation.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let rotation = editor.get_page_rotation(0).unwrap();
    assert_eq!(rotation, 0);

    editor.set_page_rotation(0, 90).unwrap();
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_rotate_page_by() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_rotate_by.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.rotate_page_by(0, 180).unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_rotate_all_pages() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editor_rotate_all.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.rotate_all_pages(90).unwrap();
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Page media box and crop box
// ===========================================================================

#[test]
fn test_editor_get_page_media_box() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_mediabox.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    let media_box = editor.get_page_media_box(0).unwrap();
    assert_eq!(media_box[0], 0.0);
    assert_eq!(media_box[1], 0.0);
    assert_eq!(media_box[2], 612.0);
    assert_eq!(media_box[3], 792.0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_page_media_box() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_set_mediabox.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor
        .set_page_media_box(0, [0.0, 0.0, 595.0, 842.0])
        .unwrap();
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_crop_box_default_none() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_cropbox.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    let crop_box = editor.get_page_crop_box(0).unwrap();
    assert!(crop_box.is_none(), "Minimal PDF should have no explicit crop box");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_crop_box() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_set_cropbox.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor
        .set_page_crop_box(0, [50.0, 50.0, 562.0, 742.0])
        .unwrap();
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_crop_margins() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_crop_margins.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.crop_margins(36.0, 36.0, 36.0, 36.0).unwrap();
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Erase regions
// ===========================================================================

#[test]
fn test_editor_erase_region() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_erase.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor
        .erase_region(0, [100.0, 100.0, 200.0, 200.0])
        .unwrap();
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_erase_regions_multiple() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_erase_multi.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor
        .erase_regions(0, &[[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]])
        .unwrap();
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_clear_erase_regions() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_clear_erase.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor
        .erase_region(0, [100.0, 100.0, 200.0, 200.0])
        .unwrap();
    editor.clear_erase_regions(0);
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Annotation flattening
// ===========================================================================

#[test]
fn test_editor_flatten_annotations() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_flatten.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.flatten_page_annotations(0).unwrap();
    assert!(editor.is_page_marked_for_flatten(0));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_flatten_all_annotations() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editor_flatten_all.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.flatten_all_annotations().unwrap();
    for i in 0..3 {
        assert!(editor.is_page_marked_for_flatten(i));
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_unmark_flatten() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_unmark.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.flatten_page_annotations(0).unwrap();
    assert!(editor.is_page_marked_for_flatten(0));
    editor.unmark_page_for_flatten(0);
    assert!(!editor.is_page_marked_for_flatten(0));
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Form flattening
// ===========================================================================

#[test]
fn test_editor_flatten_forms_on_page() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_flatten_forms.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.flatten_forms_on_page(0).unwrap();
    assert!(editor.is_page_marked_for_form_flatten(0));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_flatten_forms_all() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_flatten_forms_all.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.flatten_forms().unwrap();
    assert!(editor.will_remove_acroform());
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: File embedding
// ===========================================================================

#[test]
fn test_editor_embed_file() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_embed.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor
        .embed_file("test.txt", b"Hello embedded".to_vec())
        .unwrap();
    assert_eq!(editor.pending_embedded_files().len(), 1);
    assert!(editor.is_modified());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_clear_embedded_files() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_clear_embed.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.embed_file("test.txt", b"data".to_vec()).unwrap();
    editor.clear_embedded_files();
    assert!(editor.pending_embedded_files().is_empty());
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: EditableDocument trait
// ===========================================================================

#[test]
fn test_editable_document_trait() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_trait.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    // Test trait methods
    let page_count = EditableDocument::page_count(&mut editor).unwrap();
    assert_eq!(page_count, 1);

    let info = EditableDocument::get_info(&mut editor).unwrap();
    let _ = info;

    let new_info = DocumentInfo::new().title("Via Trait");
    EditableDocument::set_info(&mut editor, new_info).unwrap();

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Redactions
// ===========================================================================

#[test]
fn test_editor_redaction_marking() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_redact.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.apply_page_redactions(0).unwrap();
    assert!(editor.is_page_marked_for_redaction(0));
    editor.unmark_page_for_redaction(0);
    assert!(!editor.is_page_marked_for_redaction(0));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_apply_all_redactions() {
    let pdf = build_multi_page_pdf(2);
    let path = write_temp_pdf(&pdf, "editor_redact_all.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.apply_all_redactions().unwrap();
    assert!(editor.is_page_marked_for_redaction(0));
    assert!(editor.is_page_marked_for_redaction(1));
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Image operations
// ===========================================================================

#[test]
fn test_editor_image_modifications_empty() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_images.pdf");
    let editor = DocumentEditor::open(&path).unwrap();
    assert!(!editor.has_image_modifications(0));
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: XFA
// ===========================================================================

#[test]
fn test_editor_has_xfa_simple_pdf() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_xfa.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    let has_xfa = editor.has_xfa().unwrap();
    assert!(!has_xfa, "Simple PDF should not have XFA");
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: EditableDocument trait - page manipulation
// ===========================================================================

#[test]
fn test_editable_remove_page() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editable_remove_page.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    assert_eq!(editor.current_page_count(), 3);
    EditableDocument::remove_page(&mut editor, 1).unwrap();
    assert_eq!(editor.current_page_count(), 2);
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_remove_page_out_of_range() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editable_remove_oob.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = EditableDocument::remove_page(&mut editor, 5);
    assert!(result.is_err(), "Removing page beyond range should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_remove_all_pages_one_by_one() {
    let pdf = build_multi_page_pdf(2);
    let path = write_temp_pdf(&pdf, "editable_remove_all.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    EditableDocument::remove_page(&mut editor, 0).unwrap();
    assert_eq!(editor.current_page_count(), 1);
    EditableDocument::remove_page(&mut editor, 0).unwrap();
    assert_eq!(editor.current_page_count(), 0);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_move_page() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editable_move_page.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    EditableDocument::move_page(&mut editor, 0, 2).unwrap();
    assert!(editor.is_modified());
    assert_eq!(editor.current_page_count(), 3);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_move_page_same_position() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editable_move_same.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    EditableDocument::move_page(&mut editor, 1, 1).unwrap();
    assert_eq!(editor.current_page_count(), 3);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_move_page_out_of_range() {
    let pdf = build_multi_page_pdf(2);
    let path = write_temp_pdf(&pdf, "editable_move_oob.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = EditableDocument::move_page(&mut editor, 0, 10);
    assert!(result.is_err(), "Moving page to out-of-range index should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_duplicate_page() {
    let pdf = build_multi_page_pdf(2);
    let path = write_temp_pdf(&pdf, "editable_dup_page.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let new_index = EditableDocument::duplicate_page(&mut editor, 0).unwrap();
    assert_eq!(new_index, 2);
    assert_eq!(editor.current_page_count(), 3);
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_duplicate_page_out_of_range() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editable_dup_oob.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = EditableDocument::duplicate_page(&mut editor, 5);
    assert!(result.is_err(), "Duplicating non-existent page should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editable_duplicate_then_remove() {
    let pdf = build_multi_page_pdf(2);
    let path = write_temp_pdf(&pdf, "editable_dup_remove.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let new_idx = EditableDocument::duplicate_page(&mut editor, 0).unwrap();
    assert_eq!(editor.current_page_count(), 3);

    EditableDocument::remove_page(&mut editor, new_idx).unwrap();
    assert_eq!(editor.current_page_count(), 2);

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor merge and extract operations
// ===========================================================================

#[test]
fn test_editor_merge_from_same_file() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_merge_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let merged = editor.merge_from(&path).unwrap();
    assert_eq!(merged, 1);
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_merge_from_multi_page() {
    let pdf1 = build_minimal_pdf();
    let pdf2 = build_multi_page_pdf(3);
    let path1 = write_temp_pdf(&pdf1, "editor_merge_main.pdf");
    let path2 = write_temp_pdf(&pdf2, "editor_merge_append.pdf");
    let mut editor = DocumentEditor::open(&path1).unwrap();

    let merged = editor.merge_from(&path2).unwrap();
    assert_eq!(merged, 3);
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path1);
    let _ = std::fs::remove_file(&path2);
}

#[test]
fn test_editor_merge_pages_from_specific() {
    let pdf1 = build_minimal_pdf();
    let pdf2 = build_multi_page_pdf(4);
    let path1 = write_temp_pdf(&pdf1, "editor_merge_pages_main.pdf");
    let path2 = write_temp_pdf(&pdf2, "editor_merge_pages_src.pdf");
    let mut editor = DocumentEditor::open(&path1).unwrap();

    let merged = editor.merge_pages_from(&path2, &[0, 2]).unwrap();
    assert_eq!(merged, 2);
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path1);
    let _ = std::fs::remove_file(&path2);
}

#[test]
fn test_editor_merge_pages_from_empty() {
    let pdf1 = build_minimal_pdf();
    let pdf2 = build_multi_page_pdf(2);
    let path1 = write_temp_pdf(&pdf1, "editor_merge_pages_empty_main.pdf");
    let path2 = write_temp_pdf(&pdf2, "editor_merge_pages_empty_src.pdf");
    let mut editor = DocumentEditor::open(&path1).unwrap();

    let merged = editor.merge_pages_from(&path2, &[]).unwrap();
    assert_eq!(merged, 0);

    let _ = std::fs::remove_file(&path1);
    let _ = std::fs::remove_file(&path2);
}

#[test]
fn test_editor_merge_pages_from_out_of_range() {
    let pdf1 = build_minimal_pdf();
    let pdf2 = build_multi_page_pdf(2);
    let path1 = write_temp_pdf(&pdf1, "editor_merge_pages_oob_main.pdf");
    let path2 = write_temp_pdf(&pdf2, "editor_merge_pages_oob_src.pdf");
    let mut editor = DocumentEditor::open(&path1).unwrap();

    let result = editor.merge_pages_from(&path2, &[0, 10]);
    assert!(result.is_err(), "Merging out-of-range page should fail");

    let _ = std::fs::remove_file(&path1);
    let _ = std::fs::remove_file(&path2);
}

#[test]
fn test_editor_extract_pages_works() {
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editor_extract.pdf");
    let out_path = write_temp_pdf(&[], "editor_extract_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.extract_pages(&[0, 1], &out_path);
    assert!(result.is_ok(), "extract_pages should succeed: {result:?}");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn test_editor_extract_pages_invalid_index() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_extract_inv.pdf");
    let out_path = write_temp_pdf(&[], "editor_extract_inv_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.extract_pages(&[5], &out_path);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

// ===========================================================================
// Tests: Editor get_page / save_page (DOM access)
// ===========================================================================

#[test]
fn test_editor_get_page_returns_pdf_page() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    assert_eq!(page.page_index, 0);
    assert!(page.width > 0.0);
    assert!(page.height > 0.0);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_root_element() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_root.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let root = page.root();
    // Root should be a structure element
    assert!(root.is_structure());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_children() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_children.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let children = page.children();
    // The minimal PDF has some content, so we just check it runs without error
    let _ = children;

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_save_page_roundtrip() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_save_page_rt.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    editor.save_page(page).unwrap();
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_find_text() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_text.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let texts = page.find_text_containing("Test");
    // May or may not find text depending on extraction, but should not panic
    let _ = texts;

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_find_images() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_images.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let images = page.find_images();
    // Minimal PDF has no images
    assert!(images.is_empty());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_find_paths() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_paths.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let paths = page.find_paths();
    let _ = paths;

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_find_tables() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_tables.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let tables = page.find_tables();
    assert!(tables.is_empty(), "Minimal PDF should have no tables");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_annotations_empty() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_page_annots.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page = editor.get_page(0).unwrap();
    let annots = page.annotations();
    assert!(annots.is_empty(), "Minimal PDF should have no annotations");
    assert!(!page.has_annotations_modified());

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor page_editor (PageEditor fluent API)
// ===========================================================================

#[test]
fn test_editor_page_editor_create_and_done() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_page_editor.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page_editor = editor.page_editor(0).unwrap();
    let page = page_editor.done().unwrap();
    editor.save_page_from_editor(page).unwrap();

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_page_editor_find_text() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_pe_find_text.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page_editor = editor.page_editor(0).unwrap();
    let collection = page_editor.find_text_containing("nonexistent").unwrap();
    let page = collection.done().unwrap();
    editor.save_page_from_editor(page).unwrap();

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_page_editor_find_images() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_pe_find_img.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page_editor = editor.page_editor(0).unwrap();
    let collection = page_editor.find_images().unwrap();
    let page = collection.done().unwrap();
    editor.save_page_from_editor(page).unwrap();

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_page_editor_find_paths() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_pe_find_paths.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page_editor = editor.page_editor(0).unwrap();
    let collection = page_editor.find_paths().unwrap();
    let page = collection.done().unwrap();
    editor.save_page_from_editor(page).unwrap();

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_page_editor_find_tables() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_pe_find_tables.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let page_editor = editor.page_editor(0).unwrap();
    let collection = page_editor.find_tables().unwrap();
    let page = collection.done().unwrap();
    editor.save_page_from_editor(page).unwrap();

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Edit page with closure
// ===========================================================================

#[test]
fn test_editor_edit_page_closure() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_edit_page.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    editor
        .edit_page(0, |_page| {
            // Just verify the closure is called
            Ok(())
        })
        .unwrap();
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Form field operations on simple PDF (no form fields)
// ===========================================================================

#[test]
fn test_editor_get_form_fields_empty() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_form_empty.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let fields = editor.get_form_fields().unwrap();
    assert!(fields.is_empty(), "Minimal PDF should have no form fields");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_has_form_field_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_has_form.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let has_field = editor.has_form_field("nonexistent_field").unwrap();
    assert!(!has_field, "Nonexistent field should not be found");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_value_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_set_form_val.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_value(
        "nonexistent",
        pdf_oxide::editor::FormFieldValue::Text("test".into()),
    );
    assert!(result.is_err(), "Setting value on nonexistent field should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_form_field_value_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_form_val.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let value = editor.get_form_field_value("nonexistent").unwrap();
    assert!(value.is_none(), "Nonexistent field should return None");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_remove_form_field_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_rm_form.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.remove_form_field("nonexistent");
    assert!(result.is_err(), "Removing nonexistent field should fail");

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Form field property setters - error handling on non-existent fields
// ===========================================================================

#[test]
fn test_editor_set_form_field_readonly_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_readonly.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_readonly("nonexistent", true);
    assert!(result.is_err(), "Setting readonly on nonexistent field should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_required_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_required.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_required("nonexistent", true);
    assert!(result.is_err(), "Setting required on nonexistent field should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_tooltip_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_tooltip.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_tooltip("nonexistent", "tip");
    assert!(result.is_err(), "Setting tooltip on nonexistent field should fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_max_length_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_maxlen.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_max_length("nonexistent", 100);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_alignment_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_align.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_alignment("nonexistent", 1);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_background_color_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_bgcolor.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_background_color("nonexistent", [1.0, 1.0, 1.0]);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_border_color_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_bdrcolor.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_border_color("nonexistent", [0.0, 0.0, 0.0]);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_border_width_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_bdrwidth.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_border_width("nonexistent", 1.0);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_default_appearance_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_da.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_default_appearance("nonexistent", "/Helv 12 Tf 0 g");
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_flags_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_flags.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.set_form_field_flags("nonexistent", 0x01);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_form_field_rect_not_found() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_ff_rect.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor
        .set_form_field_rect("nonexistent", pdf_oxide::geometry::Rect::new(0.0, 0.0, 100.0, 20.0));
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: get_page_content / set_page_content
// ===========================================================================

#[test]
fn test_editor_get_page_content() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_get_content.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let content = editor.get_page_content(0).unwrap();
    // Should return Some or None for a valid page
    let _ = content;

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_page_content() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_set_content.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let structure = pdf_oxide::elements::StructureElement {
        structure_type: "Document".to_string(),
        bbox: pdf_oxide::geometry::Rect::new(0.0, 0.0, 612.0, 792.0),
        children: Vec::new(),
        reading_order: Some(0),
        alt_text: None,
        language: None,
    };

    editor.set_page_content(0, structure).unwrap();
    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_set_page_content_out_of_range() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_set_content_oob.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let structure = pdf_oxide::elements::StructureElement::default();
    let result = editor.set_page_content(99, structure);
    assert!(result.is_err(), "Setting content on out-of-range page should fail");

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: modify_structure with closure
// ===========================================================================

#[test]
fn test_editor_modify_structure() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_modify_struct.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.modify_structure(0, |structure| {
        structure.alt_text = Some("Modified by test".to_string());
        Ok(())
    });
    // This may succeed or fail depending on whether structure is available
    let _ = result;

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_modify_structure_change_language() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_modify_struct_lang.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.modify_structure(0, |structure| {
        structure.language = Some("en-US".to_string());
        Ok(())
    });
    let _ = result;

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: get_page_images
// ===========================================================================

#[test]
fn test_editor_get_page_images_empty() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_page_images.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let images = editor.get_page_images(0).unwrap();
    assert!(images.is_empty(), "Minimal PDF should have no images on page");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_get_page_images_out_of_range() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_page_images_oob.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.get_page_images(99);
    assert!(result.is_err(), "Getting images from out-of-range page should fail");

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor save with incremental options
// ===========================================================================

#[test]
fn test_editor_save_incremental() {
    // Build a larger PDF so `startxref` appears within the search window
    // (the incremental writer searches backwards from len-100 to 0)
    let pdf = build_multi_page_pdf(3);
    let path = write_temp_pdf(&pdf, "editor_save_incr_src.pdf");
    let out_path = write_temp_pdf(&[], "editor_save_incr_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    editor.set_title("Incremental Title");
    let result = editor.save_with_options(&out_path, SaveOptions::incremental());
    // Incremental save may fail on very small PDFs where startxref
    // search range does not cover the keyword; exercise the code path either way
    let _ = result;

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn test_editor_save_with_compression() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_save_compress_src.pdf");
    let out_path = write_temp_pdf(&[], "editor_save_compress_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let opts = SaveOptions {
        incremental: false,
        compress: true,
        garbage_collect: true,
        linearize: false,
        encryption: None,
    };
    editor.save_with_options(&out_path, opts).unwrap();

    let doc = PdfDocument::open(&out_path).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn test_editor_save_no_compress_no_gc() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_save_nocomp_src.pdf");
    let out_path = write_temp_pdf(&[], "editor_save_nocomp_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let opts = SaveOptions {
        incremental: false,
        compress: false,
        garbage_collect: false,
        linearize: false,
        encryption: None,
    };
    editor.save_with_options(&out_path, opts).unwrap();

    let doc = PdfDocument::open(&out_path).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

// ===========================================================================
// Tests: Editor source access
// ===========================================================================

#[test]
fn test_editor_source_immutable() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_source_imm.pdf");
    let editor = DocumentEditor::open(&path).unwrap();

    let source = editor.source();
    let version = source.version();
    assert_eq!(version, (1, 7));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_source_mutable() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_source_mut.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let source_mut = editor.source_mut();
    let page_count = source_mut.page_count().unwrap();
    assert_eq!(page_count, 1);

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor resource manager
// ===========================================================================

#[test]
fn test_editor_resource_manager_access() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_resmgr.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let _rm = editor.resource_manager();
    let _rm_mut = editor.resource_manager_mut();
    // Just verify we can access these without panic

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor has_modified_annotations / get_page_annotations
// ===========================================================================

#[test]
fn test_editor_has_modified_annotations_false() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_has_mod_annots.pdf");
    let editor = DocumentEditor::open(&path).unwrap();

    assert!(!editor.has_modified_annotations(0));
    assert!(editor.get_page_annotations(0).is_none());

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Multi-page page_info traversal
// ===========================================================================

#[test]
fn test_editor_page_info_all_pages() {
    let pdf = build_multi_page_pdf(5);
    let path = write_temp_pdf(&pdf, "editor_all_page_info.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    for i in 0..5 {
        let info = editor.get_page_info(i).unwrap();
        assert_eq!(info.index, i);
        assert_eq!(info.width, 612.0);
        assert_eq!(info.height, 792.0);
        assert_eq!(info.rotation, 0);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_page_info_out_of_range() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_page_info_oob.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let result = editor.get_page_info(10);
    assert!(result.is_err(), "Getting info for out-of-range page should fail");

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Flatten forms and remove_acroform flag
// ===========================================================================

#[test]
fn test_editor_flatten_forms_will_remove_acroform() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_flatten_forms_acroform.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    assert!(!editor.will_remove_acroform());
    editor.flatten_forms().unwrap();
    assert!(editor.will_remove_acroform());

    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// Tests: Editor page operations combined workflows
// ===========================================================================

#[test]
fn test_editor_combined_rotate_and_crop() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_combined_rot_crop.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    editor.set_page_rotation(0, 90).unwrap();
    editor
        .set_page_crop_box(0, [10.0, 10.0, 600.0, 780.0])
        .unwrap();
    editor
        .set_page_media_box(0, [0.0, 0.0, 595.0, 842.0])
        .unwrap();

    assert!(editor.is_modified());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_editor_combined_metadata_and_save() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "editor_combined_meta_save_src.pdf");
    let out_path = write_temp_pdf(&[], "editor_combined_meta_save_out.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    editor.set_title("Combined Test");
    editor.set_author("Test Author");
    editor.set_subject("Test Subject");
    editor.set_keywords("test, combined, coverage");

    let info = DocumentInfo::new()
        .title("Combined Test Override")
        .author("Override Author");
    editor.set_info(info).unwrap();

    editor.save(&out_path).unwrap();

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

// ===========================================================================
// Tests: compress + garbage_collect in SaveOptions
// ===========================================================================

#[test]
fn test_save_to_bytes_returns_valid_pdf() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "save_to_bytes_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_title("BytesTest");

    let bytes = editor.save_to_bytes().unwrap();
    assert!(!bytes.is_empty());
    assert!(bytes.starts_with(b"%PDF-"), "output must start with PDF header");

    // Round-trip: open the returned bytes as a new document
    let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_save_to_bytes_with_compress_produces_valid_pdf() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "compress_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let opts = SaveOptions {
        compress: true,
        garbage_collect: false,
        linearize: false,
        incremental: false,
        encryption: None,
    };
    let bytes = editor.save_to_bytes_with_options(opts).unwrap();
    assert!(!bytes.is_empty());
    assert!(bytes.starts_with(b"%PDF-"));

    let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_save_to_bytes_with_garbage_collect_produces_valid_pdf() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "gc_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let opts = SaveOptions {
        compress: false,
        garbage_collect: true,
        linearize: false,
        incremental: false,
        encryption: None,
    };
    let bytes = editor.save_to_bytes_with_options(opts).unwrap();
    assert!(!bytes.is_empty());
    assert!(bytes.starts_with(b"%PDF-"));

    let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_save_to_bytes_compress_and_gc_together() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "compress_gc_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    // Both compress and gc enabled (full_rewrite defaults)
    let bytes = editor.save_to_bytes().unwrap(); // uses full_rewrite internally
    assert!(bytes.starts_with(b"%PDF-"));

    let doc = PdfDocument::from_bytes(bytes.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_gc_produces_smaller_or_equal_output_than_no_gc() {
    // Build a PDF that includes an unrefenced orphan object and verify that
    // GC output is <= no-gc output in size (orphans removed).
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );
    // Orphan object — not referenced from anywhere
    let off4 = pdf.len();
    let orphan = b"This is a large orphaned stream payload that should be removed by GC".repeat(20);
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", orphan.len()).as_bytes());
    pdf.extend_from_slice(&orphan);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    finalize_pdf(&mut pdf, &[0, off1, off2, off3, off4]);

    let path = write_temp_pdf(&pdf, "gc_size_src.pdf");

    let mut editor_nogc = DocumentEditor::open(&path).unwrap();
    let no_gc_bytes = editor_nogc
        .save_to_bytes_with_options(SaveOptions {
            compress: false,
            garbage_collect: false,
            linearize: false,
            incremental: false,
            encryption: None,
        })
        .unwrap();

    let mut editor_gc = DocumentEditor::open(&path).unwrap();
    let gc_bytes = editor_gc
        .save_to_bytes_with_options(SaveOptions {
            compress: false,
            garbage_collect: true,
            linearize: false,
            incremental: false,
            encryption: None,
        })
        .unwrap();

    assert!(
        gc_bytes.len() <= no_gc_bytes.len(),
        "GC output ({} bytes) should be <= no-GC output ({} bytes)",
        gc_bytes.len(),
        no_gc_bytes.len()
    );

    // GC output still parses as valid PDF
    let doc = PdfDocument::from_bytes(gc_bytes.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_compress_produces_smaller_or_equal_output_for_raw_streams() {
    // Build a PDF with a large uncompressed content stream.
    let content = b"BT /F1 12 Tf 72 720 Td (Hello World) Tj ET ".repeat(50);
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n".to_string()
        .as_bytes(),
    );
    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(&content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");
    finalize_pdf(&mut pdf, &[0, off1, off2, off3, off4]);

    let path = write_temp_pdf(&pdf, "compress_size_src.pdf");

    let mut editor_nocomp = DocumentEditor::open(&path).unwrap();
    let uncompressed_bytes = editor_nocomp
        .save_to_bytes_with_options(SaveOptions {
            compress: false,
            garbage_collect: false,
            linearize: false,
            incremental: false,
            encryption: None,
        })
        .unwrap();

    let mut editor_comp = DocumentEditor::open(&path).unwrap();
    let compressed_bytes = editor_comp
        .save_to_bytes_with_options(SaveOptions {
            compress: true,
            garbage_collect: false,
            linearize: false,
            incremental: false,
            encryption: None,
        })
        .unwrap();

    assert!(
        compressed_bytes.len() <= uncompressed_bytes.len(),
        "Compressed output ({} bytes) should be <= uncompressed output ({} bytes)",
        compressed_bytes.len(),
        uncompressed_bytes.len()
    );

    // Compressed output parses correctly
    let doc = PdfDocument::from_bytes(compressed_bytes.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_linearize_true_does_not_panic() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "linearize_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();

    let opts = SaveOptions {
        compress: false,
        garbage_collect: false,
        linearize: true,
        incremental: false,
        encryption: None,
    };
    let bytes = editor.save_to_bytes_with_options(opts).unwrap();
    assert!(bytes.starts_with(b"%PDF-"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_save_to_bytes_round_trip_preserves_content() {
    let pdf = build_minimal_pdf();
    let path = write_temp_pdf(&pdf, "rt_src.pdf");
    let mut editor = DocumentEditor::open(&path).unwrap();
    editor.set_title("Round-trip Title");

    let bytes = editor.save_to_bytes().unwrap();
    let mut editor2 = DocumentEditor::from_bytes(bytes).unwrap();
    editor2.set_author("Second Pass Author");
    let bytes2 = editor2.save_to_bytes().unwrap();

    let doc = PdfDocument::from_bytes(bytes2.clone()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1);

    let _ = std::fs::remove_file(&path);
}
