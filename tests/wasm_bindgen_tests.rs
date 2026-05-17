//! Integration tests for WASM bindings using wasm-bindgen-test.
//!
//! These tests run in a real JS environment (Node.js/browser) and can fully
//! inspect JsValue contents via js_sys::Reflect. They cover what native tests
//! cannot — structured extraction, search results, and error paths.
//!
//! Run with: wasm-pack test --headless --node --features wasm

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

use pdf_oxide::api::{Pdf, PdfBuilder};
use pdf_oxide::geometry::Rect;
use pdf_oxide::wasm::{WasmDocumentBuilder, WasmEmbeddedFont, WasmPdf, WasmPdfDocument};
use pdf_oxide::writer::{CheckboxWidget, ComboBoxWidget, PdfWriter, TextFieldWidget};

wasm_bindgen_test_configure!(run_in_browser);

// ============================================================================
// Test Helpers
// ============================================================================

fn make_text_pdf(text: &str) -> Vec<u8> {
    Pdf::from_text(text).unwrap().into_bytes()
}

fn doc_from_text(text: &str) -> WasmPdfDocument {
    WasmPdfDocument::new(&make_text_pdf(text)).unwrap()
}

// ============================================================================
// Constructor — error paths
// ============================================================================

#[wasm_bindgen_test]
fn test_new_invalid_bytes() {
    let result = WasmPdfDocument::new(b"not a pdf");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_new_empty_bytes() {
    let result = WasmPdfDocument::new(b"");
    assert!(result.is_err());
}

// ============================================================================
// Structured Extraction — inspect JsValue contents
// ============================================================================

#[wasm_bindgen_test]
fn test_extract_chars_returns_array() {
    let mut doc = doc_from_text("ABC");
    let result = doc.extract_chars(0).unwrap();
    assert!(js_sys::Array::is_array(&result), "extract_chars should return an array");
    let arr = js_sys::Array::from(&result);
    assert!(arr.length() > 0, "should have at least one char");

    // Inspect first char object
    let first = arr.get(0);
    let char_val = js_sys::Reflect::get(&first, &JsValue::from_str("char")).unwrap();
    assert!(char_val.is_string(), "char field should be a string");
}

#[wasm_bindgen_test]
fn test_extract_chars_has_bbox() {
    let mut doc = doc_from_text("X");
    let result = doc.extract_chars(0).unwrap();
    let arr = js_sys::Array::from(&result);
    let first = arr.get(0);
    let bbox = js_sys::Reflect::get(&first, &JsValue::from_str("bbox")).unwrap();
    assert!(!bbox.is_undefined(), "char should have a bbox field");
}

#[wasm_bindgen_test]
fn test_extract_chars_has_font_name() {
    let mut doc = doc_from_text("A");
    let result = doc.extract_chars(0).unwrap();
    let arr = js_sys::Array::from(&result);
    let first = arr.get(0);
    let font = js_sys::Reflect::get(&first, &JsValue::from_str("font_name")).unwrap();
    assert!(!font.is_undefined(), "char should have a font_name field");
}

#[wasm_bindgen_test]
fn test_extract_chars_invalid_page() {
    let mut doc = doc_from_text("ABC");
    let result = doc.extract_chars(999);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_extract_spans_returns_array() {
    let mut doc = doc_from_text("Hello spans test");
    let result = doc.extract_spans(0).unwrap();
    assert!(js_sys::Array::is_array(&result), "extract_spans should return an array");
    let arr = js_sys::Array::from(&result);
    assert!(arr.length() > 0, "should have at least one span");

    // Inspect first span
    let first = arr.get(0);
    let text = js_sys::Reflect::get(&first, &JsValue::from_str("text")).unwrap();
    assert!(text.is_string(), "span should have a text field");
}

#[wasm_bindgen_test]
fn test_extract_spans_has_font_size() {
    let mut doc = doc_from_text("Hello");
    let result = doc.extract_spans(0).unwrap();
    let arr = js_sys::Array::from(&result);
    let first = arr.get(0);
    let font_size = js_sys::Reflect::get(&first, &JsValue::from_str("font_size")).unwrap();
    assert!(!font_size.is_undefined(), "span should have font_size");
}

// ============================================================================
// Search — inspect JsValue result structure
// ============================================================================

#[wasm_bindgen_test]
fn test_search_returns_array() {
    let mut doc = doc_from_text("Hello world search test");
    let result = doc.search("Hello", None, Some(true), None, None).unwrap();
    assert!(js_sys::Array::is_array(&result), "search should return an array");
}

#[wasm_bindgen_test]
fn test_search_result_has_fields() {
    let mut doc = doc_from_text("Hello world");
    let result = doc.search("Hello", None, Some(true), None, None).unwrap();
    let arr = js_sys::Array::from(&result);
    if arr.length() > 0 {
        let first = arr.get(0);
        let page = js_sys::Reflect::get(&first, &JsValue::from_str("page")).unwrap();
        assert!(!page.is_undefined(), "search result should have page field");
        let text = js_sys::Reflect::get(&first, &JsValue::from_str("text")).unwrap();
        assert!(!text.is_undefined(), "search result should have text field");
    }
}

#[wasm_bindgen_test]
fn test_search_not_found_empty_array() {
    let mut doc = doc_from_text("Hello world");
    let result = doc
        .search("ZZZZZ_NONEXISTENT", None, Some(true), None, None)
        .unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 0, "search for nonexistent text should return empty array");
}

#[wasm_bindgen_test]
fn test_search_page() {
    let mut doc = doc_from_text("Hello page search");
    let result = doc
        .search_page(0, "Hello", None, Some(true), None, None)
        .unwrap();
    assert!(js_sys::Array::is_array(&result));
}

#[wasm_bindgen_test]
fn test_search_case_insensitive() {
    let mut doc = doc_from_text("Hello World");
    let result = doc
        .search("hello", Some(true), Some(true), None, None)
        .unwrap();
    let arr = js_sys::Array::from(&result);
    assert!(arr.length() > 0, "case-insensitive search should find 'hello' in 'Hello World'");
}

// ============================================================================
// Image Info — inspect JsValue structure
// ============================================================================

#[wasm_bindgen_test]
fn test_extract_images_returns_array() {
    let mut doc = doc_from_text("No images");
    let result = doc.extract_images(0).unwrap();
    assert!(js_sys::Array::is_array(&result));
    let arr = js_sys::Array::from(&result);
    // Text-only PDF — expect 0 images
    assert_eq!(arr.length(), 0, "text-only PDF should have no images");
}

#[wasm_bindgen_test]
fn test_extract_images_invalid_page() {
    let mut doc = doc_from_text("Hello");
    let result = doc.extract_images(999);
    assert!(result.is_err());
}

// ============================================================================
// Page properties — JsValue paths
// ============================================================================

#[wasm_bindgen_test]
fn test_page_crop_box_null_when_unset() {
    let mut doc = doc_from_text("Hello");
    let result = doc.page_crop_box(0).unwrap();
    // CropBox is typically not set on generated PDFs
    if result.is_null() {
        // Expected: no crop box
    } else {
        // Some PDFs may set CropBox equal to MediaBox
        assert!(js_sys::Array::is_array(&result));
    }
}

#[wasm_bindgen_test]
fn test_page_rotation_invalid_page() {
    let mut doc = doc_from_text("Hello");
    let result = doc.page_rotation(999);
    assert!(result.is_err());
}

// ============================================================================
// Erase — error validation
// ============================================================================

#[wasm_bindgen_test]
fn test_erase_regions_invalid_length() {
    let mut doc = doc_from_text("Hello");
    let rects = [0.0, 0.0, 100.0]; // Not a multiple of 4
    let result = doc.erase_regions(0, &rects);
    assert!(result.is_err());
}

// ============================================================================
// Page Images — inspect structure
// ============================================================================

#[wasm_bindgen_test]
fn test_page_images_returns_array() {
    let mut doc = doc_from_text("Hello");
    let result = doc.page_images(0).unwrap();
    assert!(js_sys::Array::is_array(&result));
}

// ============================================================================
// Text extraction — error paths
// ============================================================================

#[wasm_bindgen_test]
fn test_extract_text_invalid_page() {
    let mut doc = doc_from_text("Hello");
    let result = doc.extract_text(999);
    assert!(result.is_err());
}

// ============================================================================
// Full roundtrip: create → edit → save → reopen → verify
// ============================================================================

#[wasm_bindgen_test]
fn test_full_roundtrip() {
    // Create a PDF
    let mut doc = doc_from_text("Roundtrip WASM test");

    // Edit metadata
    doc.set_title("WASM Title").unwrap();
    doc.set_author("WASM Author").unwrap();

    // Set rotation
    doc.set_page_rotation(0, 90).unwrap();

    // Save
    let bytes = doc.save_to_bytes().unwrap();
    assert!(bytes.starts_with(b"%PDF"));

    // Reopen
    let mut doc2 = WasmPdfDocument::new(&bytes).unwrap();
    assert_eq!(doc2.page_count().unwrap(), 1);

    // Verify text preserved
    let text = doc2.extract_text(0).unwrap();
    assert!(text.contains("Roundtrip"), "text should survive roundtrip");

    // Verify rotation preserved
    let rotation = doc2.page_rotation(0).unwrap();
    assert_eq!(rotation, 90, "rotation should survive roundtrip");
}

#[wasm_bindgen_test]
fn test_encrypted_roundtrip() {
    let mut doc = doc_from_text("Encrypted content");
    let bytes = doc
        .save_encrypted_to_bytes("mypass", None, None, None, None, None)
        .unwrap();
    assert!(bytes.starts_with(b"%PDF"));

    // Reopen and authenticate
    let mut doc2 = WasmPdfDocument::new(&bytes).unwrap();
    let auth = doc2.authenticate("mypass").unwrap();
    assert!(auth, "should authenticate with correct password");
}

// ============================================================================
// WasmPdf creation — verify content
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_pdf_from_markdown_roundtrip() {
    let pdf = WasmPdf::from_markdown("# Hello\n\nWorld content", None, None).unwrap();
    let mut doc = WasmPdfDocument::new(&pdf.to_bytes()).unwrap();
    let text = doc.extract_all_text().unwrap();
    assert!(!text.is_empty());
}

#[wasm_bindgen_test]
fn test_wasm_pdf_from_html_roundtrip() {
    let pdf = WasmPdf::from_html("<p>HTML content here</p>", None, None).unwrap();
    let mut doc = WasmPdfDocument::new(&pdf.to_bytes()).unwrap();
    let text = doc.extract_all_text().unwrap();
    assert!(!text.is_empty());
}

#[wasm_bindgen_test]
fn test_wasm_pdf_from_text_roundtrip() {
    let pdf = WasmPdf::from_text("Plain text content", None, None).unwrap();
    let mut doc = WasmPdfDocument::new(&pdf.to_bytes()).unwrap();
    let text = doc.extract_all_text().unwrap();
    assert!(text.contains("Plain"), "should contain source text");
}

// ============================================================================
// Outline, Annotations, Paths — new WASM bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_get_outline_returns_null_or_array() {
    let mut doc = doc_from_text("No outline here");
    let result = doc.get_outline().unwrap();
    // Text-only generated PDF has no outline, expect null
    assert!(
        result.is_null() || js_sys::Array::is_array(&result),
        "getOutline should return null or an array"
    );
}

#[wasm_bindgen_test]
fn test_get_annotations_returns_array() {
    let mut doc = doc_from_text("No annotations");
    let result = doc.get_annotations(0).unwrap();
    assert!(js_sys::Array::is_array(&result), "getAnnotations should return an array");
    let arr = js_sys::Array::from(&result);
    // Text-only PDF — expect 0 annotations
    assert_eq!(arr.length(), 0, "text-only PDF should have no annotations");
}

#[wasm_bindgen_test]
fn test_get_annotations_invalid_page() {
    let mut doc = doc_from_text("Hello");
    let result = doc.get_annotations(999);
    assert!(result.is_err(), "invalid page should return error");
}

#[wasm_bindgen_test]
fn test_extract_paths_returns_array() {
    let mut doc = doc_from_text("No paths");
    let result = doc.extract_paths(0).unwrap();
    assert!(js_sys::Array::is_array(&result), "extractPaths should return an array");
}

#[wasm_bindgen_test]
fn test_extract_paths_invalid_page() {
    let mut doc = doc_from_text("Hello");
    let result = doc.extract_paths(999);
    assert!(result.is_err(), "invalid page should return error");
}

// ============================================================================
// PDF creation — metadata verification
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_pdf_metadata() {
    let pdf = WasmPdf::from_text(
        "With metadata",
        Some("My Title".to_string()),
        Some("My Author".to_string()),
    )
    .unwrap();
    assert!(pdf.size() > 0);
    let bytes = pdf.to_bytes();
    assert!(bytes.starts_with(b"%PDF"));
}

// ============================================================================
// Form Fields (Issue #172) — getFormFields, hasXfa
// ============================================================================

/// Create a PDF with form fields for WASM testing.
fn make_form_pdf() -> Vec<u8> {
    let mut writer = PdfWriter::new();
    {
        let mut page = writer.add_page(612.0, 792.0);
        page.add_text_field(
            TextFieldWidget::new("name", Rect::new(72.0, 700.0, 200.0, 20.0)).with_value("Alice"),
        );
        page.add_checkbox(
            CheckboxWidget::new("agree", Rect::new(72.0, 650.0, 15.0, 15.0)).checked(),
        );
        page.add_combo_box(
            ComboBoxWidget::new("color", Rect::new(72.0, 600.0, 150.0, 20.0))
                .with_options(vec!["Red", "Blue", "Green"])
                .with_value("Blue"),
        );
    }
    writer.finish().expect("Failed to create form PDF")
}

#[wasm_bindgen_test]
fn test_get_form_fields_returns_array() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    let result = doc.get_form_fields().unwrap();
    assert!(js_sys::Array::is_array(&result), "getFormFields should return an array");
    let arr = js_sys::Array::from(&result);
    assert!(arr.length() >= 3, "Should have at least 3 form fields, got {}", arr.length());
}

#[wasm_bindgen_test]
fn test_get_form_fields_has_name_and_type() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    let result = doc.get_form_fields().unwrap();
    let arr = js_sys::Array::from(&result);

    // Check first field has name and field_type
    let first = arr.get(0);
    let name = js_sys::Reflect::get(&first, &JsValue::from_str("name")).unwrap();
    assert!(name.is_string(), "field should have a string 'name'");
    let ft = js_sys::Reflect::get(&first, &JsValue::from_str("field_type")).unwrap();
    assert!(ft.is_string(), "field should have a string 'field_type'");
}

#[wasm_bindgen_test]
fn test_get_form_fields_has_value() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    let result = doc.get_form_fields().unwrap();
    let arr = js_sys::Array::from(&result);

    // Find the text field — it should have a string value
    for i in 0..arr.length() {
        let field = arr.get(i);
        let ft = js_sys::Reflect::get(&field, &JsValue::from_str("field_type")).unwrap();
        if ft.as_string().as_deref() == Some("text") {
            let value = js_sys::Reflect::get(&field, &JsValue::from_str("value")).unwrap();
            assert!(value.is_string(), "text field should have a string value");
            return;
        }
    }
    // If no text field found, the test is inconclusive (but it should find one)
}

#[wasm_bindgen_test]
fn test_get_form_fields_empty_on_plain_pdf() {
    let mut doc = doc_from_text("No forms here");
    let result = doc.get_form_fields().unwrap();
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 0, "Plain text PDF should have no form fields");
}

#[wasm_bindgen_test]
fn test_has_xfa_false_on_plain_pdf() {
    let mut doc = doc_from_text("No XFA");
    let result = doc.has_xfa().unwrap();
    assert!(!result, "Plain text PDF should not have XFA");
}

#[wasm_bindgen_test]
fn test_has_xfa_false_on_acroform_pdf() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    let result = doc.has_xfa().unwrap();
    assert!(!result, "PdfWriter-created form should not have XFA");
}

// ============================================================================
// Form Field Get/Set Values — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_set_and_get_form_field_value() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();

    // Set a text field value
    doc.set_form_field_value("name", JsValue::from_str("Bob"))
        .unwrap();

    // Get it back
    let result = doc.get_form_field_value("name").unwrap();
    assert!(result.is_string(), "text field value should be a string");
    assert_eq!(result.as_string().unwrap(), "Bob");
}

#[wasm_bindgen_test]
fn test_set_checkbox_form_field() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();

    // Set checkbox to true
    doc.set_form_field_value("agree", JsValue::from(true))
        .unwrap();

    // Get it back
    let result = doc.get_form_field_value("agree").unwrap();
    assert_eq!(result.as_bool(), Some(true), "checkbox should be true");
}

#[wasm_bindgen_test]
fn test_get_form_field_value_not_found() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();

    let result = doc.get_form_field_value("nonexistent_field").unwrap();
    assert!(result.is_null(), "non-existent field should return null");
}

// ============================================================================
// Image Bytes Extraction — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_extract_image_bytes_empty() {
    let mut doc = doc_from_text("No images");
    let result = doc.extract_image_bytes(0).unwrap();
    assert!(js_sys::Array::is_array(&result), "should return an array");
    let arr = js_sys::Array::from(&result);
    assert_eq!(arr.length(), 0, "text-only PDF should have no images");
}

// ============================================================================
// PDF from Images — new bindings
// ============================================================================

/// Create a minimal valid 1x1 white JPEG image (known-good bytes).
fn create_minimal_image() -> Vec<u8> {
    vec![
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
        0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B,
        0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31,
        0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF,
        0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00,
        0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05,
        0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
        0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A,
        0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
        0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
        0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93,
        0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9,
        0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6,
        0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
        0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5,
        0xDB, 0x20, 0xA8, 0xF9, 0xFF, 0xD9,
    ]
}

#[wasm_bindgen_test]
fn test_pdf_from_image_bytes() {
    let png = create_minimal_image();
    let result = WasmPdf::from_image_bytes(&png);
    assert!(result.is_ok(), "fromImageBytes should succeed with valid PNG");
    let pdf = result.unwrap();
    assert!(pdf.size() > 0, "PDF should have content");

    // Verify it's a valid PDF we can reopen
    let mut doc = WasmPdfDocument::new(&pdf.to_bytes()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 1, "should have 1 page");
}

#[wasm_bindgen_test]
fn test_pdf_from_multiple_image_bytes() {
    let png1 = create_minimal_image();
    let png2 = create_minimal_image();
    let arr = js_sys::Array::new();
    arr.push(&js_sys::Uint8Array::from(png1.as_slice()));
    arr.push(&js_sys::Uint8Array::from(png2.as_slice()));

    let result = WasmPdf::from_multiple_image_bytes(arr.into());
    assert!(result.is_ok(), "fromMultipleImageBytes should succeed");
    let pdf = result.unwrap();

    let mut doc = WasmPdfDocument::new(&pdf.to_bytes()).unwrap();
    assert_eq!(doc.page_count().unwrap(), 2, "should have 2 pages");
}

// ============================================================================
// Form Flattening — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_flatten_forms() {
    let bytes = make_form_pdf();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();

    // Verify we have form fields before flattening
    let fields_before = doc.get_form_fields().unwrap();
    let arr_before = js_sys::Array::from(&fields_before);
    assert!(arr_before.length() >= 3, "should have fields before flatten");

    // Flatten
    doc.flatten_forms().unwrap();

    // After flatten + save/reload, form fields should be gone
    let saved = doc.save_to_bytes().unwrap();
    let mut doc2 = WasmPdfDocument::new(&saved).unwrap();
    let fields_after = doc2.get_form_fields().unwrap();
    let arr_after = js_sys::Array::from(&fields_after);
    assert_eq!(arr_after.length(), 0, "should have no fields after flatten");
}

// ============================================================================
// PDF Merging — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_merge_from() {
    let bytes1 = make_text_pdf("Document 1");
    let bytes2 = make_text_pdf("Document 2");
    let mut doc = WasmPdfDocument::new(&bytes1).unwrap();

    let count = doc.merge_from(&bytes2).unwrap();
    assert_eq!(count, 1, "should merge 1 page");
}

// ============================================================================
// File Embedding — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_embed_file() {
    let mut doc = doc_from_text("Hello");
    doc.embed_file("test.txt", b"Hello embedded").unwrap();

    // Should be able to save without error
    let bytes = doc.save_to_bytes().unwrap();
    assert!(bytes.starts_with(b"%PDF"));
}

// ============================================================================
// Page Labels — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_page_labels_empty() {
    let mut doc = doc_from_text("Hello");
    let result = doc.page_labels().unwrap();
    assert!(js_sys::Array::is_array(&result), "should return an array");
}

// ============================================================================
// XMP Metadata — new bindings
// ============================================================================

#[wasm_bindgen_test]
fn test_xmp_metadata_null_or_object() {
    let mut doc = doc_from_text("Hello");
    let result = doc.xmp_metadata().unwrap();
    // Simple generated PDF may or may not have XMP
    assert!(
        result.is_null() || result.is_object(),
        "xmpMetadata should return null or an object"
    );
}

// ============================================================================
// Write-side API — WasmDocumentBuilder + WasmEmbeddedFont round-trip
// ============================================================================
//
// Ports the Python reference test (tests/test_python_document_builder.py)
// to the WASM binding. Confirms the fluent API compiles down correctly,
// that CJK-adjacent scripts (Cyrillic + Greek) round-trip through
// `extract_text`, and that the subset_font_bytes pipeline (v0.3.38 #385)
// is wired through the WASM path (PDF size << original face size).

/// DejaVuSans TTF fixture bytes — covers Latin + Cyrillic + Greek, small
/// enough (~760 KB) to include in the wasm test binary without blowing
/// out the bundle size excessively.
const DEJAVU_TTF: &[u8] = include_bytes!("fixtures/fonts/DejaVuSans.ttf");

#[wasm_bindgen_test]
fn document_builder_minimal_ascii() {
    let mut b = WasmDocumentBuilder::new();
    b.title("Hello".to_string()).unwrap();
    let mut p = b.a4_page().unwrap();
    p.at(72.0, 720.0).unwrap();
    p.text("Hello, world.".to_string()).unwrap();
    p.done(&mut b).unwrap();
    let bytes = b.build().unwrap();
    assert!(bytes.starts_with(b"%PDF-"));
    assert!(bytes.len() > 256);
}

#[wasm_bindgen_test]
fn document_builder_cjk_round_trip() {
    let mut font = WasmEmbeddedFont::from_bytes(DEJAVU_TTF, None).unwrap();
    let mut b = WasmDocumentBuilder::new();
    b.register_embedded_font("DejaVu".to_string(), &mut font)
        .unwrap();
    let mut p = b.a4_page().unwrap();
    p.font("DejaVu".to_string(), 12.0).unwrap();
    p.at(72.0, 720.0).unwrap();
    p.text("Привет, мир!".to_string()).unwrap();
    p.at(72.0, 700.0).unwrap();
    p.text("Καλημέρα κόσμε".to_string()).unwrap();
    p.done(&mut b).unwrap();

    let pdf_bytes = b.build().unwrap();
    let doc = WasmPdfDocument::new(&pdf_bytes).unwrap();
    let text = doc.extract_text(0).unwrap();
    assert!(text.contains("Привет, мир!"), "Cyrillic round-trip failed: {text:?}");
    assert!(text.contains("Καλημέρα κόσμε"), "Greek round-trip failed: {text:?}");
}

#[wasm_bindgen_test]
fn document_builder_output_is_subsetted() {
    let mut font = WasmEmbeddedFont::from_bytes(DEJAVU_TTF, None).unwrap();
    let mut b = WasmDocumentBuilder::new();
    b.register_embedded_font("DejaVu".to_string(), &mut font)
        .unwrap();
    let mut p = b.a4_page().unwrap();
    p.font("DejaVu".to_string(), 12.0).unwrap();
    p.at(72.0, 700.0).unwrap();
    p.text("Hello world".to_string()).unwrap();
    p.done(&mut b).unwrap();
    let bytes = b.build().unwrap();

    // v0.3.38 #385 wires real font subsetting; PDF must be much smaller
    // than the embedded face (~760 KB).
    assert!(
        bytes.len() * 10 < DEJAVU_TTF.len(),
        "expected PDF ({} bytes) to be >= 10× smaller than the face ({} bytes)",
        bytes.len(),
        DEJAVU_TTF.len(),
    );
}

#[wasm_bindgen_test]
fn document_builder_to_bytes_encrypted() {
    let mut b = WasmDocumentBuilder::new();
    let mut p = b.a4_page().unwrap();
    p.at(72.0, 720.0).unwrap();
    p.text("secret".to_string()).unwrap();
    p.done(&mut b).unwrap();
    let bytes = b.to_bytes_encrypted("userpw", "ownerpw").unwrap();
    // Locate /Encrypt + /V 5 markers (AES-256).
    let pdf_str = String::from_utf8_lossy(&bytes);
    assert!(pdf_str.contains("/Encrypt"), "encrypted PDF missing /Encrypt dict");
    assert!(pdf_str.contains("/V 5"), "expected /V 5 (AES-256) marker");
}

#[wasm_bindgen_test]
fn document_builder_consumed_after_build() {
    let mut b = WasmDocumentBuilder::new();
    let mut p = b.a4_page().unwrap();
    p.at(72.0, 720.0).unwrap();
    p.text("x".to_string()).unwrap();
    p.done(&mut b).unwrap();
    let _ = b.build().unwrap();
    // Second build should fail — builder is consumed.
    assert!(b.build().is_err(), "second build should return Err");
}

#[wasm_bindgen_test]
fn fluent_page_done_is_single_use() {
    let mut b = WasmDocumentBuilder::new();
    let mut p = b.a4_page().unwrap();
    p.text("a".to_string()).unwrap();
    p.done(&mut b).unwrap();
    // Second done on the same page should error.
    assert!(p.done(&mut b).is_err(), "double done should return Err");
}

#[wasm_bindgen_test]
fn embedded_font_consumed_after_register() {
    let mut font = WasmEmbeddedFont::from_bytes(DEJAVU_TTF, None).unwrap();
    let mut b = WasmDocumentBuilder::new();
    b.register_embedded_font("DejaVu1".to_string(), &mut font)
        .unwrap();
    assert_eq!(font.name(), "", "font handle should be empty after consumption");
    // Second register using the already-consumed handle should error.
    assert!(
        b.register_embedded_font("DejaVu2".to_string(), &mut font)
            .is_err(),
        "re-registering a consumed font should return Err",
    );
}

// ----------------------------------------------------------------------
// Phase 2 — HTML+CSS pipeline on WasmPdf
// ----------------------------------------------------------------------

#[wasm_bindgen_test]
fn wasm_pdf_from_html_css() {
    let pdf = WasmPdf::from_html_css(
        "<h1>Hello</h1><p>World</p>",
        "h1 { color: blue; font-size: 24pt }",
        DEJAVU_TTF,
    )
    .unwrap();
    let bytes = pdf.to_bytes();
    assert!(bytes.starts_with(b"%PDF-"));
    let doc = WasmPdfDocument::new(&bytes).unwrap();
    let text = doc.extract_text(0).unwrap();
    assert!(text.contains("Hello"));
    assert!(text.contains("World"));
}

#[wasm_bindgen_test]
fn wasm_pdf_from_html_css_with_fonts_requires_non_empty() {
    let result = WasmPdf::from_html_css_with_fonts("<p>x</p>", "", Vec::new(), Vec::new());
    assert!(result.is_err(), "empty font list should be rejected");
}

// ----------------------------------------------------------------------
// WASM move_page — DocumentEditor mutation parity with
// Python / Go / C# / Node.
// ----------------------------------------------------------------------

#[wasm_bindgen_test]
fn wasm_pdf_document_move_page_preserves_page_count() {
    // Three-page PDF with distinct content per page so the reorder is observable.
    let mut b = WasmDocumentBuilder::new();
    for tag in &["alpha", "beta", "gamma"] {
        let mut p = b.a4_page().unwrap();
        p.at(72.0, 720.0).unwrap();
        p.text(tag.to_string()).unwrap();
        p.done(&mut b).unwrap();
    }
    let bytes = b.build().unwrap();

    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    assert_eq!(doc.page_count().unwrap(), 3);

    // Move the first page to the end. No error = plumbing works.
    doc.move_page(0, 2).unwrap();
    assert_eq!(doc.page_count().unwrap(), 3, "move_page must not change the page count",);

    // Out-of-range move must surface as an Err, not panic.
    assert!(doc.move_page(99, 0).is_err(), "out-of-range from_index should return Err",);
}

// ----------------------------------------------------------------------
// v0.3.39 — issue #393 DocumentBuilder tables + primitives
// ----------------------------------------------------------------------

#[wasm_bindgen_test]
fn fluent_page_text_in_rect_and_strokes_round_trip() {
    let mut b = WasmDocumentBuilder::new();
    let mut p = b.letter_page().unwrap();
    p.font("Helvetica".to_string(), 10.0).unwrap();
    p.text_in_rect(72.0, 700.0, 200.0, 100.0, "wrapped text".to_string(), 1 /* Center */)
        .unwrap();
    p.stroke_rect(50.0, 500.0, 200.0, 100.0, 2.0, 0.5, 0.5, 0.5)
        .unwrap();
    p.stroke_line(50.0, 400.0, 250.0, 400.0, 1.0, 0.2, 0.2, 0.2)
        .unwrap();
    p.done(&mut b).unwrap();
    let bytes = b.build().unwrap();
    assert!(bytes.starts_with(b"%PDF-"));
}

#[wasm_bindgen_test]
fn fluent_page_measure_nonzero_and_remaining_space() {
    let mut b = WasmDocumentBuilder::new();
    let mut p = b.letter_page().unwrap();
    p.font("Helvetica".to_string(), 12.0).unwrap();
    let w = p.measure("Hello");
    assert!(w > 0.0, "measure should return a positive width, got {w}");
    // Letter = 612x792, top-margin 72 puts cursor at 720.
    // Bottom margin 72 → remaining = 648.
    let r = p.remaining_space();
    assert!((r - 648.0).abs() < 1.0, "remainingSpace ≈ 648 at page start, got {r}");
    // Silence unused-must-use on p for later chaining.
    p.done(&mut b).unwrap();
    let _ = b.build().unwrap();
}

#[wasm_bindgen_test]
fn fluent_page_new_page_same_size_creates_second_page() {
    let mut b = WasmDocumentBuilder::new();
    let mut p = b.letter_page().unwrap();
    p.at(72.0, 720.0).unwrap();
    p.text("page-1".to_string()).unwrap();
    p.new_page_same_size().unwrap();
    p.at(72.0, 720.0).unwrap();
    p.text("page-2".to_string()).unwrap();
    p.done(&mut b).unwrap();
    let bytes = b.build().unwrap();
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    assert_eq!(doc.page_count().unwrap(), 2, "newPageSameSize must add a page");
}

#[wasm_bindgen_test]
fn fluent_page_buffered_table_from_js_object() {
    use wasm_bindgen::JsValue;

    let mut b = WasmDocumentBuilder::new();
    let mut p = b.letter_page().unwrap();
    p.font("Helvetica".to_string(), 10.0).unwrap();
    p.at(72.0, 720.0).unwrap();

    // Construct a JS object equivalent to:
    //   { columns: [{header:"SKU", width:100, align:0}, {header:"Qty", width:60, align:2}],
    //     rows: [["A-1","12"], ["B-2","3"]],
    //     hasHeader: true }
    let spec = js_sys::Object::new();
    let columns = js_sys::Array::new();
    for (header, width, align) in [("SKU", 100.0_f64, 0_i32), ("Qty", 60.0, 2)].iter() {
        let col = js_sys::Object::new();
        js_sys::Reflect::set(&col, &JsValue::from_str("header"), &JsValue::from_str(header))
            .unwrap();
        js_sys::Reflect::set(&col, &JsValue::from_str("width"), &JsValue::from_f64(*width))
            .unwrap();
        js_sys::Reflect::set(&col, &JsValue::from_str("align"), &JsValue::from_f64(*align as f64))
            .unwrap();
        columns.push(&col);
    }
    js_sys::Reflect::set(&spec, &JsValue::from_str("columns"), &columns).unwrap();

    let rows = js_sys::Array::new();
    for row in [["A-1", "12"], ["B-2", "3"]].iter() {
        let r = js_sys::Array::new();
        for cell in row.iter() {
            r.push(&JsValue::from_str(cell));
        }
        rows.push(&r);
    }
    js_sys::Reflect::set(&spec, &JsValue::from_str("rows"), &rows).unwrap();
    js_sys::Reflect::set(&spec, &JsValue::from_str("hasHeader"), &JsValue::from_bool(true))
        .unwrap();

    p.table(spec.into()).unwrap();
    p.done(&mut b).unwrap();
    let bytes = b.build().unwrap();
    assert!(bytes.starts_with(b"%PDF-"));
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    let text = doc.extract_all_text().unwrap();
    // At minimum the header texts should survive the round-trip.
    assert!(
        text.contains("SKU") || text.contains("Qty"),
        "buffered-table headers missing from extracted text: {text:?}",
    );
}

#[wasm_bindgen_test]
fn fluent_page_streaming_table_push_and_finish() {
    use wasm_bindgen::JsValue;

    let mut b = WasmDocumentBuilder::new();
    let mut p = b.letter_page().unwrap();
    p.font("Helvetica".to_string(), 10.0).unwrap();
    p.at(72.0, 720.0).unwrap();

    // { columns: [{header:"SKU", width:72}, {header:"Item", width:200}, {header:"Qty", width:48, align:2}],
    //   repeatHeader: true }
    let spec = js_sys::Object::new();
    let columns = js_sys::Array::new();
    for (header, width, align) in [
        ("SKU", 72.0_f64, 0_i32),
        ("Item", 200.0, 0),
        ("Qty", 48.0, 2),
    ]
    .iter()
    {
        let col = js_sys::Object::new();
        js_sys::Reflect::set(&col, &JsValue::from_str("header"), &JsValue::from_str(header))
            .unwrap();
        js_sys::Reflect::set(&col, &JsValue::from_str("width"), &JsValue::from_f64(*width))
            .unwrap();
        js_sys::Reflect::set(&col, &JsValue::from_str("align"), &JsValue::from_f64(*align as f64))
            .unwrap();
        columns.push(&col);
    }
    js_sys::Reflect::set(&spec, &JsValue::from_str("columns"), &columns).unwrap();
    js_sys::Reflect::set(&spec, &JsValue::from_str("repeatHeader"), &JsValue::from_bool(true))
        .unwrap();

    let mut t = p.streaming_table(spec.into()).unwrap();
    assert_eq!(t.column_count(), 3);
    for i in 0..3 {
        t.push_row(vec![format!("A-{i}"), "Widget".into(), (i * 10).to_string()])
            .unwrap();
    }
    // Wrong-arity row → error, not panic.
    assert!(
        t.push_row(vec!["only-one".to_string()]).is_err(),
        "wrong-arity row should return Err",
    );
    t.finish().unwrap();
    // finish() twice throws.
    assert!(t.finish().is_err(), "double finish should return Err");

    p.done(&mut b).unwrap();
    let bytes = b.build().unwrap();
    assert!(bytes.starts_with(b"%PDF-"));
    let mut doc = WasmPdfDocument::new(&bytes).unwrap();
    let text = doc.extract_all_text().unwrap();
    assert!(
        text.contains("SKU") || text.contains("Item") || text.contains("Qty"),
        "streaming-table headers missing from extracted text: {text:?}",
    );
}

// v0.3.50 parity (#235): the document-scoped PAdES-B-LTA reader signal
// is exposed to WASM as `hasDocumentTimestamp`. A freshly-built PDF has
// no /DocTimeStamp, so the binding must return `false` (not throw).
#[cfg(feature = "signatures")]
#[wasm_bindgen_test]
fn test_has_document_timestamp_plain_pdf_is_false() {
    let bytes = make_text_pdf("plain document, no archival timestamp");
    assert!(!pdf_oxide::wasm::wasm_has_document_timestamp(&bytes));
}
