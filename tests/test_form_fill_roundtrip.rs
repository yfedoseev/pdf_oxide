//! AcroForm fill → save → re-read round-trip (#616 mojibake, #617 lost /FT).
//!
//! Builds a one-field AcroForm with the engine's own `PdfWriter`, fills it via
//! `DocumentEditor`, saves, and re-parses — asserting the field survives and
//! non-ASCII values are encoded as conformant PDF text strings (UTF-16BE+BOM,
//! ISO 32000-1 §7.9.2.2). No external/MPL PDFs are used.

use pdf_oxide::editor::form_fields::FormFieldValue;
use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::extractors::forms::FormExtractor;
use pdf_oxide::geometry::Rect;
use pdf_oxide::writer::{PdfWriter, TextFieldWidget};
use pdf_oxide::PdfDocument;

fn one_field_form() -> Vec<u8> {
    let mut w = PdfWriter::new();
    {
        let mut p = w.add_page(612.0, 792.0);
        p.add_text_field(TextFieldWidget::new("full_name", Rect::new(72.0, 700.0, 400.0, 720.0)));
        p.finish();
    }
    w.finish().unwrap()
}

/// #617: a filled terminal field must remain classifiable after save.
#[test]
fn filled_field_survives_save_and_reextract() {
    // Sanity: the freshly-written form has exactly one field.
    let form = one_field_form();
    let doc0 = PdfDocument::from_bytes(form.clone()).unwrap();
    assert_eq!(
        FormExtractor::extract_fields(&doc0).unwrap().len(),
        1,
        "freshly-written form should have 1 field"
    );

    let mut ed = DocumentEditor::from_bytes(form).unwrap();
    ed.set_form_field_value("full_name", FormFieldValue::Text("John".into()))
        .unwrap();
    let out = ed.save_to_bytes().unwrap();

    let doc = PdfDocument::from_bytes(out).unwrap();
    let fields = FormExtractor::extract_fields(&doc).unwrap();
    assert_eq!(
        fields.len(),
        1,
        "field must survive fill+save (#617: /FT must be re-emitted on terminal fields)"
    );
}

/// #616 + #617: a Japanese value round-trips without mojibake.
#[test]
fn japanese_form_fill_roundtrip_no_mojibake() {
    let mut ed = DocumentEditor::from_bytes(one_field_form()).unwrap();
    ed.set_form_field_value("full_name", FormFieldValue::Text("山田太郎".into()))
        .unwrap();
    let out = ed.save_to_bytes().unwrap();

    // #616: /V is serialized as a PDF hex string of UTF-16BE + BOM
    // (`<FEFF5C71753059 2A90CE>`), never the raw UTF-8 (`E5B1B1…`). pdf_oxide
    // writes binary strings in hex form, so check the ASCII hex digits.
    fn contains_ascii_ci(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len())
            .any(|w| w.eq_ignore_ascii_case(needle))
    }
    assert!(
        contains_ascii_ci(&out, b"FEFF5C717530592A90CE"),
        "value must be a UTF-16BE+BOM hex string in the saved bytes"
    );
    assert!(
        !contains_ascii_ci(&out, b"E5B1B1E794B0E5A4AAE9838E"),
        "raw UTF-8 of the value must NOT appear (mojibake)"
    );

    // #617 + #616: re-read returns the field with the exact value.
    let doc = PdfDocument::from_bytes(out).unwrap();
    let fields = FormExtractor::extract_fields(&doc).unwrap();
    assert_eq!(fields.len(), 1, "field must survive fill+save");
    let v = format!("{:?}", fields[0].value);
    assert!(v.contains("山田太郎"), "re-read value must equal 山田太郎; got {v}");
}
